// Package ais: internal unit tests
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package ais

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NVIDIA/aistore/api/apc"
	"github.com/NVIDIA/aistore/cmn"
	"github.com/NVIDIA/aistore/cmn/cos"
	"github.com/NVIDIA/aistore/core"
	"github.com/NVIDIA/aistore/core/meta"
	"github.com/NVIDIA/aistore/tools/tassert"
)

type exactRetireBackend struct {
	core.Backend
	data []byte
}

func (*exactRetireBackend) MetricName(name string) string { return name }

func (bp *exactRetireBackend) HeadObj(context.Context, *core.LOM, *http.Request) (*cmn.ObjAttrs, int, error) {
	attrs := &cmn.ObjAttrs{Size: int64(len(bp.data))}
	attrs.SetCustomKey(cmn.ETag, "exact-retire-v1")
	return attrs, 0, nil
}

func (bp *exactRetireBackend) GetObjReader(_ context.Context, _ *core.LOM, offset, length int64, pins ...*core.ObjPin) core.GetReaderResult {
	if len(pins) != 1 || pins[0] == nil || pins[0].ETag != "exact-retire-v1" || pins[0].ExactSize != int64(len(bp.data)) {
		return core.GetReaderResult{Err: errors.New("missing exact range pin"), ErrCode: http.StatusBadGateway}
	}
	return core.GetReaderResult{R: io.NopCloser(bytes.NewReader(bp.data[offset : offset+length])), Size: length}
}

// exercise the exact helper and net/http connection reuse with the existing budget
// state. The software cases simulate an already-armed connection; the kernel cases
// install real TLS_TX, but seed a small budget instead of transferring hundreds of GiB.
func TestExactColdRangeRetire(t *testing.T) {
	bp := &exactRetireBackend{data: make([]byte, 256*cos.KiB)}
	for i := range bp.data {
		bp.data[i] = byte(i % 251)
	}
	oldBMD, oldBackends := mockTarget.owner.bmd.get(), mockTarget.bps
	t.Cleanup(func() { mockTarget.bps = oldBackends; mockTarget.owner.bmd.put(oldBMD) })
	// Reuse the remote-backend seam without changing global cloud configuration.
	mockTarget.bps = backends{apc.AIS: bp}
	bck := meta.NewBck("exact-retire", apc.AIS, cmn.Ns{UUID: "remote"})
	bmd := oldBMD.clone()
	bmd.add(bck, &cmn.Bprops{Cksum: cmn.CksumConf{Type: cos.ChecksumNone}})
	mockTarget.owner.bmd.put(bmd)

	for _, mode := range []string{"software", "kernel", "unarmed", "nil"} {
		t.Run(mode, func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				budget int64
				sizes  []int
				closes []bool
			}{
				{"reuse-then-retire", ktlsHeadroom + 513, []int{512, 128 * cos.KiB, 512}, []bool{false, true, false}},
				{"equal-boundary", ktlsHeadroom + 512, []int{512, 512}, []bool{true, true}},
				{"cross-budget", ktlsHeadroom + 512, []int{128 * cos.KiB, 512}, []bool{true, true}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					type connInfo struct {
						state *ktlsConn
						id    int64
					}
					type ctxKey struct{}
					var accepts atomic.Int64
					outcomes := make(chan error, len(tc.sizes))
					handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
						ci := req.Context().Value(ctxKey{}).(*connInfo)
						if mode == "kernel" && !ci.state.tryArm(ktlsMinSize) {
							// Surface unavailable TLS_TX to the test goroutine; never count fallback as kTLS.
							http.Error(w, "kTLS arming unavailable: "+ktlsCnt.String(), http.StatusServiceUnavailable)
							return
						}
						lom := &core.LOM{ObjName: "footer"}
						if err := lom.InitBck(bck); err != nil {
							outcomes <- err
							http.Error(w, err.Error(), 500)
							return
						}
						w.Header().Set("X-Test-Connection", strconv.FormatInt(ci.id, 10))
						goi := getOI{t: mockTarget, lom: lom, w: w, req: req, ranges: byteRanges{Range: req.Header.Get(cos.HdrRange)}}
						if mode != "nil" {
							goi.ktls = ci.state
						}
						handled, code, err := goi.tryExactColdRange(bp, int64(len(bp.data)))
						if err == nil && (!handled || code != 0 || !goi.rget) {
							err = fmt.Errorf("handled=%t code=%d rget=%t", handled, code, goi.rget)
						}
						outcomes <- err
						if mode == "software" {
							// Only body bytes are charged in the software seam; real kTLS charges wire writes.
							n, _ := strconv.ParseInt(w.Header().Get(cos.HdrContentLength), 10, 64)
							ci.state.txBytes.Add(n)
						}
					})
					server := httptest.NewUnstartedServer(handler)
					server.Config.ConnContext = func(ctx context.Context, nc net.Conn) context.Context {
						ci := &connInfo{id: accepts.Add(1)}
						if mode == "kernel" {
							ci.state = nc.(*ktlsConn)
						} else {
							ci.state = &ktlsConn{}
						}
						ci.state.txMaxBytes = tc.budget
						if mode == "software" {
							ci.state.txState.Store(ktlsArmed)
						}
						return context.WithValue(ctx, ctxKey{}, ci)
					}
					if mode == "kernel" {
						listener, err := newKtlsListener(server.Listener, testktlsServerConf(t), nil)
						tassert.CheckFatal(t, err)
						server.Listener = listener
					}
					server.Start()
					defer server.Close()
					url := server.URL
					transport := &http.Transport{TLSClientConfig: testktlsClientConf(nil)}
					defer transport.CloseIdleConnections()
					client := &http.Client{Transport: transport, Timeout: testktlsTimeout}
					if mode == "kernel" {
						url = strings.Replace(url, "http:", "https:", 1)
					}
					var previous string
					for i, size := range tc.sizes {
						req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, url, http.NoBody)
						tassert.CheckFatal(t, err)
						req.Header.Set(cos.HdrRange, fmt.Sprintf("bytes=13-%d", 12+size))
						resp, err := client.Do(req)
						tassert.CheckFatal(t, err)
						data, readErr := io.ReadAll(resp.Body)
						resp.Body.Close()
						if mode == "kernel" && resp.StatusCode == http.StatusServiceUnavailable {
							t.Skipf("real kTLS TX not verified: %s", strings.TrimSpace(string(data)))
						}
						tassert.CheckFatal(t, readErr)
						select {
						case err = <-outcomes:
							tassert.CheckFatal(t, err)
						case <-time.After(testktlsTimeout):
							t.Fatal("exact handler did not return")
						}
						wantClose := tc.closes[i] && (mode == "software" || mode == "kernel")
						tassert.Errorf(t, resp.StatusCode == http.StatusPartialContent, "status=%d", resp.StatusCode)
						tassert.Errorf(t, resp.Close == wantClose, "response %d: close=%t, want %t", i, resp.Close, wantClose)
						tassert.Errorf(t, bytes.Equal(data, bp.data[13:13+size]), "response %d: incorrect body (%d bytes)", i, len(data))
						tassert.Errorf(t, resp.Header.Get(cos.HdrContentRange) == fmt.Sprintf("bytes 13-%d/%d", 12+size, len(bp.data)), "incorrect content range")
						id := resp.Header.Get("X-Test-Connection")
						tassert.Fatalf(t, id != "", "missing connection identity")
						if i > 0 {
							wantNew := tc.closes[i-1] && (mode == "software" || mode == "kernel")
							tassert.Errorf(t, (id != previous) == wantNew, "response %d: connection %s -> %s, wantNew=%t", i, previous, id, wantNew)
						}
						t.Logf("response=%d bytes=%d close=%t connection=%s", i, len(data), resp.Close, id)
						previous = id
					}
				})
			}
		})
	}
}
