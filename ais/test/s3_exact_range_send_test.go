// Package integration_test.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package integration_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/NVIDIA/aistore/api"
	"github.com/NVIDIA/aistore/api/apc"
	"github.com/NVIDIA/aistore/api/env"
	"github.com/NVIDIA/aistore/cmn"
	"github.com/NVIDIA/aistore/cmn/cos"
	"github.com/NVIDIA/aistore/cmn/feat"
	"github.com/NVIDIA/aistore/stats"
	"github.com/NVIDIA/aistore/tools"
	"github.com/NVIDIA/aistore/tools/tassert"
	"github.com/NVIDIA/aistore/tools/trand"
)

// Keep exact and ordinary Range sends consistent: neither installs the regular full-GET deadline.
func TestS3ExactColdRangeSend(t *testing.T) {
	tools.CheckSkip(t, &tools.SkipTestArgs{RequiredDeployment: tools.ClusterTypeLocal, Long: true})
	bp := tools.BaseAPIParams(tools.RandomProxyURL(t))
	backends, err := api.GetConfiguredBackends(bp)
	tassert.CheckFatal(t, err)
	if !slices.Contains(backends, apc.AWS) {
		t.Skip("requires AWS-enabled local cluster")
	}
	endpoint, err := url.Parse(os.Getenv(env.AWSEndpoint))
	if err != nil || endpoint.Scheme != "http" || !net.ParseIP(endpoint.Hostname()).IsLoopback() || endpoint.Port() == "" || endpoint.Path != "" {
		t.Skip("requires S3_ENDPOINT=http://<loopback-IP>:<unused-port> on both test and nodes")
	}
	listener, err := net.Listen("tcp", endpoint.Host)
	tassert.CheckFatal(t, err)
	mux := http.NewServeMux()
	server := httptest.NewUnstartedServer(mux)
	server.Listener.Close()
	server.Listener = listener
	server.Start()
	defer server.Close()

	cfg, err := api.GetClusterConfig(bp)
	tassert.CheckFatal(t, err)
	tassert.CheckFatal(t, api.SetClusterConfig(bp, cos.StrKVs{"timeout.send_file_time": "1m"}, true))
	t.Cleanup(func() {
		tassert.CheckError(t, api.SetClusterConfig(bp, cos.StrKVs{"timeout.send_file_time": cfg.Timeout.SendFile.String()}, true))
	})
	smap, err := api.GetClusterMap(bp)
	tassert.CheckFatal(t, err)
	si, err := smap.GetRandTarget()
	tassert.CheckFatal(t, err)
	nodeCfg, err := api.GetDaemonConfig(bp, si)
	tassert.CheckFatal(t, err)
	tassert.Fatalf(t, nodeCfg.Timeout.SendFile.D() == time.Minute && !nodeCfg.Features.IsSet(feat.DisableGetDeadline), "requires enabled 1m send deadline")
	counts := func() (int64, int64) {
		ds, er := api.GetDaemonStats(bp, si)
		tassert.CheckFatal(t, er)
		return ds.Tracker[stats.GetCount].Value, ds.Tracker[stats.ErrGetCount].Value
	}
	bck := cmn.Bck{Name: "send-" + strings.ToLower(trand.String(10)), Provider: apc.AWS}
	props := &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{
		Endpoint: apc.Ptr(endpoint.String()), CloudRegion: apc.Ptr("us-east-1"), ColdRangeLimit: apc.Ptr(cos.SizeIEC(16 * cos.MiB)),
	}}}
	tassert.CheckFatal(t, api.CreateBucket(bp, bck, props, true))
	t.Cleanup(func() { tassert.CheckError(t, api.EvictRemoteBucket(bp, bck, false)) })
	for _, mode := range []string{"exact-progress", "ordinary-progress", "exact-disconnect"} {
		t.Run(mode, func(t *testing.T) {
			const size, length = 17 * cos.MiB, 16 * cos.MiB
			limit := cos.SizeIEC(length)
			if mode == "ordinary-progress" {
				limit = 0
			}
			props.Extra.AWS.ColdRangeLimit = &limit
			_, er := api.SetBucketProps(bp, bck, props)
			tassert.CheckFatal(t, er)
			key := tools.GenerateObjectNameOnTarget(mode+"-", bck, smap, si)
			origin := &exactRangeOrigin{version: '1', size: size}
			mux.HandleFunc("/"+bck.Name+"/"+key, origin.serve)
			transport := &http.Transport{DisableKeepAlives: true, DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
				conn, er := (&net.Dialer{}).DialContext(ctx, network, address)
				if er == nil {
					er = conn.(*net.TCPConn).SetReadBuffer(32 * cos.KiB)
					if er != nil {
						conn.Close()
					}
				}
				return conn, er
			}}
			defer transport.CloseIdleConnections()
			client := &http.Client{Transport: transport, Timeout: 3 * time.Minute}
			req, er := http.NewRequestWithContext(t.Context(), http.MethodGet, bp.URL+apc.URLPathS3.Join(bck.Name, key), http.NoBody)
			tassert.CheckFatal(t, er)
			req.Header.Set(cos.HdrRange, fmt.Sprintf("bytes=%d-%d", size-length, size-1))
			before, beforeErr := counts()
			resp, er := client.Do(req)
			tassert.CheckFatal(t, er)
			defer resp.Body.Close()
			tassert.Fatalf(t, resp.StatusCode == http.StatusPartialContent && resp.ContentLength == length, "status=%d length=%d", resp.StatusCode, resp.ContentLength)
			start := time.Now()
			buf := make([]byte, 32*cos.KiB)
			var received int64
			read := func() error {
				n, err := io.ReadFull(resp.Body, buf)
				received += int64(n)
				tassert.Errorf(t, bytes.Count(buf[:n], []byte{'1'}) == n, "unexpected content")
				return err
			}
			if mode == "exact-disconnect" {
				tassert.CheckFatal(t, read())
				tassert.CheckFatal(t, resp.Body.Close())
				deadline := time.Now().Add(10 * time.Second)
				var completed, failed int64
				for time.Now().Before(deadline) {
					completed, failed = counts()
					if failed > beforeErr {
						break
					}
					time.Sleep(50 * time.Millisecond)
				}
				t.Logf("closed after=%d bytes; completed=%d failed=%d elapsed=%v", received, completed-before, failed-beforeErr, time.Since(start))
				tassert.Fatalf(t, completed == before && failed == beforeErr+1, "disconnect did not return through GET error handling")
				// A new read of the same key must still complete after the failed send.
				req.Header.Set(cos.HdrRange, "bytes=0-524287")
				resp2, err := bp.Client.Do(req)
				tassert.CheckFatal(t, err)
				body, err := io.ReadAll(resp2.Body)
				resp2.Body.Close()
				tassert.CheckFatal(t, err)
				tassert.Fatalf(t, resp2.StatusCode == http.StatusPartialContent && bytes.Equal(body, bytes.Repeat([]byte{'1'}, 512*cos.KiB)), "same-key read failed")
			} else {
				for time.Since(start) < 65*time.Second {
					if er = read(); er != nil {
						break
					}
					time.Sleep(250 * time.Millisecond)
				}
				completed, failed := counts()
				t.Logf("before draining: elapsed=%v received=%d completed=%d failed=%d", time.Since(start), received, completed-before, failed-beforeErr)
				tassert.Errorf(t, completed == before && failed == beforeErr, "send finished before draining: completed=%d failed=%d", completed-before, failed-beforeErr)
				tassert.Errorf(t, time.Since(start) > time.Minute && received < length, "did not exercise send backpressure beyond the window")
				for er == nil {
					er = read()
				}
				t.Logf("final: elapsed=%v received=%d error=%v", time.Since(start), received, er)
				tassert.Fatalf(t, er == io.EOF && received == length, "incomplete response: bytes=%d error=%v", received, er)
			}
			_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
			if mode == "ordinary-progress" {
				tassert.CheckFatal(t, er)
			} else {
				tassert.Fatalf(t, cmn.IsStatusNotFound(er), "exact read unexpectedly resident: %v", er)
			}
			origin.mu.Lock()
			defer origin.mu.Unlock()
			t.Logf("origin ranges=%v", origin.ranges)
			want := []string{req.Header.Get(cos.HdrRange)}
			switch mode {
			case "ordinary-progress":
				want = []string{""}
			case "exact-disconnect":
				want = []string{fmt.Sprintf("bytes=%d-%d", size-length, size-1), "bytes=0-524287"}
			}
			tassert.Errorf(t, slices.Equal(origin.ranges, want), "unexpected source reads: %v", origin.ranges)
		})
	}
}
