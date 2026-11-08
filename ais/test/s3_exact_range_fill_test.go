// Package integration_test.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package integration_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/NVIDIA/aistore/api"
	"github.com/NVIDIA/aistore/api/apc"
	"github.com/NVIDIA/aistore/api/env"
	"github.com/NVIDIA/aistore/cmn"
	"github.com/NVIDIA/aistore/cmn/cos"
	"github.com/NVIDIA/aistore/tools"
	"github.com/NVIDIA/aistore/tools/tassert"
	"github.com/NVIDIA/aistore/tools/trand"
)

const fillTestChunk = 8 * cos.MiB

type fillEvent struct {
	Method   string `json:"Method"`
	Key      string `json:"Key"`
	Range    string `json:"Range"`
	Match    string `json:"Match"`
	ETag     string `json:"ETag"`
	Status   int    `json:"Status"`
	Bytes    int    `json:"Bytes"`
	ID       int    `json:"ID"`
	Started  int64  `json:"Started"`
	Finished int64  `json:"Finished"`
}
type fillReceipt struct {
	Bucket       string      `json:"Bucket"`
	Key          string      `json:"Key"`
	Scenario     string      `json:"Scenario"`
	Range        string      `json:"Range"`
	Match        string      `json:"Match"`
	ETag         string      `json:"ETag"`
	ContentRange string      `json:"ContentRange"`
	SHA256       string      `json:"SHA256"`
	Size         int         `json:"Size"`
	Status       int         `json:"Status"`
	Bytes        int         `json:"Bytes"`
	Presence     int         `json:"Presence"`
	Events       []fillEvent `json:"Events"`
}

type fillOrigin struct {
	mu                                        sync.Mutex
	events                                    []fillEvent
	gate                                      chan struct{}
	size, fault, active, next, ranges, wholes int
	replace, ignore, parallel                 bool
}

// Use an offset- and version-dependent representation to detect mixed or reordered chunks.
func fillBytes(start, n, version int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((start + i + version) % 251)
	}
	return b
}

func (o *fillOrigin) serve(w http.ResponseWriter, r *http.Request) {
	o.mu.Lock()
	o.active++
	o.next++
	e := fillEvent{ID: o.next, Method: r.Method, Key: r.URL.Path, Range: r.Header.Get(cos.HdrRange), Match: r.Header.Get("If-Match"), Started: time.Now().UnixNano()}
	whole := 0
	if r.Method == http.MethodGet {
		if e.Range == "" {
			o.wholes++
			whole = o.wholes
		} else {
			o.ranges++
			if o.ranges == 2 {
				close(o.gate)
			}
		}
	}
	o.mu.Unlock()
	defer func() {
		e.ETag = w.Header().Get(cos.HdrETag)
		e.Finished = time.Now().UnixNano()
		o.mu.Lock()
		o.events = append(o.events, e)
		o.active--
		o.mu.Unlock()
	}()
	version := 1
	if r.Method == http.MethodGet && e.Range == "" && o.replace && (!o.parallel || whole > 1) {
		version = 2
	}
	e.ETag = fmt.Sprintf(`"v%d"`, version)
	start, end := 0, o.size-1
	if e.Range != "" {
		if _, err := fmt.Sscanf(e.Range, "bytes=%d-%d", &start, &end); err != nil || start < 0 || end < start || start >= o.size {
			e.Status = 416
			w.WriteHeader(e.Status)
			return
		}
		end = min(end, o.size-1)
		timer := time.NewTimer(20 * time.Second)
		defer timer.Stop()
		select {
		case <-o.gate:
		case <-r.Context().Done():
			return
		case <-timer.C:
			e.Status = 504
			w.WriteHeader(e.Status)
			return
		}
		if o.fault != 0 {
			if start == fillTestChunk {
				e.Status = o.fault
				code := "PreconditionFailed"
				if o.fault == 429 {
					code = "TooManyRequests"
				}
				if o.fault == 503 {
					code = "SlowDown"
				}
				w.Header().Set(cos.HdrContentType, "application/xml")
				w.WriteHeader(e.Status)
				e.Bytes, _ = fmt.Fprintf(w, "<Error><Code>%s</Code></Error>", code)
				return
			}
			// Other in-flight chunks remain pending until AIS cancels them. Only the
			// selected later chunk can determine the failure (including SDK retries).
			select {
			case <-r.Context().Done():
				return
			case <-timer.C:
				e.Status = 504
				w.WriteHeader(e.Status)
				return
			}
		}
	}
	if r.Method == http.MethodGet && e.Match != "" && e.Match != e.ETag && !o.ignore {
		e.Status = 412
		w.Header().Set(cos.HdrContentType, "application/xml")
		w.WriteHeader(e.Status)
		e.Bytes, _ = io.WriteString(w, "<Error><Code>PreconditionFailed</Code></Error>")
		return
	}
	e.Status = 200
	if e.Range != "" {
		e.Status = 206
		w.Header().Set(cos.HdrContentRange, fmt.Sprintf("bytes %d-%d/%d", start, end, o.size))
	}
	w.Header().Set(cos.HdrContentLength, strconv.Itoa(end-start+1))
	w.Header().Set(cos.HdrETag, e.ETag)
	w.Header().Set(cos.HdrLastModified, "Mon, 02 Nov 2026 00:00:00 GMT")
	w.WriteHeader(e.Status)
	if r.Method == http.MethodHead {
		return
	}
	for offset := start; offset <= end; {
		n := min(64*cos.KiB, end-offset+1)
		written, err := w.Write(fillBytes(offset, n, version))
		e.Bytes += written
		offset += written
		if err != nil {
			return
		}
	}
}

func (o *fillOrigin) drained(t *testing.T) []fillEvent {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		o.mu.Lock()
		active := o.active
		events := slices.Clone(o.events)
		o.mu.Unlock()
		if active == 0 {
			return events
		}
		if time.Now().After(deadline) {
			t.Fatal("source requests still active")
		}
		time.Sleep(time.Millisecond)
	}
}

// Exercise conditioned whole fills, real parallel ranges, and terminal/retry faults.
func TestS3ExactColdRangeFill(t *testing.T) {
	tools.CheckSkip(t, &tools.SkipTestArgs{RequiredDeployment: tools.ClusterTypeLocal})
	bp := tools.BaseAPIParams(tools.RandomProxyURL(t))
	backends, err := api.GetConfiguredBackends(bp)
	tassert.CheckFatal(t, err)
	if !slices.Contains(backends, apc.AWS) {
		t.Skip("requires AWS-enabled local cluster")
	}
	endpoint, err := url.Parse(os.Getenv(env.AWSEndpoint))
	if err != nil || endpoint.Scheme != "http" || !net.ParseIP(endpoint.Hostname()).IsLoopback() || endpoint.Port() == "" || endpoint.Path != "" {
		t.Skip("requires local S3_ENDPOINT on test and nodes")
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
	// A throttle is target-wide; a new bucket alone does not clear the backoff.
	var backoffUntil time.Time
	for _, provider := range []string{apc.AWS, apc.AIS} {
		for _, tc := range []struct {
			name                      string
			parallel, replace, ignore bool
			fault, want               int
		}{
			{"whole-replaced", false, true, false, 0, 412},
			{"whole-ignored-condition", false, true, true, 0, 502},
			{"parallel-success", true, false, false, 0, 206},
			{"later-412", true, false, false, 412, 412},
			{"later-429-stable", true, false, false, 429, 206},
			{"later-429-replaced", true, true, false, 429, 412},
			{"later-503-stable", true, false, false, 503, 206},
			{"later-503-replaced", true, true, false, 503, 412},
		} {
			t.Run(provider+"/"+tc.name, func(t *testing.T) {
				if delay := time.Until(backoffUntil); delay > 0 {
					timer := time.NewTimer(delay)
					defer timer.Stop()
					select {
					case <-timer.C:
					case <-t.Context().Done():
						t.Fatal("backoff wait canceled")
					}
				}
				bck := cmn.Bck{Name: "fill-" + strings.ToLower(trand.String(10)), Provider: provider}
				originBck := bck
				props := func() *cmn.BpropsToSet {
					return &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{Endpoint: apc.Ptr(endpoint.String()), CloudRegion: apc.Ptr("us-east-1"), ColdRangeLimit: apc.Ptr(cos.SizeIEC(512 * cos.KiB))}}}
				}
				cleanup := func(b cmn.Bck) {
					t.Cleanup(func() {
						if b.Provider == apc.AIS {
							tassert.CheckError(t, api.DestroyBucket(bp, b))
						} else {
							tassert.CheckError(t, api.EvictRemoteBucket(bp, b, false))
						}
					})
				}
				p := props()
				if provider == apc.AIS {
					originBck = cmn.Bck{Name: bck.Name + "-origin", Provider: apc.AWS}
					tassert.CheckFatal(t, api.CreateBucket(bp, originBck, props(), true))
					cleanup(originBck)
					p.BackendBck = &cmn.BackendBckToSet{Name: apc.Ptr(originBck.Name), Provider: apc.Ptr(apc.AWS)}
				}
				threshold := cos.SizeIEC(0)
				size := 2 * cos.MiB
				if tc.parallel {
					threshold = cos.SizeIEC(16 * cos.MiB)
					size = 64 * cos.MiB
				}
				p.Chunks = &cmn.ChunksConfToSet{ObjSizeLimit: apc.Ptr(threshold), ChunkSize: apc.Ptr(cos.SizeIEC(fillTestChunk))}
				p.Cksum = &cmn.CksumConfToSet{Type: apc.Ptr(cos.ChecksumCRC32C), ValidateColdGet: apc.Ptr(false)}
				tassert.CheckFatal(t, api.CreateBucket(bp, bck, p, true))
				cleanup(bck)
				key := strings.ToLower(trand.String(12))
				o := &fillOrigin{gate: make(chan struct{}), size: size, fault: tc.fault, replace: tc.replace, ignore: tc.ignore, parallel: tc.parallel}
				mux.HandleFunc("/"+originBck.Name+"/"+key, o.serve)
				_, err = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
				tassert.Fatalf(t, cmn.IsStatusNotFound(err), "initial residency: %v", err)
				get := func(span string) (int, http.Header, []byte) {
					req, er := http.NewRequestWithContext(t.Context(), http.MethodGet, bp.URL+apc.URLPathS3.Join(bck.Name, key), http.NoBody)
					tassert.CheckFatal(t, er)
					req.Header.Set("If-Match", `"v1"`)
					if span != "" {
						req.Header.Set(cos.HdrRange, span)
					}
					resp, er := bp.Client.Do(req)
					tassert.CheckFatal(t, er)
					defer resp.Body.Close()
					body, er := io.ReadAll(resp.Body)
					tassert.CheckFatal(t, er)
					return resp.StatusCode, resp.Header, body
				}
				const length = 512*cos.KiB + 1
				lo := size - length
				status, h, body := get(fmt.Sprintf("bytes=%d-%d", lo, size-1))
				events := o.drained(t)
				if tc.fault == 429 || tc.fault == 503 {
					backoffUntil = time.Now().Add(cfg.Timeout.ColdRangeBackoff())
				}
				raw, er := json.Marshal(events)
				tassert.CheckFatal(t, er)
				t.Logf("fill key=%s scenario=%s bucket=%s status=%d events=%s", key, tc.name, bck.Cname(""), status, raw)
				tassert.Errorf(t, status == tc.want, "status %d want %d body %.200s", status, tc.want, body)
				if status == 206 {
					tassert.Errorf(t, h.Get(cos.HdrETag) == `"v1"` && bytes.Equal(body, fillBytes(lo, length, 1)), "incorrect successful range")
				}
				var wholes, ranges []fillEvent
				for _, e := range events {
					tassert.Errorf(t, e.Key == "/"+originBck.Name+"/"+key, "source key mismatch")
					if e.Method == http.MethodGet {
						tassert.Errorf(t, e.Match == `"v1"`, "lost condition: %+v", e)
						if e.Range == "" {
							wholes = append(wholes, e)
						} else {
							ranges = append(ranges, e)
						}
					}
				}
				wantWholes := 1
				if tc.fault == 429 || tc.fault == 503 {
					wantWholes = 2
				}
				tassert.Errorf(t, len(wholes) == wantWholes, "whole GET count %d want %d", len(wholes), wantWholes)
				if tc.parallel {
					overlap := false
					for i, a := range ranges {
						for _, b := range ranges[i+1:] {
							if a.Range != b.Range && max(a.Started, b.Started) < min(a.Finished, b.Finished) {
								overlap = true
							}
						}
					}
					tassert.Errorf(t, overlap, "parallel range overlap absent")
					if tc.fault != 0 {
						failures := 0
						firstFault := int64(0)
						for _, e := range ranges {
							if e.Status == tc.fault {
								failures++
								tassert.Errorf(t, e.Range == fmt.Sprintf("bytes=%d-%d", fillTestChunk, 2*fillTestChunk-1), "wrong fault chunk")
								if firstFault == 0 || e.Finished < firstFault {
									firstFault = e.Finished
								}
							}
						}
						tassert.Errorf(t, failures > 0, "expected later fault absent")
						retries := 0
						for _, e := range wholes {
							if firstFault != 0 && e.Started > firstFault {
								retries++
								want := 200
								if tc.replace {
									want = 412
								}
								tassert.Errorf(t, e.Status == want, "retry response wrong")
							}
						}
						wantRetries := 0
						if tc.fault == 429 || tc.fault == 503 {
							wantRetries = 1
						}
						tassert.Errorf(t, retries == wantRetries, "sequential retries=%d want=%d", retries, wantRetries)
					}
				} else {
					tassert.Errorf(t, len(ranges) == 0, "unexpected range source GET")
				}
				_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
				resident := tc.want == 206
				tassert.Errorf(t, (resident && er == nil) || (!resident && cmn.IsStatusNotFound(er)), "residency expected=%t: %v", resident, er)
				residentStatus := 404
				if er == nil {
					residentStatus = 200
				}
				if resident {
					before := len(events)
					code, _, full := get("")
					tassert.Errorf(t, code == 200 && bytes.Equal(full, fillBytes(0, size, 1)), "resident whole body mismatch")
					tassert.Errorf(t, len(o.drained(t)) == before, "resident verification refetched")
				}
				receipt := fillReceipt{bck.Cname(""), key, tc.name, fmt.Sprintf("bytes=%d-%d", lo, size-1), `"v1"`, h.Get(cos.HdrETag), h.Get(cos.HdrContentRange), fmt.Sprintf("%x", sha256.Sum256(body)), size, status, len(body), residentStatus, events}
				raw, er = json.Marshal(receipt)
				tassert.CheckFatal(t, er)
				t.Logf("fill-receipt=%s", raw)
			})
		}
	}
}
