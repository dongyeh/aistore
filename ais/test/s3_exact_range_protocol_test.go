// Package integration_test.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package integration_test

import (
	"bytes"
	"context"
	"crypto/md5"
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
	"github.com/NVIDIA/aistore/tools/readers"
	"github.com/NVIDIA/aistore/tools/tassert"
	"github.com/NVIDIA/aistore/tools/trand"
)

const (
	exactProtocolV1 = `"b4fc0c467a8d5b04868d12a0d3f43a92"`
	exactProtocolV2 = `"9d512c92f6bf0f77cb1a029cfc5e7c6c"`
)

type (
	exactProtocolEvent struct {
		Method, Key, Range, Match, ETag string
		Status, Bytes, ID               int
		Started, Finished               int64
	}
	exactProtocolOrigin struct {
		mu                 sync.Mutex
		events             []exactProtocolEvent
		fault, gate        string
		entered, release   chan struct{}
		once               sync.Once
		version            byte
		size, active, next int
	}
	exactProtocolResponse struct {
		header http.Header
		body   []byte
		err    error
		status int
	}
)

func (o *exactProtocolOrigin) set(version byte, fault string) {
	o.mu.Lock()
	o.version, o.fault = version, fault
	o.mu.Unlock()
}

// hold either the first HEAD snapshot or the exact GET until the competing fill finishes.
func (o *exactProtocolOrigin) serve(w http.ResponseWriter, r *http.Request) {
	o.mu.Lock()
	version, fault := o.version, o.fault
	o.next++
	o.active++
	id := o.next
	o.mu.Unlock()
	started := time.Now().UnixNano()
	defer func() { o.mu.Lock(); o.active--; o.mu.Unlock() }()
	if (o.gate == "head" && r.Method == http.MethodHead) ||
		(o.gate == "get" && r.Method == http.MethodGet && r.Header.Get(cos.HdrRange) != "") {
		wait := false
		o.once.Do(func() { wait = true; close(o.entered) })
		if wait {
			timer := time.NewTimer(20 * time.Second)
			defer timer.Stop()
			select {
			case <-o.release:
			case <-r.Context().Done():
				return
			case <-timer.C:
				http.Error(w, "origin gate timed out", http.StatusGatewayTimeout)
				return
			}
		}
		if o.gate == "get" {
			o.mu.Lock()
			version = o.version
			o.mu.Unlock()
		}
	}
	if r.Method == http.MethodGet && fault == "replace-after-head" {
		version = '2'
	}
	tag := fmt.Sprintf(`"%x"`, md5.Sum(bytes.Repeat([]byte{version}, o.size)))
	event := exactProtocolEvent{ID: id, Started: started, Method: r.Method, Key: r.URL.Path, Range: r.Header.Get(cos.HdrRange), Match: r.Header.Get("If-Match"), ETag: tag}
	defer func() {
		event.Finished = time.Now().UnixNano()
		o.mu.Lock()
		o.events = append(o.events, event)
		o.mu.Unlock()
	}()
	if r.Method == http.MethodGet && event.Match != "" && event.Match != tag {
		event.Status = http.StatusPreconditionFailed
		w.Header().Set(cos.HdrContentType, "application/xml")
		w.WriteHeader(event.Status)
		event.Bytes, _ = io.WriteString(w, "<Error><Code>PreconditionFailed</Code></Error>")
		return
	}
	start, end, status := 0, o.size-1, http.StatusOK
	if r.Method == http.MethodGet && event.Range != "" {
		if _, err := fmt.Sscanf(event.Range, "bytes=%d-%d", &start, &end); err != nil || start < 0 || end < start || end >= o.size {
			event.Status = http.StatusRequestedRangeNotSatisfiable
			w.WriteHeader(event.Status)
			return
		}
		status = http.StatusPartialContent
	}
	length := end - start + 1
	hdr := w.Header()
	hdr.Set(cos.HdrETag, tag)
	hdr.Set(cos.HdrLastModified, "Mon, 02 Nov 2026 00:00:00 GMT")
	hdr.Set(cos.HdrContentType, "application/octet-stream")
	hdr.Set(cos.HdrContentLength, strconv.Itoa(length))
	if fault == "missing-head-etag" && r.Method == http.MethodHead {
		hdr.Del(cos.HdrETag)
	}
	if status == http.StatusPartialContent {
		hdr.Set(cos.HdrContentRange, fmt.Sprintf("bytes %d-%d/%d", start, end, o.size))
		switch fault {
		case "status-200":
			status = http.StatusOK
		case "wrong-etag":
			hdr.Set(cos.HdrETag, exactProtocolV2)
			version = '2'
		case "missing-etag":
			hdr.Del(cos.HdrETag)
		case "wrong-total":
			hdr.Set(cos.HdrContentRange, fmt.Sprintf("bytes %d-%d/%d", start, end, o.size+1))
		case "wrong-offset":
			hdr.Set(cos.HdrContentRange, fmt.Sprintf("bytes %d-%d/%d", start+1, end+1, o.size))
		case "wrong-length":
			hdr.Set(cos.HdrContentLength, strconv.Itoa(length-1))
			length--
		case "short-body":
			hdr.Set("Connection", "close")
			length--
		}
	}
	event.Status = status
	event.ETag = hdr.Get(cos.HdrETag)
	w.WriteHeader(status)
	if r.Method == http.MethodGet {
		event.Bytes, _ = w.Write(bytes.Repeat([]byte{version}, length))
	}
}

// cover source protocol faults and controlled version changes through the real S3 endpoint.
func TestS3ExactColdRangeProtocol(t *testing.T) {
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
	bck := cmn.Bck{Name: "protocol-" + strings.ToLower(trand.String(10)), Provider: apc.AWS}
	const limit = 512 * cos.KiB
	props := &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{Endpoint: apc.Ptr(endpoint.String()), CloudRegion: apc.Ptr("us-east-1"), ColdRangeLimit: apc.Ptr(cos.SizeIEC(limit))}}}
	tassert.CheckFatal(t, api.CreateBucket(bp, bck, props, true))
	defer tools.EvictRemoteBucket(t, bp.URL, bck, false)
	get := func(ctx context.Context, key, span, match string) (out exactProtocolResponse) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, bp.URL+apc.URLPathS3.Join(bck.Name, key), http.NoBody)
		if err != nil {
			out.err = err
			return
		}
		if span != "" {
			req.Header.Set(cos.HdrRange, span)
		}
		if match != "" {
			req.Header.Set("If-Match", match)
		}
		resp, err := bp.Client.Do(req)
		if err != nil {
			out.err = err
			return
		}
		defer resp.Body.Close()
		out.status, out.header = resp.StatusCode, resp.Header
		out.body, out.err = io.ReadAll(resp.Body)
		return
	}
	check := func(t *testing.T, out exactProtocolResponse, status, size int, version byte) {
		t.Helper()
		tassert.CheckFatal(t, out.err)
		tassert.Fatalf(t, out.status == status, "status=%d want=%d body=%.200s", out.status, status, out.body)
		if status == http.StatusOK || status == http.StatusPartialContent {
			tassert.Fatalf(t, bytes.Equal(out.body, bytes.Repeat([]byte{version}, size)), "incorrect body: %d bytes", len(out.body))
		}
	}
	presence := func(t *testing.T, key string, want bool) {
		t.Helper()
		_, err := api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
		if want {
			tassert.CheckFatal(t, err)
		} else {
			tassert.Fatalf(t, cmn.IsStatusNotFound(err), "unexpected resident object: %v", err)
		}
	}
	register := func(key string, o *exactProtocolOrigin) { mux.HandleFunc("/"+bck.Name+"/"+key, o.serve) }
	logEvents := func(t *testing.T, o *exactProtocolOrigin) []exactProtocolEvent {
		t.Helper()
		// The client can receive the last bytes before the origin handler returns.
		deadline := time.Now().Add(5 * time.Second)
		var events []exactProtocolEvent
		for {
			o.mu.Lock()
			active := o.active
			if active == 0 {
				events = slices.Clone(o.events)
			}
			o.mu.Unlock()
			if active == 0 {
				break
			}
			if time.Now().After(deadline) {
				t.Fatal("origin handler did not finish")
			}
			time.Sleep(time.Millisecond)
		}
		for _, e := range events {
			t.Logf("origin id=%d method=%s key=%s range=%q match=%q etag=%q status=%d bytes=%d started=%d finished=%d", e.ID, e.Method, e.Key, e.Range, e.Match, e.ETag, e.Status, e.Bytes, e.Started, e.Finished)
		}
		return events
	}
	for _, fault := range []string{"replace-after-head", "status-200", "wrong-etag", "missing-etag", "wrong-total", "wrong-offset", "wrong-length", "short-body", "missing-head-etag"} {
		t.Run(fault, func(t *testing.T) {
			o := &exactProtocolOrigin{version: '1', size: 2 * cos.MiB, fault: fault}
			key := fault
			register(key, o)
			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()
			want := http.StatusBadGateway
			if fault == "replace-after-head" {
				want = http.StatusPreconditionFailed
			}
			check(t, get(ctx, key, "bytes=-524288", ""), want, 0, 0)
			presence(t, key, false)
			events := logEvents(t, o)
			gets := 0
			for _, e := range events {
				if e.Method == http.MethodGet {
					gets++
					tassert.Errorf(t, e.Range == "bytes=1572864-2097151" && e.Match == exactProtocolV1, "unexpected unpinned/fallback GET: %+v", e)
				}
			}
			if fault == "missing-head-etag" {
				tassert.Errorf(t, gets == 0, "source GET despite missing HEAD validator")
			} else {
				tassert.Errorf(t, gets == 1, "expected one source GET, got %d", gets)
			}
			// Reuse the same key after failure: no leaked object lock or failed cache publication.
			o.set('1', "")
			out := get(ctx, key, "bytes=-524288", exactProtocolV1)
			check(t, out, http.StatusPartialContent, limit, '1')
			tassert.Errorf(t, out.header.Get(cos.HdrETag) == exactProtocolV1, "recovery ETag mismatch")
			presence(t, key, false)
		})
	}
	for _, tc := range []struct {
		name         string
		size, length int
		match        string
		resident     bool
	}{
		{"exact-entire-small-object", 64 * cos.KiB, 64 * cos.KiB, "", false},
		{"equal-limit", 2 * cos.MiB, limit, "", false},
		{"over-limit-unconditional", 2 * cos.MiB, limit + 1, "", true},
		{"over-limit-pinned", 2 * cos.MiB, limit + 1, exactProtocolV1, true},
		{"no-range", 2 * cos.MiB, 0, "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := &exactProtocolOrigin{version: '1', size: tc.size}
			register(tc.name, o)
			ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
			defer cancel()
			span := ""
			status, length := http.StatusOK, tc.size
			if tc.length > 0 {
				span = fmt.Sprintf("bytes=0-%d", tc.length-1)
				status, length = http.StatusPartialContent, tc.length
			}
			check(t, get(ctx, tc.name, span, tc.match), status, length, '1')
			presence(t, tc.name, tc.resident)
			events := logEvents(t, o)
			gets := 0
			for _, e := range events {
				if e.Method != http.MethodGet {
					continue
				}
				gets++
				wantRange := span
				if tc.resident {
					wantRange = ""
				}
				tassert.Errorf(t, e.Range == wantRange, "wrong source path: %+v", e)
				if tc.match != "" {
					tassert.Errorf(t, e.Match == tc.match, "lost condition: %+v", e)
				}
			}
			tassert.Errorf(t, gets == 1, "expected one GET, got %d", gets)
		})
	}
	t.Run("non-s3-aws-range", func(t *testing.T) {
		key := "native-api"
		o := &exactProtocolOrigin{version: '1', size: 2 * cos.MiB}
		register(key, o)
		var body bytes.Buffer
		_, err := api.GetObject(bp, bck, key, &api.GetArgs{Writer: &body,
			Header: http.Header{cos.HdrRange: []string{"bytes=0-524287"}}})
		tassert.CheckFatal(t, err)
		tassert.Fatalf(t, bytes.Equal(body.Bytes(), bytes.Repeat([]byte{'1'}, limit)), "incorrect native API body")
		presence(t, key, true)
		gets := 0
		for _, e := range logEvents(t, o) {
			if e.Method != http.MethodGet {
				continue
			}
			gets++
			tassert.Errorf(t, e.Range == "", "native API unexpectedly used exact source Range: %+v", e)
		}
		tassert.Errorf(t, gets == 1, "native API source GETs=%d", gets)
	})
	t.Run("non-aws-local-range", func(t *testing.T) {
		local := cmn.Bck{Name: "local-" + strings.ToLower(trand.String(10)), Provider: apc.AIS}
		localProps := &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{
			ColdRangeLimit: apc.Ptr(cos.SizeIEC(limit)),
		}}}
		tassert.CheckFatal(t, api.CreateBucket(bp, local, localProps))
		defer func() { tassert.CheckError(t, api.DestroyBucket(bp, local)) }()
		data := bytes.Repeat([]byte{'1'}, 64*cos.KiB)
		_, err := api.PutObject(&api.PutArgs{BaseParams: bp, Bck: local, ObjName: "local",
			Reader: readers.NewBytes(data), Size: uint64(len(data))})
		tassert.CheckFatal(t, err)
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, bp.URL+apc.URLPathS3.Join(local.Name, "local"), http.NoBody)
		tassert.CheckFatal(t, err)
		req.Header.Set(cos.HdrRange, "bytes=13-524")
		resp, err := bp.Client.Do(req)
		tassert.CheckFatal(t, err)
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		tassert.CheckFatal(t, err)
		tassert.Errorf(t, resp.StatusCode == http.StatusPartialContent && bytes.Equal(body, data[13:525]), "local range changed: status=%d bytes=%d", resp.StatusCode, len(body))
	})
	for round := range 3 {
		for _, tc := range []struct {
			name, gate, match string
			version           byte
			status            int
		}{
			{"head-fill-v1", "head", exactProtocolV1, '1', http.StatusPartialContent},
			{"head-fill-v2", "head", exactProtocolV1, '2', http.StatusPreconditionFailed},
			{"head-fill-unconditional", "head", "", '2', http.StatusPartialContent},
			{"exact-fill-v2", "get", exactProtocolV1, '2', http.StatusPreconditionFailed},
		} {
			t.Run(fmt.Sprintf("race-window-%d/%s", round, tc.name), func(t *testing.T) {
				key := fmt.Sprintf("race-%d-%s", round, tc.name)
				o := &exactProtocolOrigin{version: '1', size: 2 * cos.MiB, gate: tc.gate, entered: make(chan struct{}), release: make(chan struct{})}
				register(key, o)
				ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
				defer cancel()
				length := limit + 1
				if tc.gate == "get" {
					length = limit
				}
				done := make(chan exactProtocolResponse, 1)
				go func() { done <- get(ctx, key, fmt.Sprintf("bytes=0-%d", length-1), tc.match) }()
				var release sync.Once
				defer release.Do(func() { close(o.release) })
				select {
				case <-o.entered:
				case <-ctx.Done():
					t.Fatal("request A did not reach origin gate")
				}
				t.Logf("key=%s phase=A-gated time=%d", key, time.Now().UnixNano())
				o.set(tc.version, "")
				check(t, get(ctx, key, "", ""), http.StatusOK, o.size, tc.version)
				presence(t, key, true)
				t.Logf("key=%s phase=B-resident time=%d", key, time.Now().UnixNano())
				release.Do(func() { close(o.release) })
				t.Logf("key=%s phase=A-released time=%d", key, time.Now().UnixNano())
				select {
				case out := <-done:
					check(t, out, tc.status, length, tc.version)
				case <-ctx.Done():
					t.Fatal("request A did not finish")
				}
				events := logEvents(t, o)
				ranged := 0
				full := 0
				for _, e := range events {
					if e.Method == http.MethodGet {
						if e.Range != "" {
							ranged++
							tassert.Errorf(t, e.Match == exactProtocolV1 && e.Status == http.StatusPreconditionFailed, "wrong exact version decision: %+v", e)
						} else {
							full++
						}
					}
				}
				wantRanges := 0
				if tc.gate == "get" {
					wantRanges = 1
				}
				tassert.Errorf(t, ranged == wantRanges && full == 1, "unexpected source GETs: full=%d ranged=%d", full, ranged)
				check(t, get(ctx, key, "", ""), http.StatusOK, o.size, tc.version)
				presence(t, key, true)
			})
		}
	}
}
