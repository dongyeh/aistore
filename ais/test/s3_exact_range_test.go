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

type exactRangeOrigin struct {
	mu               sync.Mutex
	version          byte
	badStatus        bool
	ranges, matches  []string
	entered, release chan struct{}
	once             sync.Once
}

// respond with deterministic content and record the backend GETs, including their version conditions.
func (o *exactRangeOrigin) serve(w http.ResponseWriter, r *http.Request) {
	const size = 2 * cos.MiB
	o.mu.Lock()
	version, badStatus := o.version, o.badStatus
	if r.Method == http.MethodGet {
		o.ranges = append(o.ranges, r.Header.Get(cos.HdrRange))
		o.matches = append(o.matches, r.Header.Get("If-Match"))
	}
	o.mu.Unlock()
	tag := `"b4fc0c467a8d5b04868d12a0d3f43a92"`
	if version == '2' {
		tag = `"9d512c92f6bf0f77cb1a029cfc5e7c6c"`
	}
	if r.Method == http.MethodGet && r.Header.Get("If-Match") != "" && r.Header.Get("If-Match") != tag {
		w.WriteHeader(http.StatusPreconditionFailed)
		return
	}
	w.Header().Set(cos.HdrETag, tag)
	w.Header().Set(cos.HdrLastModified, "Mon, 02 Nov 2026 00:00:00 GMT")
	w.Header().Set(cos.HdrContentType, "application/octet-stream")
	w.Header().Set(cos.HdrContentLength, strconv.Itoa(size))
	if r.Method == http.MethodHead {
		return
	}
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	start, end, status := 0, size-1, http.StatusOK
	if span := r.Header.Get(cos.HdrRange); span != "" {
		if o.entered != nil {
			o.once.Do(func() { close(o.entered) })
			select {
			case <-o.release:
			case <-r.Context().Done():
				return
			}
		}
		if !badStatus {
			if _, err := fmt.Sscanf(span, "bytes=%d-%d", &start, &end); err != nil || start < 0 || end < start || end >= size {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			status = http.StatusPartialContent
			w.Header().Set(cos.HdrContentRange, fmt.Sprintf("bytes %d-%d/%d", start, end, size))
			w.Header().Set(cos.HdrContentLength, strconv.Itoa(end-start+1))
		}
	}
	w.WriteHeader(status)
	_, _ = w.Write(bytes.Repeat([]byte{version}, end-start+1))
}

// run against a local AWS-enabled cluster whose S3_ENDPOINT is an unused loopback HTTP port.
// The test owns that endpoint for its duration; use dummy AWS credentials when deploying the cluster.
// No public cloud bucket is used. Matching the node endpoint preserves its normal egress policy.
func TestS3ExactColdRange(t *testing.T) {
	tools.CheckSkip(t, &tools.SkipTestArgs{RequiredDeployment: tools.ClusterTypeLocal})
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

	bck := cmn.Bck{Name: "exact-" + strings.ToLower(trand.String(10)), Provider: apc.AWS}
	props := &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{
		Endpoint: apc.Ptr(endpoint.String()), CloudRegion: apc.Ptr("us-east-1"), ColdRangeLimit: apc.Ptr(cos.SizeIEC(512 * cos.KiB)),
	}}}
	tassert.CheckFatal(t, api.CreateBucket(bp, bck, props, true /*dontHeadRemote*/))
	defer tools.EvictRemoteBucket(t, bp.URL, bck, false /*keepMD*/)

	get := func(ctx context.Context, key, span, match string) (int, http.Header, []byte, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, bp.URL+apc.URLPathS3.Join(bck.Name, key), http.NoBody)
		if err != nil {
			return 0, nil, nil, err
		}
		if span != "" {
			req.Header.Set(cos.HdrRange, span)
		}
		if match != "" {
			req.Header.Set("If-Match", match)
		}
		resp, err := bp.Client.Do(req)
		if err != nil {
			return 0, nil, nil, err
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		return resp.StatusCode, resp.Header, body, err
	}
	absent := func(t *testing.T, key string) {
		t.Helper()
		_, err := api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
		tassert.Fatalf(t, cmn.IsStatusNotFound(err), "expected no complete cached object: %v", err)
	}
	check := func(t *testing.T, key, span, match string, want int, size int, version byte) http.Header {
		t.Helper()
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		status, hdr, body, err := get(ctx, key, span, match)
		tassert.CheckFatal(t, err)
		tassert.Fatalf(t, status == want, "status %d, expected %d: %.200s", status, want, body)
		if want == http.StatusPartialContent || want == http.StatusOK {
			tassert.Fatalf(t, bytes.Equal(body, bytes.Repeat([]byte{version}, size)), "wrong response body (length %d, expected %d)", len(body), size)
		}
		return hdr
	}

	t.Run("exact-and-conditional-reread", func(t *testing.T) {
		key := "reread"
		origin := &exactRangeOrigin{version: '1'}
		mux.HandleFunc("/"+bck.Name+"/"+key, origin.serve)
		hdr := check(t, key, "bytes=-524288", "", http.StatusPartialContent, 512*cos.KiB, '1')
		tassert.Errorf(t, hdr.Get(cos.HdrContentRange) == "bytes 1572864-2097151/2097152", "wrong content range: %v", hdr)
		tassert.Fatalf(t, hdr.Get(cos.HdrETag) == `"b4fc0c467a8d5b04868d12a0d3f43a92"`, "wrong ETag: %v", hdr)
		absent(t, key)
		check(t, key, "bytes=1048576-1572863", hdr.Get(cos.HdrETag), http.StatusPartialContent, 512*cos.KiB, '1')
		absent(t, key)
		origin.mu.Lock()
		origin.version = '2'
		origin.mu.Unlock()
		check(t, key, "bytes=1048576-1572863", hdr.Get(cos.HdrETag), http.StatusPreconditionFailed, 0, 0)
		absent(t, key)
		origin.mu.Lock()
		defer origin.mu.Unlock()
		tassert.Errorf(t, slices.Equal(origin.ranges, []string{"bytes=1572864-2097151", "bytes=1048576-1572863"}), "unexpected backend GETs: %v", origin.ranges)
		tassert.Errorf(t, slices.Equal(origin.matches, []string{`"b4fc0c467a8d5b04868d12a0d3f43a92"`, `"b4fc0c467a8d5b04868d12a0d3f43a92"`}), "unpinned backend GETs: %v", origin.matches)
	})

	t.Run("reject-origin-200", func(t *testing.T) {
		key := "bad-status"
		origin := &exactRangeOrigin{version: '1', badStatus: true}
		mux.HandleFunc("/"+bck.Name+"/"+key, origin.serve)
		check(t, key, "bytes=-524288", "", http.StatusBadGateway, 0, 0)
		absent(t, key)
	})

	t.Run("release-lock-before-origin-get", func(t *testing.T) {
		key := "concurrent"
		origin := &exactRangeOrigin{version: '1', entered: make(chan struct{}), release: make(chan struct{})}
		mux.HandleFunc("/"+bck.Name+"/"+key, origin.serve)
		ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
		defer cancel()
		done := make(chan error, 1)
		go func() {
			status, _, body, err := get(ctx, key, "bytes=-524288", "")
			if err == nil && (status != http.StatusPartialContent || !bytes.Equal(body, bytes.Repeat([]byte{'1'}, 512*cos.KiB))) {
				err = fmt.Errorf("exact GET: status %d, length %d", status, len(body))
			}
			done <- err
		}()
		defer func() { close(origin.release); tassert.CheckError(t, <-done) }()
		select {
		case <-origin.entered:
		case <-ctx.Done():
			t.Fatal("exact GET did not reach source")
		}
		// The exact GET remains blocked at the source until this whole-object fill completes.
		check(t, key, "", "", http.StatusOK, 2*cos.MiB, '1')
		_, err := api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent})
		tassert.CheckFatal(t, err)
	})

	t.Run("disabled-keeps-whole-fill", func(t *testing.T) {
		props.Extra.AWS.ColdRangeLimit = apc.Ptr(cos.SizeIEC(0))
		_, err := api.SetBucketProps(bp, bck, props)
		tassert.CheckFatal(t, err)
		key := "disabled"
		origin := &exactRangeOrigin{version: '1'}
		mux.HandleFunc("/"+bck.Name+"/"+key, origin.serve)
		check(t, key, "bytes=-524288", "", http.StatusPartialContent, 512*cos.KiB, '1')
		_, err = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent})
		tassert.CheckFatal(t, err)
		origin.mu.Lock()
		defer origin.mu.Unlock()
		tassert.Errorf(t, slices.Equal(origin.ranges, []string{""}), "expected whole backend GET: %v", origin.ranges)
	})
}
