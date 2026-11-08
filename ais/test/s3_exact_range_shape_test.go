// Package integration_test.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package integration_test

import (
	"bytes"
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

// pair request forms across residency and both AWS bucket shapes through the S3 endpoint.
func TestS3ExactColdRangeShapes(t *testing.T) {
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
	const size = 2 * cos.MiB
	tag := fmt.Sprintf(`"%x"`, md5.Sum(bytes.Repeat([]byte{'1'}, size)))
	type shape struct {
		name   string
		header http.Header
		query  string
		status int
		reads  bool
	}
	cases := []shape{
		{name: "unconditional", status: 206, reads: true},
		{name: "star", header: http.Header{"If-Match": {"*"}}, status: 206, reads: true},
		{name: "match", header: http.Header{"If-Match": {tag}}, status: 206, reads: true},
		{name: "mismatch", header: http.Header{"If-Match": {`"other"`}}, status: 412, reads: true},
		{name: "quoted-comma", header: http.Header{"If-Match": {`"one,two"`}}, status: 412, reads: true},
		{name: "list", header: http.Header{"If-Match": {`"v1", "v2"`}}, status: 501},
		{name: "duplicate-match", header: http.Header{"If-Match": {tag, tag}}, status: 501},
		{name: "empty-match", header: http.Header{"If-Match": {""}}, status: 501},
		{name: "bare-match", header: http.Header{"If-Match": {"abc"}}, status: 501},
		{name: "unterminated-match", header: http.Header{"If-Match": {`"abc`}}, status: 501},
		{name: "weak", header: http.Header{"If-Match": {`W/"abc"`}}, status: 412},
		{name: "malformed-weak", header: http.Header{"If-Match": {"W/abc"}}, status: 501},
		{name: "duplicate-range", header: http.Header{"Range": {"bytes=0-15", "bytes=16-31"}}, status: 416},
		{name: "comma-range", header: http.Header{"Range": {"bytes=0-15,16-31"}}, status: 416},
		{name: "empty-range", header: http.Header{"Range": {""}}, status: 416},
		{name: "blank-range", header: http.Header{"Range": {"  "}}, status: 416},
		{name: "version", query: "versionId=v1", status: 501},
		{name: "empty-version", query: "versionId=", status: 501},
		{name: "duplicate-version", query: "versionId=v1&versionId=v2", status: 501},
		{name: "range-before-condition", header: http.Header{"Range": {""}, "If-Match": {"abc"}}, status: 416},
	}
	for _, h := range []string{"If-None-Match", "If-Range", "If-Modified-Since", "If-Unmodified-Since"} {
		for _, value := range []string{"", tag} {
			cases = append(cases, shape{name: h + strconv.Itoa(len(value)), header: http.Header{h: {value}}, status: 501})
		}
	}
	propsFor := func() *cmn.BpropsToSet {
		return &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{Endpoint: apc.Ptr(endpoint.String()), CloudRegion: apc.Ptr("us-east-1"), ColdRangeLimit: apc.Ptr(cos.SizeIEC(512 * cos.KiB))}}}
	}
	cleanup := func(bck cmn.Bck) {
		t.Cleanup(func() {
			if bck.Provider == apc.AIS {
				tassert.CheckError(t, api.DestroyBucket(bp, bck))
			} else {
				tassert.CheckError(t, api.EvictRemoteBucket(bp, bck, false))
			}
		})
	}
	for _, provider := range []string{apc.AWS, apc.AIS} {
		t.Run(provider, func(t *testing.T) {
			bck := cmn.Bck{Name: "shape-" + strings.ToLower(trand.String(10)), Provider: provider}
			originBck := bck
			props := propsFor()
			if provider == apc.AIS {
				originBck = cmn.Bck{Name: bck.Name + "-origin", Provider: apc.AWS}
				tassert.CheckFatal(t, api.CreateBucket(bp, originBck, propsFor(), true))
				cleanup(originBck)
				props.BackendBck = &cmn.BackendBckToSet{Name: apc.Ptr(originBck.Name), Provider: apc.Ptr(apc.AWS)}
			}
			tassert.CheckFatal(t, api.CreateBucket(bp, bck, props, true))
			cleanup(bck)
			request := func(t *testing.T, key string, h http.Header, query string) (int, http.Header, []byte) {
				t.Helper()
				req, er := http.NewRequestWithContext(t.Context(), http.MethodGet, bp.URL+apc.URLPathS3.Join(bck.Name, key)+"?"+query, http.NoBody)
				tassert.CheckFatal(t, er)
				req.Header = h.Clone()
				resp, er := bp.Client.Do(req)
				tassert.CheckFatal(t, er)
				defer resp.Body.Close()
				body, er := io.ReadAll(resp.Body)
				tassert.CheckFatal(t, er)
				return resp.StatusCode, resp.Header, body
			}
			events := func(t *testing.T, o *exactProtocolOrigin) []exactProtocolEvent {
				t.Helper()
				deadline := time.Now().Add(5 * time.Second)
				for {
					o.mu.Lock()
					active := o.active
					ev := slices.Clone(o.events)
					o.mu.Unlock()
					if active == 0 {
						return ev
					}
					if time.Now().After(deadline) {
						t.Fatal("source requests did not drain")
					}
					time.Sleep(time.Millisecond)
				}
			}
			for _, objectSize := range []int{64 * cos.KiB, 2 * cos.MiB} {
				for _, warm := range []bool{false, true} {
					for _, tc := range []struct {
						name, blob, threshold, span string
						rangePresent, disabled      bool
						status                      int
					}{
						{"empty-blob", "true", "", "", true, false, 416},
						{"empty-threshold", "", "1", "", true, false, 416},
						{"blank-threshold", "", "1", " ", true, false, 416},
						{"valid-blob", "true", "", "bytes=0-15", true, false, 400},
						{"valid-threshold", "", "1", "bytes=0-15", true, false, 206},
						{"no-range-blob", "true", "", "", false, false, 200},
						{"no-range-threshold", "", "1", "", false, false, 200},
						{"disabled-blob", "true", "", "", true, true, 200},
						{"disabled-threshold", "", "1", "", true, true, 200},
					} {
						t.Run(fmt.Sprintf("blob/%d/warm=%t/%s", objectSize, warm, tc.name), func(t *testing.T) {
							limit := cos.SizeIEC(512 * cos.KiB)
							if tc.disabled {
								limit = 0
							}
							_, er := api.SetBucketProps(bp, bck, &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{ColdRangeLimit: apc.Ptr(limit)}}})
							tassert.CheckFatal(t, er)
							key := strings.ToLower(trand.String(12))
							o := &exactProtocolOrigin{version: '1', size: objectSize}
							mux.HandleFunc("/"+originBck.Name+"/"+key, o.serve)
							if warm {
								status, _, body := request(t, key, http.Header{}, "")
								tassert.Fatalf(t, status == 200 && len(body) == objectSize, "warm setup failed: %d", status)
								events(t, o)
								o.mu.Lock()
								o.events = nil
								o.mu.Unlock()
							}
							_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
							tassert.Fatalf(t, (warm && er == nil) || (!warm && cmn.IsStatusNotFound(er)), "initial presence: %v", er)
							h := make(http.Header)
							if tc.rangePresent {
								h.Set(cos.HdrRange, tc.span)
							}
							if tc.blob != "" {
								h.Set(apc.HdrBlobDownload, tc.blob)
							}
							if tc.threshold != "" {
								h.Set(apc.HdrBlobThreshold, tc.threshold)
							}
							status, _, body := request(t, key, h, "")
							es := events(t, o)
							t.Logf("blob bucket=%s key=%s size=%d warm=%t disabled=%t header=%v status=%d source=%+v", bck.Cname(""), key, objectSize, warm, tc.disabled, h, status, es)
							tassert.Errorf(t, status == tc.status, "status=%d want=%d body=%.200s", status, tc.status, body)
							if tc.status >= 400 || warm {
								tassert.Errorf(t, len(es) == 0, "refused/warm request contacted source: %+v", es)
							}
							if tc.status < 400 {
								n := objectSize
								if tc.status == 206 {
									n = 16
								}
								tassert.Errorf(t, bytes.Equal(body, bytes.Repeat([]byte{'1'}, n)), "incorrect response body")
							}
							wantPresent := warm || tc.status == 200
							_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
							tassert.Errorf(t, (wantPresent && er == nil) || (!wantPresent && cmn.IsStatusNotFound(er)), "final presence expected=%t: %v", wantPresent, er)
						})
					}
				}
			}
			_, err = api.SetBucketProps(bp, bck, &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{ColdRangeLimit: apc.Ptr(cos.SizeIEC(512 * cos.KiB))}}})
			tassert.CheckFatal(t, err)

			for _, checksum := range []string{cos.ChecksumMD5, cos.ChecksumOneXxh, cos.ChecksumNone} {
				t.Run("resident-tag/"+checksum, func(t *testing.T) {
					_, er := api.SetBucketProps(bp, bck, &cmn.BpropsToSet{Cksum: &cmn.CksumConfToSet{Type: apc.Ptr(checksum)}})
					tassert.CheckFatal(t, er)
					key := strings.ToLower(trand.String(12))
					o := &exactProtocolOrigin{version: '1', size: size, fault: "no-origin-etag"}
					mux.HandleFunc("/"+originBck.Name+"/"+key, o.serve)
					status, _, body := request(t, key, http.Header{}, "")
					tassert.Fatalf(t, status == 200 && len(body) == size, "fill failed: %d", status)
					events(t, o)
					o.mu.Lock()
					o.events = nil
					o.mu.Unlock()
					status, h, _ := request(t, key, http.Header{"Range": {"bytes=0-15"}}, "")
					tassert.Fatalf(t, status == 206, "warm probe: %d", status)
					residentTag := h.Get(cos.HdrETag)
					want := 206
					if checksum == cos.ChecksumNone {
						tassert.Errorf(t, residentTag == "", "unexpected tag %s", residentTag)
						residentTag = `"unavailable"`
						want = 412
					} else {
						tassert.Fatalf(t, residentTag != "", "missing derived tag")
					}
					status, h, body = request(t, key, http.Header{"Range": {"bytes=0-15"}, "If-Match": {residentTag}}, "")
					tassert.Errorf(t, status == want, "derived tag status=%d want=%d body=%.200s", status, want, body)
					if want == 206 {
						tassert.Errorf(t, h.Get(cos.HdrETag) == residentTag && bytes.Equal(body, bytes.Repeat([]byte{'1'}, 16)), "tag/body mismatch")
					}
					status, _, _ = request(t, key, http.Header{"Range": {"bytes=0-15"}, "If-Match": {`"wrong"`}}, "")
					tassert.Errorf(t, status == 412, "wrong tag accepted: %d", status)
					tassert.Errorf(t, len(events(t, o)) == 0, "warm comparison accessed origin")
					t.Logf("bucket=%s checksum=%s response-tag=%s", bck.Cname(""), checksum, residentTag)
				})
			}
			_, err = api.SetBucketProps(bp, bck, &cmn.BpropsToSet{Cksum: &cmn.CksumConfToSet{Type: apc.Ptr(cos.ChecksumOneXxh)}})
			tassert.CheckFatal(t, err)
			for _, mode := range []string{"disabled", "no-range", "native-api"} {
				t.Run("scope/"+mode, func(t *testing.T) {
					limit := cos.SizeIEC(512 * cos.KiB)
					if mode == "disabled" {
						limit = 0
					}
					_, er := api.SetBucketProps(bp, bck, &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{ColdRangeLimit: apc.Ptr(limit)}}})
					tassert.CheckFatal(t, er)
					key := strings.ToLower(trand.String(12))
					o := &exactProtocolOrigin{version: '1', size: size}
					mux.HandleFunc("/"+originBck.Name+"/"+key, o.serve)
					h := http.Header{"If-Match": {`"v1", "v2"`}}
					if mode != "no-range" {
						h.Set("Range", "bytes=0-15")
					}
					if mode == "native-api" {
						var body bytes.Buffer
						_, er = api.GetObject(bp, bck, key, &api.GetArgs{Header: h, Writer: &body})
						tassert.CheckFatal(t, er)
						tassert.Errorf(t, body.Len() == 16, "native body length")
					} else {
						status, _, body := request(t, key, h, "")
						want := 206
						n := 16
						if mode == "no-range" {
							want = 200
							n = size
						}
						tassert.Errorf(t, status == want && len(body) == n, "scope changed status=%d bytes=%d", status, len(body))
					}
					_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent})
					tassert.CheckFatal(t, er)
					for _, e := range events(t, o) {
						if e.Method == http.MethodGet {
							tassert.Errorf(t, e.Range == "", "scope used exact range: %+v", e)
						}
					}
				})
			}
			_, err = api.SetBucketProps(bp, bck, &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{ColdRangeLimit: apc.Ptr(cos.SizeIEC(512 * cos.KiB))}}})
			tassert.CheckFatal(t, err)

			for _, tc := range cases {
				for _, warm := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/warm=%t", tc.name, warm), func(t *testing.T) {
						key := strings.ToLower(trand.String(12))
						o := &exactProtocolOrigin{version: '1', size: size}
						mux.HandleFunc("/"+originBck.Name+"/"+key, o.serve)
						if warm {
							status, _, body := request(t, key, http.Header{}, "")
							tassert.Fatalf(t, status == 200 && bytes.Equal(body, bytes.Repeat([]byte{'1'}, size)), "warm setup failed: %d", status)
							events(t, o)
							o.mu.Lock()
							o.events = nil
							o.mu.Unlock()
						}
						_, er := api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
						tassert.Fatalf(t, (warm && er == nil) || (!warm && cmn.IsStatusNotFound(er)), "wrong initial residency: %v", er)
						hdr := tc.header.Clone()
						if hdr == nil {
							hdr = make(http.Header)
						}
						if _, ok := hdr["Range"]; !ok {
							hdr.Set("Range", "bytes=0-15")
						}
						status, response, body := request(t, key, hdr, tc.query)
						ev := events(t, o)
						t.Logf("bucket=%s key=%s warm=%t header=%v query=%s status=%d etag=%s source=%+v", bck.Cname(""), key, warm, hdr, tc.query, status, response.Get(cos.HdrETag), ev)
						tassert.Errorf(t, status == tc.status, "status=%d want=%d body=%.200s", status, tc.status, body)
						if tc.status == 206 {
							tassert.Errorf(t, bytes.Equal(body, bytes.Repeat([]byte{'1'}, 16)), "wrong response bytes")
							tassert.Errorf(t, response.Get(cos.HdrETag) == tag, "wrong ETag")
						}
						if warm || !tc.reads {
							tassert.Errorf(t, len(ev) == 0, "rejected/warm request accessed source: %+v", ev)
						} else {
							wantGets := 0
							if tc.status == 206 {
								wantGets = 1
							}
							heads, gets := 0, 0
							for _, e := range ev {
								if e.Method == http.MethodHead {
									heads++
								}
								if e.Method == http.MethodGet {
									gets++
									tassert.Errorf(t, e.Match == tag && e.Range == "bytes=0-15", "wrong pinned GET: %+v", e)
								}
							}
							tassert.Errorf(t, heads == 1 && gets == wantGets, "unexpected source requests: %+v", ev)
						}
						_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
						tassert.Errorf(t, (warm && er == nil) || (!warm && cmn.IsStatusNotFound(er)), "residency changed: %v", er)
					})
				}
			}
		})
	}
}
