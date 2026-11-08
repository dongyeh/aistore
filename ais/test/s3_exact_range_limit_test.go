// Package integration_test.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package integration_test

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/NVIDIA/aistore/api"
	"github.com/NVIDIA/aistore/api/apc"
	"github.com/NVIDIA/aistore/api/env"
	"github.com/NVIDIA/aistore/cmn"
	"github.com/NVIDIA/aistore/cmn/cos"
	"github.com/NVIDIA/aistore/tools"
	"github.com/NVIDIA/aistore/tools/tassert"
	"github.com/NVIDIA/aistore/tools/trand"

	jsoniter "github.com/json-iterator/go"
)

// validate the same limit through real bucket APIs and S3 reads for both AWS bucket shapes.
func TestS3ExactColdRangeLimits(t *testing.T) {
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
	propsFor := func(limit cos.SizeIEC) *cmn.BpropsToSet {
		return &cmn.BpropsToSet{Extra: &cmn.ExtraToSet{AWS: &cmn.ExtraPropsAWSToSet{
			Endpoint: apc.Ptr(endpoint.String()), CloudRegion: apc.Ptr("us-east-1"), ColdRangeLimit: apc.Ptr(limit),
		}}}
	}
	// SizeIEC marshals to rounded IEC units. Send exact bytes so limit+1 reaches the server unchanged.
	applyLimit := func(bck cmn.Bck, props *cmn.BpropsToSet, limit string, create bool) error {
		var value map[string]any
		if er := jsoniter.Unmarshal(cos.MustMarshal(props), &value); er != nil {
			return er
		}
		value["extra"].(map[string]any)["aws"].(map[string]any)["cold_range_limit"] = limit
		params := api.AllocRp()
		defer api.FreeRp(params)
		params.BaseParams = bp
		params.BaseParams.Method = http.MethodPatch
		action := apc.ActSetBprops
		params.Query = bck.AddToQuery(url.Values{})
		if create {
			params.BaseParams.Method = http.MethodPost
			action = apc.ActCreateBck
			params.Query.Set(apc.QparamDontHeadRemote, "true")
		}
		params.Path = apc.URLPathBuckets.Join(bck.Name)
		params.Header = http.Header{cos.HdrContentType: []string{cos.ContentJSON}}
		params.Body = cos.MustMarshal(apc.ActMsg{Action: action, Value: value})
		return params.DoRequest()
	}
	newBucket := func(provider string) cmn.Bck {
		return cmn.Bck{Name: "limit-" + strings.ToLower(trand.String(10)), Provider: provider}
	}
	cleanup := func(t *testing.T, bck cmn.Bck) {
		t.Helper()
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
			backend := newBucket(apc.AWS)
			if provider == apc.AIS {
				tassert.CheckFatal(t, api.CreateBucket(bp, backend, propsFor(0), true))
				cleanup(t, backend)
			}
			makeProps := func(limit cos.SizeIEC) *cmn.BpropsToSet {
				props := propsFor(limit)
				if provider == apc.AIS {
					props.BackendBck = &cmn.BackendBckToSet{Name: apc.Ptr(backend.Name), Provider: apc.Ptr(apc.AWS)}
				}
				return props
			}
			for _, op := range []string{"create", "update"} {
				for _, limit := range []cos.SizeIEC{-1, 0, 512 * cos.KiB, 16 * cos.MiB, 16*cos.MiB + 1, math.MaxInt64} {
					t.Run(fmt.Sprintf("%s/%d", op, limit), func(t *testing.T) {
						bck := newBucket(provider)
						if op == "update" {
							tassert.CheckFatal(t, api.CreateBucket(bp, bck, makeProps(512*cos.KiB), true))
							cleanup(t, bck)
							err = applyLimit(bck, makeProps(limit), fmt.Sprintf("%dB", limit), false)
						} else {
							err = applyLimit(bck, makeProps(limit), fmt.Sprintf("%dB", limit), true)
							if err == nil {
								cleanup(t, bck)
							}
						}
						invalid := limit < 0 || limit > 16*cos.MiB
						t.Logf("bucket=%s operation=%s limit=%d error=%v", bck.Cname(""), op, limit, err)
						if invalid {
							tassert.Fatalf(t, err != nil && strings.Contains(err.Error(), "cold_range_limit"), "invalid limit accepted or wrong rejection: %v", err)
							if op == "create" {
								return
							}
							limit = 512 * cos.KiB
						} else {
							tassert.CheckFatal(t, err)
						}
						props, er := api.HeadBucket(bp, bck, true)
						tassert.CheckFatal(t, er)
						tassert.Errorf(t, props.Extra.AWS.ColdRangeLimit == limit, "stored limit=%d, expected %d", props.Extra.AWS.ColdRangeLimit, limit)
						if provider == apc.AIS {
							tassert.Errorf(t, props.BackendBck.Equal(&backend), "backend mismatch: %v", props.BackendBck)
						}
					})
				}
			}
			t.Run("overflow-input", func(t *testing.T) {
				bck := newBucket(provider)
				tassert.CheckFatal(t, api.CreateBucket(bp, bck, makeProps(512*cos.KiB), true))
				cleanup(t, bck)
				err := applyLimit(bck, makeProps(0), "18446744073709551616B", false)
				tassert.Fatalf(t, err != nil, "overflowing byte limit accepted")
				p, er := api.HeadBucket(bp, bck, true)
				tassert.CheckFatal(t, er)
				tassert.Errorf(t, p.Extra.AWS.ColdRangeLimit == 512*cos.KiB, "overflow changed stored limit: %d", p.Extra.AWS.ColdRangeLimit)
			})
			t.Run("reads", func(t *testing.T) {
				bck := newBucket(provider)
				tassert.CheckFatal(t, api.CreateBucket(bp, bck, makeProps(512*cos.KiB), true))
				cleanup(t, bck)
				for _, tc := range []struct {
					name   string
					limit  cos.SizeIEC
					length int
					fill   bool
				}{
					{"small", 512 * cos.KiB, 512 * cos.KiB, false},
					{"maximum", 16 * cos.MiB, 16 * cos.MiB, false},
					{"over-limit", 512 * cos.KiB, 512*cos.KiB + 1, true},
					{"disabled", 0, 512 * cos.KiB, true},
				} {
					t.Run(tc.name, func(t *testing.T) {
						_, er := api.SetBucketProps(bp, bck, makeProps(tc.limit))
						tassert.CheckFatal(t, er)
						const size = 17 * cos.MiB
						origin := &exactRangeOrigin{version: '1', size: size}
						key := tc.name
						originBucket := bck.Name
						if provider == apc.AIS {
							originBucket = backend.Name
						}
						mux.HandleFunc("/"+originBucket+"/"+key, origin.serve)
						span := fmt.Sprintf("bytes=%d-%d", size-tc.length, size-1)
						req, er := http.NewRequestWithContext(t.Context(), http.MethodGet, bp.URL+apc.URLPathS3.Join(bck.Name, key), http.NoBody)
						tassert.CheckFatal(t, er)
						req.Header.Set(cos.HdrRange, span)
						req.Header.Set("If-Match", `"b4fc0c467a8d5b04868d12a0d3f43a92"`)
						resp, er := bp.Client.Do(req)
						tassert.CheckFatal(t, er)
						body, er := io.ReadAll(resp.Body)
						resp.Body.Close()
						tassert.CheckFatal(t, er)
						tassert.Fatalf(t, resp.StatusCode == http.StatusPartialContent, "status=%d: %.200s", resp.StatusCode, body)
						tassert.Fatalf(t, bytes.Equal(body, bytes.Repeat([]byte{'1'}, tc.length)), "wrong response body: %d bytes", len(body))
						_, er = api.HeadObject(bp, bck, key, api.HeadArgs{FltPresence: apc.FltPresent, Silent: true})
						if tc.fill {
							tassert.CheckFatal(t, er)
						} else {
							tassert.Fatalf(t, cmn.IsStatusNotFound(er), "unexpected residency: %v", er)
						}
						origin.mu.Lock()
						defer origin.mu.Unlock()
						wantRange := span
						if tc.fill {
							wantRange = ""
						}
						t.Logf("limit=%d length=%d origin_ranges=%v origin_matches=%v resident=%t", tc.limit, tc.length, origin.ranges, origin.matches, tc.fill)
						tassert.Errorf(t, slices.Equal(origin.ranges, []string{wantRange}), "unexpected origin ranges: %v", origin.ranges)
						if tc.limit > 0 {
							tassert.Errorf(t, slices.Equal(origin.matches, []string{req.Header.Get("If-Match")}), "lost version condition: %v", origin.matches)
						}
					})
				}
			})
		})
	}
}
