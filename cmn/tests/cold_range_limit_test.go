// Package test provides tests for common low-level types and utilities for all aistore projects.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package tests_test

import (
	"strings"
	"testing"

	"github.com/NVIDIA/aistore/api/apc"
	"github.com/NVIDIA/aistore/cmn"
	"github.com/NVIDIA/aistore/cmn/cos"
)

func TestColdRangeBucketScope(t *testing.T) {
	for _, tc := range []struct {
		name, provider, backend string
		limit                   cos.SizeIEC
		multipart               cos.SizeIEC
		invalid                 bool
	}{
		{"aws", apc.AWS, "", 16*cos.MiB + 1, 0, true},
		{"attached-aws", apc.AIS, apc.AWS, 16*cos.MiB + 1, 0, true},
		{"attached-negative", apc.AIS, apc.AWS, -1, 0, true},
		{"attached-disabled", apc.AIS, apc.AWS, 0, 0, false},
		{"attached-maximum", apc.AIS, apc.AWS, 16 * cos.MiB, 0, false},
		{"local-unchanged", apc.AIS, "", 16*cos.MiB + 1, 0, false},
		{"gcp-unchanged", apc.GCP, "", 16*cos.MiB + 1, 0, false},
		{"attached-gcp-unchanged", apc.AIS, apc.GCP, 16*cos.MiB + 1, 0, false},
		{"other-aws-properties-unchanged", apc.AIS, apc.AWS, 512 * cos.KiB, 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := cmn.Bprops{Provider: tc.provider, Cksum: cmn.CksumConf{Type: cos.ChecksumNone}}
			if tc.backend != "" {
				p.BackendBck = cmn.Bck{Name: "origin", Provider: tc.backend}
			}
			p.Extra.AWS.ColdRangeLimit = tc.limit
			p.Extra.AWS.MultiPartSize = tc.multipart
			err := p.Validate(3)
			if tc.invalid {
				if err == nil || !strings.Contains(err.Error(), "cold_range_limit") {
					t.Fatalf("expected limit rejection, got %v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}
