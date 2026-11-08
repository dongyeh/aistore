// Package ais provides AIStore's proxy and target nodes.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package ais

import (
	"math"
	"net/http"
	"strconv"
	"strings"
	"testing"

	"github.com/NVIDIA/aistore/cmn/cos"
)

func TestExactColdRangeInvalidLimit(t *testing.T) {
	for _, limit := range []int64{-1, math.MinInt64, 16*cos.MiB + 1, math.MaxInt64} {
		t.Run(strconv.FormatInt(limit, 10), func(t *testing.T) {
			// No request, LOM, target or backend is available: reject before touching origin I/O.
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("invalid limit reached request/backend state: %v", p)
				}
			}()
			var goi getOI
			handled, code, err := goi.tryExactColdRange(nil, limit)
			if !handled || code != http.StatusInternalServerError || err == nil || !strings.Contains(err.Error(), "cold_range_limit") {
				t.Fatalf("handled=%t code=%d err=%v", handled, code, err)
			}
		})
	}
}

func TestExactColdRangeDisabled(t *testing.T) {
	var goi getOI
	handled, code, err := goi.tryExactColdRange(nil, 0)
	if handled || code != 0 || err != nil {
		t.Fatalf("disabled exact path: handled=%t code=%d err=%v", handled, code, err)
	}
}
