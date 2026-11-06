// Package ais provides AIStore's proxy and target nodes.
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */
package ais

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/NVIDIA/aistore/ais/s3"
	"github.com/NVIDIA/aistore/api/apc"
	"github.com/NVIDIA/aistore/cmn"
	"github.com/NVIDIA/aistore/cmn/cos"
	"github.com/NVIDIA/aistore/cmn/feat"
	"github.com/NVIDIA/aistore/core"
)

// serve a bounded range from a pinned remote object without holding the object lock.
func (goi *getOI) tryExactColdRange(bp core.Backend, limit int64) (handled bool, code int, err error) {
	if err = cmn.ValidateColdRangeLimit(cos.SizeIEC(limit)); err != nil {
		return true, http.StatusInternalServerError, err
	}
	if limit == 0 {
		return false, 0, nil
	}
	req := goi.req
	if req.URL.Query().Has("versionId") {
		return true, http.StatusNotImplemented, errors.New("exact cold range does not support version-specific requests")
	}
	if goi.lom.IsFeatureSet(feat.S3PresignedRequest) {
		return true, http.StatusNotImplemented, errors.New("exact cold range does not support presigned requests")
	}
	for _, h := range []string{"If-None-Match", "If-Range", "If-Modified-Since", "If-Unmodified-Since"} {
		if req.Header.Get(h) != "" {
			return true, http.StatusNotImplemented, fmt.Errorf("exact cold range does not support %s", h)
		}
	}
	if strings.Contains(goi.ranges.Range, ",") || len(req.Header.Values("Range")) != 1 {
		return true, http.StatusRequestedRangeNotSatisfiable, errors.New("exact cold range requires one byte range")
	}
	attrs, code, err := goi.t.HeadCold(goi.lom, req)
	if err != nil {
		return true, code, err
	}
	ranges, err := parseMultiRange(goi.ranges.Range, attrs.Size)
	if err != nil || len(ranges) != 1 || ranges[0].Length == 0 {
		goi.w.Header().Set(cos.HdrContentRange, fmt.Sprintf("bytes */%d", attrs.Size))
		return true, http.StatusRequestedRangeNotSatisfiable, errors.New("invalid or unsatisfiable exact range")
	}
	span := ranges[0]
	pin := core.NewObjPin(apc.AWS, attrs)
	// Decide before issuing a ranged GET. Once admitted, errors are final.
	if span.Length > limit && len(req.Header.Values("If-Match")) == 0 {
		return false, 0, nil
	}
	if pin == nil || pin.ETag == "" {
		return true, http.StatusBadGateway, errors.New("exact cold range requires a strong origin ETag")
	}
	tag := cmn.QuoteETag(pin.ETag)
	if values := req.Header.Values("If-Match"); len(values) > 0 {
		if len(values) != 1 {
			return true, http.StatusNotImplemented, errors.New("multiple If-Match fields are unsupported")
		}
		if match := values[0]; match != tag && match != "*" {
			if strings.Contains(match, ",") {
				return true, http.StatusNotImplemented, errors.New("If-Match lists are unsupported")
			}
			return true, http.StatusPreconditionFailed, errors.New("origin ETag changed")
		}
	}
	if span.Length > limit {
		goi.fillPin = pin
		return false, 0, nil
	}
	pin.ExactSize = attrs.Size
	res := bp.GetObjReader(req.Context(), goi.lom, span.Start, span.Length, pin)
	if res.Err != nil {
		return true, res.ErrCode, res.Err
	}
	defer cos.Close(res.R)
	data := make([]byte, span.Length)
	if _, err = io.ReadFull(res.R, data); err != nil {
		return true, http.StatusBadGateway, fmt.Errorf("exact range read: %w", err)
	}
	hdr := goi.w.Header()
	s3.SetS3Headers(hdr, attrs)
	attrs.ContentTypeToHeader(hdr)
	hdr.Set(cos.HdrETag, tag)
	hdr.Set(cos.HdrAcceptRanges, "bytes")
	hdr.Set(cos.HdrContentRange, span.contentRange(attrs.Size))
	hdr.Set(cos.HdrContentLength, strconv.Itoa(len(data)))
	ktlsRetire(goi.ktls, hdr, int64(len(data)))
	goi.w.WriteHeader(http.StatusPartialContent)
	n, err := goi.w.Write(data)
	if err != nil || n != len(data) {
		return true, 0, cmn.ErrGetTxBenign
	}
	goi.rget = true
	goi.stats(int64(n))
	return true, 0, nil
}
