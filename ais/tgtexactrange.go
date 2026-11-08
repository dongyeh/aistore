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
	match, code, err := exactRangeShape(req)
	if err != nil {
		return true, code, err
	}
	if goi.lom.IsFeatureSet(feat.S3PresignedRequest) {
		return true, http.StatusNotImplemented, errors.New("exact cold range does not support presigned requests")
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
	if match != "" && match != "*" && match != tag {
		return true, http.StatusPreconditionFailed, errors.New("origin ETag changed")
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

// validate the conditional subset before residency selects the read path.
// - accept a single quoted tag, including commas inside its opaque value
// - do not issue remote requests or interpret the object's version here
func exactRangeShape(req *http.Request) (match string, code int, err error) {
	ranges := req.Header.Values(cos.HdrRange)
	if len(ranges) != 1 || strings.TrimSpace(ranges[0]) == "" || strings.Contains(ranges[0], ",") {
		return "", http.StatusRequestedRangeNotSatisfiable, errors.New("exact range requires one byte range")
	}
	if req.URL.Query().Has("versionId") {
		return "", http.StatusNotImplemented, errors.New("exact range does not support version-specific requests")
	}
	for _, h := range []string{"If-None-Match", "If-Range", "If-Modified-Since", "If-Unmodified-Since"} {
		if _, ok := req.Header[h]; ok {
			return "", http.StatusNotImplemented, fmt.Errorf("exact range does not support %s", h)
		}
	}
	values, present := req.Header["If-Match"]
	if !present {
		return "", 0, nil
	}
	if len(values) != 1 {
		return "", http.StatusNotImplemented, errors.New("multiple If-Match fields are unsupported")
	}
	match = strings.Trim(values[0], " \t")
	if match == "*" {
		return match, 0, nil
	}
	tag := match
	weak := strings.HasPrefix(tag, "W/")
	if weak {
		tag = tag[2:]
	}
	valid := len(tag) >= 2 && tag[0] == '"' && tag[len(tag)-1] == '"'
	if valid {
		for i := 1; i < len(tag)-1; i++ {
			c := tag[i]
			if c < 0x21 || c == '"' || c == 0x7f {
				valid = false
				break
			}
		}
	}
	if !valid {
		return "", http.StatusNotImplemented, errors.New("If-Match requires one quoted entity tag")
	}
	if weak {
		return "", http.StatusPreconditionFailed, errors.New("weak If-Match cannot match a strong ETag")
	}
	return match, 0, nil
}

// apply the enabled S3 Range contract before choosing any object GET route.
func (t *target) exactRangeRequest(req *http.Request, dpq *dpq, lom *core.LOM) (string, int, error) {
	if !dpq.isS3 || req.Header[cos.HdrRange] == nil || t.Backend(lom.Bck()).Provider() != apc.AWS {
		return "", 0, nil
	}
	limit := lom.Bprops().Extra.AWS.ColdRangeLimit
	if err := cmn.ValidateColdRangeLimit(limit); err != nil {
		return "", http.StatusInternalServerError, err
	}
	if limit == 0 {
		return "", 0, nil
	}
	return exactRangeShape(req)
}
