package jsonutil

import (
	"fmt"
	"strconv"
	"strings"
)

// Segment represents one path segment in the repository's narrow JSON path syntax.
type Segment struct {
	Key    string
	Index  *int
	Append bool
}

// ParsePath parses a dot-separated path that may contain object keys, array
// indexes, and the append token "-1".
//
// Numeric segments preserve their raw key and are only interpreted as array
// indexes when the runtime container at that segment is actually an array. The
// special "-1" token behaves similarly: it is treated as an object key inside
// objects, and as an append marker only inside arrays. When a container is
// missing, the index and append hints are still retained so Set can preserve
// the legacy array-creation behavior used by the previous sjson-based helpers.
func ParsePath(path string) ([]Segment, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, fmt.Errorf("empty path")
	}

	rawSegments := splitPath(path)
	segments := make([]Segment, 0, len(rawSegments))
	for _, rawSegment := range rawSegments {
		rawSegment = strings.TrimSpace(rawSegment)
		if rawSegment == "" {
			return nil, fmt.Errorf("invalid empty path segment in %q", path)
		}
		segment := Segment{Key: rawSegment}
		if rawSegment == "-1" {
			segment.Append = true
			segments = append(segments, segment)
			continue
		}
		if index, errParse := strconv.Atoi(rawSegment); errParse == nil && index >= 0 {
			segment.Index = &index
			segments = append(segments, segment)
			continue
		}
		segments = append(segments, segment)
	}
	return segments, nil
}

func splitPath(path string) []string {
	segments := make([]string, 0, 4)
	var builder strings.Builder
	escaped := false

	for _, char := range path {
		if escaped {
			builder.WriteRune(char)
			escaped = false
			continue
		}
		if char == '\\' {
			escaped = true
			continue
		}
		if char == '.' {
			segments = append(segments, builder.String())
			builder.Reset()
			continue
		}
		builder.WriteRune(char)
	}
	if escaped {
		builder.WriteRune('\\')
	}
	segments = append(segments, builder.String())
	return segments
}
