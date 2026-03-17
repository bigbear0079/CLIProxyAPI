package jsonutil

import "bytes"

func MergeArrays(left, right []any) []any {
	out := make([]any, 0, len(left)+len(right))
	out = append(out, left...)
	out = append(out, right...)
	return out
}

func NormalizeJSONArrayBytes(raw []byte) ([]any, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return []any{}, nil
	}
	value, errParse := ParseAnyBytes(raw)
	if errParse != nil {
		return nil, errParse
	}
	array, ok := value.([]any)
	if !ok {
		return []any{}, nil
	}
	return array, nil
}
