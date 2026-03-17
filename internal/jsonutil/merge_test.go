package jsonutil

import (
	"encoding/json"
	"testing"
)

func TestNormalizeJSONArrayBytesDefaultsToEmptyArray(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
	}{
		{name: "empty", input: nil},
		{name: "spaces", input: []byte("   ")},
		{name: "object", input: []byte(`{"id":"x"}`)},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := NormalizeJSONArrayBytes(tc.input)
			if err != nil {
				t.Fatalf("NormalizeJSONArrayBytes returned error: %v", err)
			}
			if len(got) != 0 {
				t.Fatalf("len(got) = %d, want 0", len(got))
			}
		})
	}
}

func TestMergeArraysPreservesOrder(t *testing.T) {
	merged := MergeArrays(
		[]any{map[string]any{"id": "msg-1"}},
		[]any{map[string]any{"id": "msg-2"}},
	)

	if len(merged) != 2 {
		t.Fatalf("len(merged) = %d, want 2", len(merged))
	}

	raw, err := json.Marshal(merged)
	if err != nil {
		t.Fatalf("json.Marshal returned error: %v", err)
	}

	if string(raw) != `[{"id":"msg-1"},{"id":"msg-2"}]` {
		t.Fatalf("merged JSON = %s, want %s", string(raw), `[{"id":"msg-1"},{"id":"msg-2"}]`)
	}
}
