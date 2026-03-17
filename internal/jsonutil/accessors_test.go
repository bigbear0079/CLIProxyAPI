package jsonutil

import (
	"encoding/json"
	"testing"
)

func TestParseObjectBytesOrEmptyFallsBackToEmptyObject(t *testing.T) {
	root := ParseObjectBytesOrEmpty([]byte(`not-json`))

	if len(root) != 0 {
		t.Fatalf("root length = %d, want 0", len(root))
	}
}

func TestTypedAccessors(t *testing.T) {
	root := map[string]any{
		"string": "value",
		"number": json.Number("42"),
		"bool":   true,
		"object": map[string]any{},
		"array":  []any{"x"},
	}

	if got, ok := String(root, "string"); !ok || got != "value" {
		t.Fatalf("String(string) = (%q, %v), want (%q, true)", got, ok, "value")
	}
	if got, ok := Int64(root, "number"); !ok || got != 42 {
		t.Fatalf("Int64(number) = (%d, %v), want (42, true)", got, ok)
	}
	if got, ok := Bool(root, "bool"); !ok || !got {
		t.Fatalf("Bool(bool) = (%v, %v), want (true, true)", got, ok)
	}
	if _, ok := Object(root, "object"); !ok {
		t.Fatal("Object(object) returned ok=false")
	}
	if _, ok := Array(root, "array"); !ok {
		t.Fatal("Array(array) returned ok=false")
	}
	if !IsEmptyObject(root, "object") {
		t.Fatal("IsEmptyObject(object) = false, want true")
	}
}
