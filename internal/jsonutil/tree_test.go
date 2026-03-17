package jsonutil

import "testing"

func TestSetRawBytesInsertsObject(t *testing.T) {
	root := map[string]any{}

	if err := SetRawBytes(root, "request.systemInstruction", []byte(`{"parts":[{"text":"hello"}]}`)); err != nil {
		t.Fatalf("SetRawBytes returned error: %v", err)
	}

	got, ok := Get(root, "request.systemInstruction.parts.0.text")
	if !ok {
		t.Fatal("Get did not find request.systemInstruction.parts.0.text")
	}
	if got != "hello" {
		t.Fatalf("text = %#v, want %q", got, "hello")
	}
}

func TestExistsReportsPresence(t *testing.T) {
	root := map[string]any{
		"request": map[string]any{
			"contents": []any{
				map[string]any{"role": "user"},
			},
		},
	}

	if !Exists(root, "request.contents.0.role") {
		t.Fatal("Exists returned false for present path")
	}
	if Exists(root, "request.contents.1.role") {
		t.Fatal("Exists returned true for missing path")
	}
}
