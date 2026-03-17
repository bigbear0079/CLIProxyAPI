package gemini

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiCliResponseToGeminiUnwrapsResponse(t *testing.T) {
	ctx := context.WithValue(context.Background(), "alt", "")
	out := ConvertGeminiCliResponseToGemini(ctx, "", nil, nil, []byte(`data: {"response":{"modelVersion":"gemini-test"}}`), nil)
	if len(out) != 1 {
		t.Fatalf("chunk count = %d, want %d", len(out), 1)
	}

	root := mustParseGeminiCLIGeminiObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(root, "modelVersion"); got != "gemini-test" {
		t.Fatalf("modelVersion = %q, want %q", got, "gemini-test")
	}
}

func TestConvertGeminiCliResponseToGeminiAltCollectsResponses(t *testing.T) {
	ctx := context.WithValue(context.Background(), "alt", "array")
	out := ConvertGeminiCliResponseToGemini(ctx, "", nil, nil, []byte(`[
		{"response":{"id":"a"}},
		{"response":{"id":"b"}}
	]`), nil)
	if len(out) != 1 {
		t.Fatalf("chunk count = %d, want %d", len(out), 1)
	}

	root, errParse := jsonutil.ParseArrayBytes([]byte(out[0]))
	if errParse != nil {
		t.Fatalf("ParseArrayBytes returned error: %v\npayload: %s", errParse, out[0])
	}
	if len(root) != 2 {
		t.Fatalf("array length = %d, want %d", len(root), 2)
	}
	if got, _ := jsonutil.String(root[0], "id"); got != "a" {
		t.Fatalf("root[0].id = %q, want %q", got, "a")
	}
	if got, _ := jsonutil.String(root[1], "id"); got != "b" {
		t.Fatalf("root[1].id = %q, want %q", got, "b")
	}
}

func TestConvertGeminiCliResponseToGeminiNonStreamUnwrapsResponse(t *testing.T) {
	out := ConvertGeminiCliResponseToGeminiNonStream(context.Background(), "", nil, nil, []byte(`{"response":{"id":"resp_1"}}`), nil)
	root := mustParseGeminiCLIGeminiObject(t, []byte(out))
	if got, _ := jsonutil.String(root, "id"); got != "resp_1" {
		t.Fatalf("id = %q, want %q", got, "resp_1")
	}
}

func mustParseGeminiCLIGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
