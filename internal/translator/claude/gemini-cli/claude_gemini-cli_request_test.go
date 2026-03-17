package geminiCLI

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiCLIRequestToClaudePromotesRequestEnvelope(t *testing.T) {
	input := []byte(`{
		"model":"gemini-cli-original",
		"request":{
			"systemInstruction":{"parts":[{"text":"sys"}]},
			"contents":[
				{"role":"user","parts":[{"text":"hello"}]}
			]
		}
	}`)

	output := ConvertGeminiCLIRequestToClaude("claude-test", input, true)
	root := mustParseClaudeGeminiCLIObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "claude-test" {
		t.Fatalf("model = %q, want %q", got, "claude-test")
	}
	if got, ok := jsonutil.Bool(root, "stream"); !ok || !got {
		t.Fatalf("stream = (%v, %v), want (true, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "messages.0.role"); got != "user" {
		t.Fatalf("messages.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.0.text"); got != "sys" {
		t.Fatalf("messages.0.content.0.text = %q, want %q", got, "sys")
	}
	if got, _ := jsonutil.String(root, "messages.1.content.0.text"); got != "hello" {
		t.Fatalf("messages.1.content.0.text = %q, want %q", got, "hello")
	}
}

func mustParseClaudeGeminiCLIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
