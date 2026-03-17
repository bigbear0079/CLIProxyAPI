package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiResponseToClaude_StreamMapsToolNameAndContinuation(t *testing.T) {
	ctx := context.Background()
	var param any

	originalRequest := []byte(`{
		"tools":[
			{"name":"ReadFile"}
		]
	}`)

	out := ConvertGeminiResponseToClaude(ctx, "", originalRequest, nil, []byte(`{
		"modelVersion":"gemini-2.5-pro",
		"responseId":"resp_1",
		"candidates":[{
			"content":{
				"parts":[
					{"text":"thinking","thought":true},
					{"functionCall":{"name":"readfile","args":{"path":"a.txt"}}}
				]
			}
		}]
	}`), &param)
	if len(out) != 1 {
		t.Fatalf("chunk count = %d, want %d", len(out), 1)
	}
	if !strings.Contains(out[0], `"name":"ReadFile"`) {
		t.Fatalf("expected mapped tool name in output: %s", out[0])
	}
	if !strings.Contains(out[0], `"partial_json":"{\"path\":\"a.txt\"}"`) {
		t.Fatalf("expected initial tool args in output: %s", out[0])
	}

	out = ConvertGeminiResponseToClaude(ctx, "", originalRequest, nil, []byte(`{
		"candidates":[{
			"content":{
				"parts":[
					{"functionCall":{"args":{"line":12}}}
				]
			}
		}]
	}`), &param)
	if len(out) != 1 {
		t.Fatalf("continuation chunk count = %d, want %d", len(out), 1)
	}
	if !strings.Contains(out[0], `"partial_json":"{\"line\":12}"`) {
		t.Fatalf("expected continued tool args in output: %s", out[0])
	}

	out = ConvertGeminiResponseToClaude(ctx, "", originalRequest, nil, []byte(`{
		"usageMetadata":{
			"promptTokenCount":10,
			"candidatesTokenCount":5,
			"thoughtsTokenCount":2
		},
		"candidates":[{
			"finishReason":"STOP"
		}]
	}`), &param)
	if len(out) != 1 {
		t.Fatalf("final chunk count = %d, want %d", len(out), 1)
	}
	if !strings.Contains(out[0], `"stop_reason":"tool_use"`) {
		t.Fatalf("expected tool_use stop reason in output: %s", out[0])
	}
	if !strings.Contains(out[0], `"output_tokens":7`) {
		t.Fatalf("expected output token count in output: %s", out[0])
	}
}

func TestConvertGeminiResponseToClaudeNonStreamBuildsContentAndMappedTool(t *testing.T) {
	originalRequest := []byte(`{
		"tools":[
			{"name":"ReadFile"}
		]
	}`)

	output := ConvertGeminiResponseToClaudeNonStream(context.Background(), "", originalRequest, nil, []byte(`{
		"responseId":"resp_2",
		"modelVersion":"gemini-2.5-pro",
		"usageMetadata":{
			"promptTokenCount":8,
			"candidatesTokenCount":4,
			"thoughtsTokenCount":1
		},
		"candidates":[{
			"content":{
				"parts":[
					{"text":"plan","thought":true},
					{"text":"hello"},
					{"functionCall":{"name":"readfile","args":{"path":"a.txt"}}}
				]
			},
			"finishReason":"STOP"
		}]
	}`), nil)

	root := mustParseGeminiClaudeObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "id"); got != "resp_2" {
		t.Fatalf("id = %q, want %q", got, "resp_2")
	}
	if got, _ := jsonutil.String(root, "model"); got != "gemini-2.5-pro" {
		t.Fatalf("model = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(root, "content.0.type"); got != "thinking" {
		t.Fatalf("content.0.type = %q, want %q", got, "thinking")
	}
	if got, _ := jsonutil.String(root, "content.0.thinking"); got != "plan" {
		t.Fatalf("content.0.thinking = %q, want %q", got, "plan")
	}
	if got, _ := jsonutil.String(root, "content.1.text"); got != "hello" {
		t.Fatalf("content.1.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "content.2.name"); got != "ReadFile" {
		t.Fatalf("content.2.name = %q, want %q", got, "ReadFile")
	}
	if got, _ := jsonutil.String(root, "content.2.input.path"); got != "a.txt" {
		t.Fatalf("content.2.input.path = %q, want %q", got, "a.txt")
	}
	if got, _ := jsonutil.String(root, "stop_reason"); got != "tool_use" {
		t.Fatalf("stop_reason = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "usage.output_tokens"); got != "5" {
		t.Fatalf("usage.output_tokens = %q, want %q", got, "5")
	}
}

func mustParseGeminiClaudeObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
