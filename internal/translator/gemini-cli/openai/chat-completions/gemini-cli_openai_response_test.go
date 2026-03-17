package chat_completions

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertCliResponseToOpenAIBuildsStreamingChunk(t *testing.T) {
	input := []byte(`{
		"response":{
			"modelVersion":"gemini-2.5-pro",
			"createTime":"2026-01-02T03:04:05Z",
			"responseId":"resp_1",
			"usageMetadata":{
				"promptTokenCount":10,
				"candidatesTokenCount":5,
				"totalTokenCount":15,
				"thoughtsTokenCount":2
			},
			"candidates":[
				{
					"content":{
						"parts":[
							{"text":"thinking","thought":true},
							{"text":"hello"},
							{"functionCall":{"name":"lookup","args":{"q":"hi"}}},
							{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
						]
					},
					"finishReason":"STOP"
				}
			]
		}
	}`)

	var param any
	chunks := ConvertCliResponseToOpenAI(context.Background(), "", nil, nil, input, &param)
	if len(chunks) != 1 {
		t.Fatalf("chunk count = %d, want %d", len(chunks), 1)
	}

	root := mustParseGeminiCLIOpenAIObject(t, []byte(chunks[0]))

	if got, _ := jsonutil.String(root, "model"); got != "gemini-2.5-pro" {
		t.Fatalf("model = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(root, "id"); got != "resp_1" {
		t.Fatalf("id = %q, want %q", got, "resp_1")
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.reasoning_content"); got != "thinking" {
		t.Fatalf("choices.0.delta.reasoning_content = %q, want %q", got, "thinking")
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.content"); got != "hello" {
		t.Fatalf("choices.0.delta.content = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.function.name"); got != "lookup" {
		t.Fatalf("choices.0.delta.tool_calls.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.function.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("choices.0.delta.tool_calls.0.function.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.images.0.image_url.url"); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Fatalf("choices.0.delta.images.0.image_url.url = %q, want data URL", got)
	}
	if got, _ := jsonutil.String(root, "choices.0.finish_reason"); got != "tool_calls" {
		t.Fatalf("choices.0.finish_reason = %q, want %q", got, "tool_calls")
	}
	if got, _ := jsonutil.String(root, "usage.prompt_tokens"); got != "10" {
		t.Fatalf("usage.prompt_tokens = %q, want %q", got, "10")
	}
	if got, _ := jsonutil.String(root, "usage.completion_tokens_details.reasoning_tokens"); got != "2" {
		t.Fatalf("usage.completion_tokens_details.reasoning_tokens = %q, want %q", got, "2")
	}
}

func TestConvertCliResponseToOpenAINonStreamUnwrapsResponse(t *testing.T) {
	input := []byte(`{
		"response":{
			"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],
			"modelVersion":"gemini-2.5-pro",
			"createTime":"2026-01-02T03:04:05Z",
			"responseId":"resp_2"
		}
	}`)

	var param any
	output := ConvertCliResponseToOpenAINonStream(context.Background(), "", nil, nil, input, &param)
	if output == "" {
		t.Fatal("expected non-empty non-stream response")
	}
}

func mustParseGeminiCLIOpenAIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
