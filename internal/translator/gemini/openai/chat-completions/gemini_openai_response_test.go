package chat_completions

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiResponseToOpenAI_StreamBuildsCandidateChunks(t *testing.T) {
	input := []byte(`{
		"modelVersion":"gemini-2.5-pro",
		"createTime":"2026-01-02T03:04:05Z",
		"responseId":"resp_1",
		"usageMetadata":{
			"promptTokenCount":10,
			"candidatesTokenCount":5,
			"totalTokenCount":15,
			"thoughtsTokenCount":2,
			"cachedContentTokenCount":3
		},
		"candidates":[
			{
				"index":0,
				"content":{
					"parts":[
						{"text":"thinking","thought":true},
						{"text":"hello"},
						{"functionCall":{"name":"lookup","args":{"q":"hi"}}},
						{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
					]
				},
				"finishReason":"STOP"
			},
			{
				"index":1,
				"content":{
					"parts":[
						{"text":"world"}
					]
				},
				"finishReason":"STOP"
			}
		]
	}`)

	var param any
	chunks := ConvertGeminiResponseToOpenAI(context.Background(), "", nil, nil, input, &param)
	if len(chunks) != 2 {
		t.Fatalf("chunk count = %d, want %d", len(chunks), 2)
	}

	firstRoot := mustParseGeminiOpenAIObject(t, []byte(chunks[0]))
	if got, _ := jsonutil.String(firstRoot, "model"); got != "gemini-2.5-pro" {
		t.Fatalf("model = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(firstRoot, "id"); got != "resp_1" {
		t.Fatalf("id = %q, want %q", got, "resp_1")
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.index"); got != "0" {
		t.Fatalf("choices.0.index = %q, want %q", got, "0")
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.delta.reasoning_content"); got != "thinking" {
		t.Fatalf("choices.0.delta.reasoning_content = %q, want %q", got, "thinking")
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.delta.content"); got != "hello" {
		t.Fatalf("choices.0.delta.content = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.delta.tool_calls.0.function.name"); got != "lookup" {
		t.Fatalf("choices.0.delta.tool_calls.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.delta.tool_calls.0.function.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("choices.0.delta.tool_calls.0.function.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.delta.images.0.image_url.url"); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Fatalf("choices.0.delta.images.0.image_url.url = %q, want data URL", got)
	}
	if got, _ := jsonutil.String(firstRoot, "choices.0.finish_reason"); got != "tool_calls" {
		t.Fatalf("choices.0.finish_reason = %q, want %q", got, "tool_calls")
	}
	if got, _ := jsonutil.String(firstRoot, "usage.prompt_tokens_details.cached_tokens"); got != "3" {
		t.Fatalf("usage.prompt_tokens_details.cached_tokens = %q, want %q", got, "3")
	}
	if got, _ := jsonutil.String(firstRoot, "usage.completion_tokens_details.reasoning_tokens"); got != "2" {
		t.Fatalf("usage.completion_tokens_details.reasoning_tokens = %q, want %q", got, "2")
	}

	secondRoot := mustParseGeminiOpenAIObject(t, []byte(chunks[1]))
	if got, _ := jsonutil.String(secondRoot, "choices.0.index"); got != "1" {
		t.Fatalf("choices.0.index = %q, want %q", got, "1")
	}
	if got, _ := jsonutil.String(secondRoot, "choices.0.delta.content"); got != "world" {
		t.Fatalf("choices.0.delta.content = %q, want %q", got, "world")
	}
	if got, _ := jsonutil.String(secondRoot, "choices.0.finish_reason"); got != "stop" {
		t.Fatalf("choices.0.finish_reason = %q, want %q", got, "stop")
	}
}

func TestConvertGeminiResponseToOpenAINonStreamBuildsChoices(t *testing.T) {
	input := []byte(`{
		"modelVersion":"gemini-2.5-pro",
		"createTime":"2026-01-02T03:04:05Z",
		"responseId":"resp_2",
		"usageMetadata":{
			"promptTokenCount":8,
			"candidatesTokenCount":4,
			"totalTokenCount":12,
			"thoughtsTokenCount":1,
			"cachedContentTokenCount":2
		},
		"candidates":[
			{
				"index":0,
				"content":{
					"parts":[
						{"text":"plan","thought":true},
						{"text":"hello"},
						{"functionCall":{"name":"lookup","args":{"q":"hi"}}}
					]
				},
				"finishReason":"STOP"
			},
			{
				"index":1,
				"content":{
					"parts":[
						{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
					]
				},
				"finishReason":"MAX_TOKENS"
			}
		]
	}`)

	output := ConvertGeminiResponseToOpenAINonStream(context.Background(), "", nil, nil, input, nil)
	root := mustParseGeminiOpenAIObject(t, []byte(output))

	if got, _ := jsonutil.String(root, "id"); got != "resp_2" {
		t.Fatalf("id = %q, want %q", got, "resp_2")
	}
	if got, _ := jsonutil.String(root, "model"); got != "gemini-2.5-pro" {
		t.Fatalf("model = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.content"); got != "hello" {
		t.Fatalf("choices.0.message.content = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.reasoning_content"); got != "plan" {
		t.Fatalf("choices.0.message.reasoning_content = %q, want %q", got, "plan")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.tool_calls.0.function.name"); got != "lookup" {
		t.Fatalf("choices.0.message.tool_calls.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "choices.0.finish_reason"); got != "tool_calls" {
		t.Fatalf("choices.0.finish_reason = %q, want %q", got, "tool_calls")
	}
	if got, _ := jsonutil.String(root, "choices.1.finish_reason"); got != "max_tokens" {
		t.Fatalf("choices.1.finish_reason = %q, want %q", got, "max_tokens")
	}
	if got, _ := jsonutil.String(root, "choices.1.message.images.0.image_url.url"); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Fatalf("choices.1.message.images.0.image_url.url = %q, want data URL", got)
	}
	if got, _ := jsonutil.String(root, "usage.prompt_tokens_details.cached_tokens"); got != "2" {
		t.Fatalf("usage.prompt_tokens_details.cached_tokens = %q, want %q", got, "2")
	}
}

func mustParseGeminiOpenAIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
