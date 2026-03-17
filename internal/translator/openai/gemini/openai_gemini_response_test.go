package gemini

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertOpenAIResponseToGemini_StreamUsageOnlyChunk(t *testing.T) {
	var param any

	out := ConvertOpenAIResponseToGemini(context.Background(), "", nil, nil, []byte(`data: {
		"model":"gpt-5",
		"choices":[],
		"usage":{
			"prompt_tokens":10,
			"completion_tokens":5,
			"total_tokens":15,
			"completion_tokens_details":{"reasoning_tokens":2}
		}
	}`), &param)
	if len(out) != 1 {
		t.Fatalf("chunk count = %d, want %d", len(out), 1)
	}

	root := mustParseOpenAIGeminiResponseObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(root, "model"); got != "gpt-5" {
		t.Fatalf("model = %q, want %q", got, "gpt-5")
	}
	if got, _ := jsonutil.String(root, "usageMetadata.promptTokenCount"); got != "10" {
		t.Fatalf("usageMetadata.promptTokenCount = %q, want %q", got, "10")
	}
	if got, _ := jsonutil.String(root, "usageMetadata.thoughtsTokenCount"); got != "2" {
		t.Fatalf("usageMetadata.thoughtsTokenCount = %q, want %q", got, "2")
	}
}

func TestConvertOpenAIResponseToGemini_StreamBuildsReasoningAndToolCall(t *testing.T) {
	var param any

	out := ConvertOpenAIResponseToGemini(context.Background(), "", nil, nil, []byte(`data: {
		"model":"gpt-5",
		"choices":[{
			"index":0,
			"delta":{
				"reasoning_content":[{"text":"plan"}],
				"content":"hello"
			}
		}]
	}`), &param)
	if len(out) != 2 {
		t.Fatalf("chunk count = %d, want %d", len(out), 2)
	}

	reasoningRoot := mustParseOpenAIGeminiResponseObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(reasoningRoot, "candidates.0.content.parts.0.text"); got != "plan" {
		t.Fatalf("reasoning text = %q, want %q", got, "plan")
	}
	if got, _ := jsonutil.String(reasoningRoot, "candidates.0.content.parts.0.thought"); got != "true" {
		t.Fatalf("reasoning thought = %q, want %q", got, "true")
	}

	contentRoot := mustParseOpenAIGeminiResponseObject(t, []byte(out[1]))
	if got, _ := jsonutil.String(contentRoot, "candidates.0.content.parts.0.text"); got != "hello" {
		t.Fatalf("content text = %q, want %q", got, "hello")
	}

	out = ConvertOpenAIResponseToGemini(context.Background(), "", nil, nil, []byte(`data: {
		"model":"gpt-5",
		"choices":[{
			"index":0,
			"delta":{
				"tool_calls":[{
					"index":0,
					"id":"call_1",
					"type":"function",
					"function":{"name":"lookup","arguments":"{\"location\": 北京, \"unit\": celsius}"}
				}]
			}
		}]
	}`), &param)
	if len(out) != 0 {
		t.Fatalf("tool delta chunk count = %d, want %d", len(out), 0)
	}

	out = ConvertOpenAIResponseToGemini(context.Background(), "", nil, nil, []byte(`data: {
		"model":"gpt-5",
		"choices":[{
			"index":0,
			"finish_reason":"tool_calls"
		}]
	}`), &param)
	if len(out) != 1 {
		t.Fatalf("finish chunk count = %d, want %d", len(out), 1)
	}

	toolRoot := mustParseOpenAIGeminiResponseObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(toolRoot, "candidates.0.finishReason"); got != "STOP" {
		t.Fatalf("finishReason = %q, want %q", got, "STOP")
	}
	if got, _ := jsonutil.String(toolRoot, "candidates.0.content.parts.0.functionCall.name"); got != "lookup" {
		t.Fatalf("functionCall.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(toolRoot, "candidates.0.content.parts.0.functionCall.args.location"); got != "北京" {
		t.Fatalf("functionCall.args.location = %q, want %q", got, "北京")
	}
	if got, _ := jsonutil.String(toolRoot, "candidates.0.content.parts.0.functionCall.args.unit"); got != "celsius" {
		t.Fatalf("functionCall.args.unit = %q, want %q", got, "celsius")
	}
}

func TestConvertOpenAIResponseToGeminiNonStreamBuildsPartsAndUsage(t *testing.T) {
	output := ConvertOpenAIResponseToGeminiNonStream(context.Background(), "", nil, nil, []byte(`{
		"model":"gpt-5",
		"choices":[{
			"index":7,
			"message":{
				"role":"assistant",
				"reasoning_content":[{"text":"plan"}],
				"content":"hello",
				"tool_calls":[
					{"type":"function","function":{"name":"lookup","arguments":"{\"q\":\"hi\"}"}}
				]
			},
			"finish_reason":"stop"
		}],
		"usage":{
			"prompt_tokens":8,
			"completion_tokens":4,
			"total_tokens":12,
			"completion_tokens_details":{"reasoning_tokens":1}
		}
	}`), nil)

	root := mustParseOpenAIGeminiResponseObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "model"); got != "gpt-5" {
		t.Fatalf("model = %q, want %q", got, "gpt-5")
	}
	if got, _ := jsonutil.String(root, "candidates.0.index"); got != "7" {
		t.Fatalf("candidates.0.index = %q, want %q", got, "7")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.text"); got != "plan" {
		t.Fatalf("parts.0.text = %q, want %q", got, "plan")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.thought"); got != "true" {
		t.Fatalf("parts.0.thought = %q, want %q", got, "true")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.1.text"); got != "hello" {
		t.Fatalf("parts.1.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.2.functionCall.name"); got != "lookup" {
		t.Fatalf("functionCall.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.2.functionCall.args.q"); got != "hi" {
		t.Fatalf("functionCall.args.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "candidates.0.finishReason"); got != "STOP" {
		t.Fatalf("finishReason = %q, want %q", got, "STOP")
	}
	if got, _ := jsonutil.String(root, "usageMetadata.thoughtsTokenCount"); got != "1" {
		t.Fatalf("usageMetadata.thoughtsTokenCount = %q, want %q", got, "1")
	}
}

func mustParseOpenAIGeminiResponseObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
