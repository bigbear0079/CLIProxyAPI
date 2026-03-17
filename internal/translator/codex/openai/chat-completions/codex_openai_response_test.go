package chat_completions

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/testjson"
)

func TestConvertCodexResponseToOpenAI_StreamSetsModelFromResponseCreated(t *testing.T) {
	ctx := context.Background()
	var param any

	modelName := "gpt-5.3-codex"

	out := ConvertCodexResponseToOpenAI(ctx, modelName, nil, nil, []byte(`data: {"type":"response.created","response":{"id":"resp_123","created_at":1700000000,"model":"gpt-5.3-codex"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("expected no output for response.created, got %d chunks", len(out))
	}

	out = ConvertCodexResponseToOpenAI(ctx, modelName, nil, nil, []byte(`data: {"type":"response.output_text.delta","delta":"hello"}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	gotModel := testjson.Get(out[0], "model").String()
	if gotModel != modelName {
		t.Fatalf("expected model %q, got %q", modelName, gotModel)
	}
}

func TestConvertCodexResponseToOpenAI_FirstChunkUsesRequestModelName(t *testing.T) {
	ctx := context.Background()
	var param any

	modelName := "gpt-5.3-codex"

	out := ConvertCodexResponseToOpenAI(ctx, modelName, nil, nil, []byte(`data: {"type":"response.output_text.delta","delta":"hello"}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	gotModel := testjson.Get(out[0], "model").String()
	if gotModel != modelName {
		t.Fatalf("expected model %q, got %q", modelName, gotModel)
	}
}

func TestConvertCodexResponseToOpenAI_FallbackToolCallDoneRestoresOriginalName(t *testing.T) {
	ctx := context.Background()
	var param any

	originalName := "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"
	originalRequest := []byte(`{
		"tools":[
			{"type":"function","function":{"name":"` + originalName + `"}}
		]
	}`)
	shortName := buildShortNameMap([]string{originalName})[originalName]

	out := ConvertCodexResponseToOpenAI(ctx, "gpt-5.3-codex", originalRequest, nil, []byte(`data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"`+shortName+`","arguments":"{\"q\":\"hi\"}"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(out))
	}

	root := mustParseCodexOpenAIObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.function.name"); got != originalName {
		t.Fatalf("function.name = %q, want %q", got, originalName)
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.function.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("function.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
}

func TestConvertCodexResponseToOpenAINonStreamBuildsReasoningAndToolCalls(t *testing.T) {
	ctx := context.Background()

	originalName := "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"
	originalRequest := []byte(`{
		"tools":[
			{"type":"function","function":{"name":"` + originalName + `"}}
		]
	}`)
	shortName := buildShortNameMap([]string{originalName})[originalName]

	output := ConvertCodexResponseToOpenAINonStream(ctx, "", originalRequest, nil, []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_2",
			"model":"gpt-5.3-codex",
			"created_at":1700000000,
			"status":"completed",
			"usage":{
				"input_tokens":10,
				"output_tokens":5,
				"total_tokens":15,
				"input_tokens_details":{"cached_tokens":3},
				"output_tokens_details":{"reasoning_tokens":2}
			},
			"output":[
				{"type":"reasoning","summary":[{"type":"summary_text","text":"think"}]},
				{"type":"message","content":[{"type":"output_text","text":"hello"}]},
				{"type":"function_call","call_id":"call_2","name":"`+shortName+`","arguments":"{\"q\":\"hi\"}"}
			]
		}
	}`), nil)

	root := mustParseCodexOpenAIObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "choices.0.message.reasoning_content"); got != "think" {
		t.Fatalf("message.reasoning_content = %q, want %q", got, "think")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.content"); got != "hello" {
		t.Fatalf("message.content = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.tool_calls.0.function.name"); got != originalName {
		t.Fatalf("tool_calls.0.function.name = %q, want %q", got, originalName)
	}
	if got, _ := jsonutil.String(root, "usage.prompt_tokens_details.cached_tokens"); got != "3" {
		t.Fatalf("usage.prompt_tokens_details.cached_tokens = %q, want %q", got, "3")
	}
	if got, _ := jsonutil.String(root, "usage.completion_tokens_details.reasoning_tokens"); got != "2" {
		t.Fatalf("usage.completion_tokens_details.reasoning_tokens = %q, want %q", got, "2")
	}
}

func mustParseCodexOpenAIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
