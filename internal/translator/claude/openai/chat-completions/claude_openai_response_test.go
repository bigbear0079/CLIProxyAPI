package chat_completions

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertClaudeResponseToOpenAI_StreamTextReasoningAndUsage(t *testing.T) {
	ctx := context.Background()
	var param any

	out := ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"message_start","message":{"id":"msg_1"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("message_start chunk count = %d, want %d", len(out), 1)
	}

	startRoot := mustParseClaudeOpenAIObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(startRoot, "id"); got != "msg_1" {
		t.Fatalf("id = %q, want %q", got, "msg_1")
	}
	if got, _ := jsonutil.String(startRoot, "model"); got != "claude-3-5-sonnet" {
		t.Fatalf("model = %q, want %q", got, "claude-3-5-sonnet")
	}
	if got, _ := jsonutil.String(startRoot, "choices.0.delta.role"); got != "assistant" {
		t.Fatalf("choices.0.delta.role = %q, want %q", got, "assistant")
	}

	out = ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"plan"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("thinking_delta chunk count = %d, want %d", len(out), 1)
	}
	thinkingRoot := mustParseClaudeOpenAIObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(thinkingRoot, "choices.0.delta.reasoning_content"); got != "plan" {
		t.Fatalf("choices.0.delta.reasoning_content = %q, want %q", got, "plan")
	}

	out = ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}`), &param)
	if len(out) != 1 {
		t.Fatalf("text_delta chunk count = %d, want %d", len(out), 1)
	}
	textRoot := mustParseClaudeOpenAIObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(textRoot, "choices.0.delta.content"); got != "hello" {
		t.Fatalf("choices.0.delta.content = %q, want %q", got, "hello")
	}

	out = ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":12,"output_tokens":5,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}}`), &param)
	if len(out) != 1 {
		t.Fatalf("message_delta chunk count = %d, want %d", len(out), 1)
	}
	usageRoot := mustParseClaudeOpenAIObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(usageRoot, "choices.0.finish_reason"); got != "stop" {
		t.Fatalf("choices.0.finish_reason = %q, want %q", got, "stop")
	}
	if got, _ := jsonutil.String(usageRoot, "usage.prompt_tokens"); got != "14" {
		t.Fatalf("usage.prompt_tokens = %q, want %q", got, "14")
	}
	if got, _ := jsonutil.String(usageRoot, "usage.completion_tokens"); got != "5" {
		t.Fatalf("usage.completion_tokens = %q, want %q", got, "5")
	}
	if got, _ := jsonutil.String(usageRoot, "usage.total_tokens"); got != "17" {
		t.Fatalf("usage.total_tokens = %q, want %q", got, "17")
	}
	if got, _ := jsonutil.String(usageRoot, "usage.prompt_tokens_details.cached_tokens"); got != "3" {
		t.Fatalf("usage.prompt_tokens_details.cached_tokens = %q, want %q", got, "3")
	}
}

func TestConvertClaudeResponseToOpenAI_StreamToolCallEmitsOnBlockStop(t *testing.T) {
	ctx := context.Background()
	var param any

	ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"message_start","message":{"id":"msg_2"}}`), &param)

	out := ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"content_block_start","index":4,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("content_block_start chunk count = %d, want %d", len(out), 0)
	}

	out = ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"content_block_delta","index":4,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"hi\"}"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("input_json_delta chunk count = %d, want %d", len(out), 0)
	}

	out = ConvertClaudeResponseToOpenAI(ctx, "claude-3-5-sonnet", nil, nil, []byte(`data: {"type":"content_block_stop","index":4}`), &param)
	if len(out) != 1 {
		t.Fatalf("content_block_stop chunk count = %d, want %d", len(out), 1)
	}

	root := mustParseClaudeOpenAIObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.id"); got != "toolu_1" {
		t.Fatalf("tool_calls.0.id = %q, want %q", got, "toolu_1")
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.function.name"); got != "lookup" {
		t.Fatalf("tool_calls.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.function.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("tool_calls.0.function.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(root, "choices.0.delta.tool_calls.0.index"); got != "4" {
		t.Fatalf("tool_calls.0.index = %q, want %q", got, "4")
	}
}

func TestConvertClaudeResponseToOpenAINonStreamBuildsMessageAndToolCalls(t *testing.T) {
	ctx := context.Background()

	output := ConvertClaudeResponseToOpenAINonStream(ctx, "", nil, nil, []byte(
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_3\",\"model\":\"claude-3-5-sonnet\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n"+
			"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_3\",\"name\":\"lookup\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"hi\\\"}\"}}\n"+
			"data: {\"type\":\"content_block_stop\",\"index\":0}\n"+
			"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":9,\"output_tokens\":4,\"cache_read_input_tokens\":2}}\n",
	), nil)

	root := mustParseClaudeOpenAIObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "id"); got != "msg_3" {
		t.Fatalf("id = %q, want %q", got, "msg_3")
	}
	if got, _ := jsonutil.String(root, "model"); got != "claude-3-5-sonnet" {
		t.Fatalf("model = %q, want %q", got, "claude-3-5-sonnet")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.content"); got != "hello" {
		t.Fatalf("choices.0.message.content = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.reasoning"); got != "plan" {
		t.Fatalf("choices.0.message.reasoning = %q, want %q", got, "plan")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.tool_calls.0.function.name"); got != "lookup" {
		t.Fatalf("choices.0.message.tool_calls.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "choices.0.message.tool_calls.0.function.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("choices.0.message.tool_calls.0.function.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(root, "choices.0.finish_reason"); got != "tool_calls" {
		t.Fatalf("choices.0.finish_reason = %q, want %q", got, "tool_calls")
	}
	if got, _ := jsonutil.String(root, "usage.prompt_tokens_details.cached_tokens"); got != "2" {
		t.Fatalf("usage.prompt_tokens_details.cached_tokens = %q, want %q", got, "2")
	}
}

func mustParseClaudeOpenAIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
