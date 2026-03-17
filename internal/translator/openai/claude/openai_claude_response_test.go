package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertOpenAIResponseToClaude_StreamBuildsAnthropicEvents(t *testing.T) {
	ctx := context.Background()
	var param any
	originalRequest := []byte(`{
		"stream": true,
		"tools": [
			{"name":"Lookup"}
		]
	}`)

	out := ConvertOpenAIResponseToClaude(ctx, "", originalRequest, nil, []byte(`data: {"id":"resp_1","model":"gpt-5","created":1700000000,"choices":[{"delta":{"reasoning_content":[{"text":"plan"}]}}]}`), &param)
	if len(out) != 3 {
		t.Fatalf("reasoning chunk count = %d, want %d", len(out), 3)
	}
	startRoot := mustParseOpenAIClaudeSSEPayload(t, out[0])
	if got, _ := jsonutil.String(startRoot, "message.id"); got != "resp_1" {
		t.Fatalf("message.id = %q, want %q", got, "resp_1")
	}
	if got, _ := jsonutil.String(startRoot, "message.model"); got != "gpt-5" {
		t.Fatalf("message.model = %q, want %q", got, "gpt-5")
	}
	reasoningDeltaRoot := mustParseOpenAIClaudeSSEPayload(t, out[2])
	if got, _ := jsonutil.String(reasoningDeltaRoot, "delta.thinking"); got != "plan" {
		t.Fatalf("delta.thinking = %q, want %q", got, "plan")
	}

	out = ConvertOpenAIResponseToClaude(ctx, "", originalRequest, nil, []byte(`data: {"choices":[{"delta":{"content":"hello"}}]}`), &param)
	if len(out) != 3 {
		t.Fatalf("text chunk count = %d, want %d", len(out), 3)
	}
	textDeltaRoot := mustParseOpenAIClaudeSSEPayload(t, out[2])
	if got, _ := jsonutil.String(textDeltaRoot, "delta.text"); got != "hello" {
		t.Fatalf("delta.text = %q, want %q", got, "hello")
	}

	out = ConvertOpenAIResponseToClaude(ctx, "", originalRequest, nil, []byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"hi\"}"}}]}}]}`), &param)
	if len(out) != 2 {
		t.Fatalf("tool chunk count = %d, want %d", len(out), 2)
	}
	toolStartRoot := mustParseOpenAIClaudeSSEPayload(t, out[1])
	if got, _ := jsonutil.String(toolStartRoot, "content_block.name"); got != "Lookup" {
		t.Fatalf("content_block.name = %q, want %q", got, "Lookup")
	}

	out = ConvertOpenAIResponseToClaude(ctx, "", originalRequest, nil, []byte(`data: {"choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3}}}`), &param)
	if len(out) != 4 {
		t.Fatalf("finish chunk count = %d, want %d", len(out), 4)
	}
	inputDeltaRoot := mustParseOpenAIClaudeSSEPayload(t, out[0])
	if got, _ := jsonutil.String(inputDeltaRoot, "delta.partial_json"); got != `{"q":"hi"}` {
		t.Fatalf("delta.partial_json = %q, want %q", got, `{"q":"hi"}`)
	}
	messageDeltaRoot := mustParseOpenAIClaudeSSEPayload(t, out[2])
	if got, _ := jsonutil.String(messageDeltaRoot, "delta.stop_reason"); got != "tool_use" {
		t.Fatalf("delta.stop_reason = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(messageDeltaRoot, "usage.input_tokens"); got != "7" {
		t.Fatalf("usage.input_tokens = %q, want %q", got, "7")
	}
	if got, _ := jsonutil.String(messageDeltaRoot, "usage.cache_read_input_tokens"); got != "3" {
		t.Fatalf("usage.cache_read_input_tokens = %q, want %q", got, "3")
	}
	messageStopRoot := mustParseOpenAIClaudeSSEPayload(t, out[3])
	if got, _ := jsonutil.String(messageStopRoot, "type"); got != "message_stop" {
		t.Fatalf("message_stop type = %q, want %q", got, "message_stop")
	}
}

func TestConvertOpenAIResponseToClaudeNonStreamBuildsMappedToolUse(t *testing.T) {
	originalRequest := []byte(`{
		"tools": [
			{"name":"Lookup"}
		]
	}`)

	output := ConvertOpenAIResponseToClaudeNonStream(context.Background(), "", originalRequest, nil, []byte(`{
		"id":"resp_2",
		"model":"gpt-5",
		"choices":[{
			"message":{
				"content":[
					{"type":"reasoning","text":"plan"},
					{"type":"text","text":"hello"},
					{"type":"tool_calls","tool_calls":[
						{"id":"call_2","function":{"name":"lookup","arguments":"{\"q\":\"hi\"}"}}
					]}
				]
			}
		}],
		"usage":{
			"prompt_tokens":8,
			"completion_tokens":3,
			"prompt_tokens_details":{"cached_tokens":2}
		}
	}`), nil)

	root := mustParseOpenAIClaudeObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "id"); got != "resp_2" {
		t.Fatalf("id = %q, want %q", got, "resp_2")
	}
	if got, _ := jsonutil.String(root, "model"); got != "gpt-5" {
		t.Fatalf("model = %q, want %q", got, "gpt-5")
	}
	if got, _ := jsonutil.String(root, "content.0.thinking"); got != "plan" {
		t.Fatalf("content.0.thinking = %q, want %q", got, "plan")
	}
	if got, _ := jsonutil.String(root, "content.1.text"); got != "hello" {
		t.Fatalf("content.1.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "content.2.name"); got != "Lookup" {
		t.Fatalf("content.2.name = %q, want %q", got, "Lookup")
	}
	if got, _ := jsonutil.String(root, "content.2.input.q"); got != "hi" {
		t.Fatalf("content.2.input.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "stop_reason"); got != "tool_use" {
		t.Fatalf("stop_reason = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "usage.input_tokens"); got != "6" {
		t.Fatalf("usage.input_tokens = %q, want %q", got, "6")
	}
	if got, _ := jsonutil.String(root, "usage.cache_read_input_tokens"); got != "2" {
		t.Fatalf("usage.cache_read_input_tokens = %q, want %q", got, "2")
	}
}

func mustParseOpenAIClaudeSSEPayload(t *testing.T, event string) map[string]any {
	t.Helper()

	lines := strings.Split(event, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			return mustParseOpenAIClaudeObject(t, []byte(strings.TrimSpace(strings.TrimPrefix(line, "data: "))))
		}
	}
	t.Fatalf("event missing data payload: %q", event)
	return nil
}

func mustParseOpenAIClaudeObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
