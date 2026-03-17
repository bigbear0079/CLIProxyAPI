package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

func TestConvertCodexResponseToClaudeStreamBuildsToolUseEvents(t *testing.T) {
	originalName := "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"
	originalRequest := []byte(`{"tools":[{"name":"` + originalName + `"}]}`)
	shortName := buildShortNameMap([]string{originalName})[originalName]

	var param any
	chunks := ConvertCodexResponseToClaude(context.Background(), "", originalRequest, nil, []byte(`data: {"type":"response.created","response":{"id":"resp_1","model":"claude-test"}}`), &param)
	if len(chunks) != 1 {
		t.Fatalf("response.created chunk count = %d, want %d", len(chunks), 1)
	}
	events, payloads := parseClaudeSSEPayloads(t, chunks[0])
	if len(events) != 1 || events[0] != "message_start" {
		t.Fatalf("events = %#v, want [message_start]", events)
	}
	if got, _ := jsonutil.String(payloads[0], "message.id"); got != "resp_1" {
		t.Fatalf("message.id = %q, want %q", got, "resp_1")
	}

	chunks = ConvertCodexResponseToClaude(context.Background(), "", originalRequest, nil, []byte(`data: {"type":"response.output_item.added","item":{"type":"function_call","name":"`+shortName+`","call_id":"call_1"}}`), &param)
	events, payloads = parseClaudeSSEPayloads(t, chunks[0])
	if len(events) != 2 || events[0] != "content_block_start" || events[1] != "content_block_delta" {
		t.Fatalf("events = %#v, want tool_use start/delta", events)
	}
	if got, _ := jsonutil.String(payloads[0], "content_block.name"); got != originalName {
		t.Fatalf("content_block.name = %q, want %q", got, originalName)
	}
	if got, _ := jsonutil.String(payloads[0], "content_block.id"); got != util.SanitizeClaudeToolID("call_1") {
		t.Fatalf("content_block.id = %q, want sanitized tool id", got)
	}
	if got, _ := jsonutil.String(payloads[1], "delta.partial_json"); got != "" {
		t.Fatalf("delta.partial_json = %q, want empty string", got)
	}

	chunks = ConvertCodexResponseToClaude(context.Background(), "", originalRequest, nil, []byte(`data: {"type":"response.function_call_arguments.done","arguments":"{\"q\":\"hi\"}"}`), &param)
	events, payloads = parseClaudeSSEPayloads(t, chunks[0])
	if len(events) != 1 || events[0] != "content_block_delta" {
		t.Fatalf("events = %#v, want [content_block_delta]", events)
	}
	if got, _ := jsonutil.String(payloads[0], "delta.partial_json"); got != `{"q":"hi"}` {
		t.Fatalf("delta.partial_json = %q, want %q", got, `{"q":"hi"}`)
	}

	chunks = ConvertCodexResponseToClaude(context.Background(), "", originalRequest, nil, []byte(`data: {"type":"response.output_item.done","item":{"type":"function_call"}}`), &param)
	events, _ = parseClaudeSSEPayloads(t, chunks[0])
	if len(events) != 1 || events[0] != "content_block_stop" {
		t.Fatalf("events = %#v, want [content_block_stop]", events)
	}

	chunks = ConvertCodexResponseToClaude(context.Background(), "", originalRequest, nil, []byte(`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":3}}}}`), &param)
	events, payloads = parseClaudeSSEPayloads(t, chunks[0])
	if len(events) != 2 || events[0] != "message_delta" || events[1] != "message_stop" {
		t.Fatalf("events = %#v, want message_delta/message_stop", events)
	}
	if got, _ := jsonutil.String(payloads[0], "delta.stop_reason"); got != "tool_use" {
		t.Fatalf("delta.stop_reason = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(payloads[0], "usage.input_tokens"); got != "7" {
		t.Fatalf("usage.input_tokens = %q, want %q", got, "7")
	}
	if got, _ := jsonutil.String(payloads[0], "usage.cache_read_input_tokens"); got != "3" {
		t.Fatalf("usage.cache_read_input_tokens = %q, want %q", got, "3")
	}
}

func TestConvertCodexResponseToClaudeNonStreamBuildsContent(t *testing.T) {
	originalName := "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"
	originalRequest := []byte(`{"tools":[{"name":"` + originalName + `"}]}`)
	shortName := buildShortNameMap([]string{originalName})[originalName]

	output := ConvertCodexResponseToClaudeNonStream(context.Background(), "", originalRequest, nil, []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_2",
			"model":"claude-test",
			"usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":3}},
			"output":[
				{"type":"reasoning","summary":[{"text":"think"}]},
				{"type":"message","content":[{"type":"output_text","text":"hello"}]},
				{"type":"function_call","name":"`+shortName+`","call_id":"call_2","arguments":"{\"q\":\"hi\"}"}
			]
		}
	}`), nil)

	root := mustParseCodexClaudeResponseObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "id"); got != "resp_2" {
		t.Fatalf("id = %q, want %q", got, "resp_2")
	}
	if got, _ := jsonutil.String(root, "content.0.type"); got != "thinking" {
		t.Fatalf("content.0.type = %q, want %q", got, "thinking")
	}
	if got, _ := jsonutil.String(root, "content.0.thinking"); got != "think" {
		t.Fatalf("content.0.thinking = %q, want %q", got, "think")
	}
	if got, _ := jsonutil.String(root, "content.1.type"); got != "text" {
		t.Fatalf("content.1.type = %q, want %q", got, "text")
	}
	if got, _ := jsonutil.String(root, "content.1.text"); got != "hello" {
		t.Fatalf("content.1.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "content.2.type"); got != "tool_use" {
		t.Fatalf("content.2.type = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "content.2.name"); got != originalName {
		t.Fatalf("content.2.name = %q, want %q", got, originalName)
	}
	if got, _ := jsonutil.String(root, "content.2.id"); got != util.SanitizeClaudeToolID("call_2") {
		t.Fatalf("content.2.id = %q, want sanitized tool id", got)
	}
	if got, _ := jsonutil.String(root, "content.2.input.q"); got != "hi" {
		t.Fatalf("content.2.input.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "stop_reason"); got != "tool_use" {
		t.Fatalf("stop_reason = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "usage.input_tokens"); got != "7" {
		t.Fatalf("usage.input_tokens = %q, want %q", got, "7")
	}
	if got, _ := jsonutil.String(root, "usage.cache_read_input_tokens"); got != "3" {
		t.Fatalf("usage.cache_read_input_tokens = %q, want %q", got, "3")
	}
}

func parseClaudeSSEPayloads(t *testing.T, s string) ([]string, []map[string]any) {
	t.Helper()

	lines := strings.Split(strings.TrimSpace(s), "\n")
	events := make([]string, 0)
	payloads := make([]map[string]any, 0)
	var currentEvent string

	for _, line := range lines {
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event: "))
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
			root, errParse := jsonutil.ParseObjectBytes([]byte(payload))
			if errParse != nil {
				t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, payload)
			}
			events = append(events, currentEvent)
			payloads = append(payloads, root)
		}
	}

	return events, payloads
}

func mustParseCodexClaudeResponseObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
