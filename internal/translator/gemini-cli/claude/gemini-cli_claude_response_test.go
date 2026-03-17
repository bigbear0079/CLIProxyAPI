package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiCLIResponseToClaudeStreamBuildsToolUseAndFinalDelta(t *testing.T) {
	var param any

	chunks := ConvertGeminiCLIResponseToClaude(context.Background(), "", nil, nil, []byte(`{
		"response":{
			"modelVersion":"claude-test",
			"responseId":"resp_1",
			"candidates":[
				{"content":{"parts":[
					{"text":"hello"},
					{"functionCall":{"name":"lookup","args":{"q":"hi"}}}
				]},"finishReason":"STOP"}
			],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}
		}
	}`), &param)
	if len(chunks) != 1 {
		t.Fatalf("chunk count = %d, want %d", len(chunks), 1)
	}

	events, payloads := parseGeminiCLIClaudeSSEPayloads(t, chunks[0])
	if len(events) < 6 {
		t.Fatalf("expected at least 6 events, got %d: %#v", len(events), events)
	}
	if events[0] != "message_start" {
		t.Fatalf("events[0] = %q, want %q", events[0], "message_start")
	}
	if got, _ := jsonutil.String(payloads[0], "message.model"); got != "claude-test" {
		t.Fatalf("message.model = %q, want %q", got, "claude-test")
	}
	if got, _ := jsonutil.String(payloads[1], "content_block.type"); got != "text" {
		t.Fatalf("first content block type = %q, want %q", got, "text")
	}
	if got, _ := jsonutil.String(payloads[2], "delta.text"); got != "hello" {
		t.Fatalf("text delta = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(payloads[4], "content_block.type"); got != "tool_use" {
		t.Fatalf("tool block type = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(payloads[5], "delta.partial_json"); got != `{"q":"hi"}` {
		t.Fatalf("input_json_delta = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(payloads[len(payloads)-1], "delta.stop_reason"); got != "tool_use" {
		t.Fatalf("final stop_reason = %q, want %q", got, "tool_use")
	}
}

func TestConvertGeminiCLIResponseToClaudeNonStreamBuildsContent(t *testing.T) {
	output := ConvertGeminiCLIResponseToClaudeNonStream(context.Background(), "", nil, nil, []byte(`{
		"response":{
			"responseId":"resp_2",
			"modelVersion":"claude-test",
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":2},
			"candidates":[
				{"content":{"parts":[
					{"text":"think","thought":true},
					{"text":"hello"},
					{"functionCall":{"name":"lookup","args":{"q":"hi"}}}
				]},"finishReason":"STOP"}
			]
		}
	}`), nil)

	root := mustParseGeminiCLIClaudeObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "id"); got != "resp_2" {
		t.Fatalf("id = %q, want %q", got, "resp_2")
	}
	if got, _ := jsonutil.String(root, "content.0.type"); got != "thinking" {
		t.Fatalf("content.0.type = %q, want %q", got, "thinking")
	}
	if got, _ := jsonutil.String(root, "content.1.type"); got != "text" {
		t.Fatalf("content.1.type = %q, want %q", got, "text")
	}
	if got, _ := jsonutil.String(root, "content.2.type"); got != "tool_use" {
		t.Fatalf("content.2.type = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "content.2.input.q"); got != "hi" {
		t.Fatalf("content.2.input.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "stop_reason"); got != "tool_use" {
		t.Fatalf("stop_reason = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "usage.output_tokens"); got != "7" {
		t.Fatalf("usage.output_tokens = %q, want %q", got, "7")
	}
}

func parseGeminiCLIClaudeSSEPayloads(t *testing.T, s string) ([]string, []map[string]any) {
	t.Helper()

	lines := strings.Split(strings.TrimSpace(s), "\n")
	events := make([]string, 0)
	payloads := make([]map[string]any, 0)
	currentEvent := ""

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

func mustParseGeminiCLIClaudeObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
