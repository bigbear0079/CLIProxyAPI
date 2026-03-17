package responses

import (
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertOpenAIResponsesRequestToClaudeBuildsMessagesToolsAndToolChoice(t *testing.T) {
	input := []byte(`{
		"instructions":"be helpful",
		"input":[
			{"role":"user","content":[{"type":"input_text","text":"hello"}]},
			{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"hi\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"done"}
		],
		"tools":[
			{"type":"function","name":"lookup","description":"desc","parameters":{"type":"object"}}
		],
		"tool_choice":{"type":"function","function":{"name":"lookup"}},
		"max_output_tokens":123
	}`)

	output := ConvertOpenAIResponsesRequestToClaude("claude-test", input, false)
	root := mustParseClaudeResponsesObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "claude-test" {
		t.Fatalf("model = %q, want %q", got, "claude-test")
	}
	if got, ok := jsonutil.Int64(root, "max_tokens"); !ok || got != 123 {
		t.Fatalf("max_tokens = (%d, %v), want (123, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "metadata.user_id"); !strings.HasPrefix(got, "user_") {
		t.Fatalf("metadata.user_id = %q, want prefix %q", got, "user_")
	}
	if got, _ := jsonutil.String(root, "messages.0.role"); got != "user" {
		t.Fatalf("messages.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "messages.0.content"); got != "be helpful" {
		t.Fatalf("messages.0.content = %q, want %q", got, "be helpful")
	}
	if got, _ := jsonutil.String(root, "messages.1.content"); got != "hello" {
		t.Fatalf("messages.1.content = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "messages.2.role"); got != "assistant" {
		t.Fatalf("messages.2.role = %q, want %q", got, "assistant")
	}
	if got, _ := jsonutil.String(root, "messages.2.content.0.type"); got != "tool_use" {
		t.Fatalf("messages.2.content.0.type = %q, want %q", got, "tool_use")
	}
	if got, _ := jsonutil.String(root, "messages.2.content.0.name"); got != "lookup" {
		t.Fatalf("messages.2.content.0.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "messages.2.content.0.input.q"); got != "hi" {
		t.Fatalf("messages.2.content.0.input.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "messages.3.content.0.type"); got != "tool_result" {
		t.Fatalf("messages.3.content.0.type = %q, want %q", got, "tool_result")
	}
	if got, _ := jsonutil.String(root, "messages.3.content.0.tool_use_id"); got != "call_1" {
		t.Fatalf("messages.3.content.0.tool_use_id = %q, want %q", got, "call_1")
	}
	if got, _ := jsonutil.String(root, "messages.3.content.0.content"); got != "done" {
		t.Fatalf("messages.3.content.0.content = %q, want %q", got, "done")
	}
	if got, _ := jsonutil.String(root, "tools.0.name"); got != "lookup" {
		t.Fatalf("tools.0.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "tools.0.input_schema.type"); got != "object" {
		t.Fatalf("tools.0.input_schema.type = %q, want %q", got, "object")
	}
	if got, _ := jsonutil.String(root, "tool_choice.type"); got != "tool" {
		t.Fatalf("tool_choice.type = %q, want %q", got, "tool")
	}
	if got, _ := jsonutil.String(root, "tool_choice.name"); got != "lookup" {
		t.Fatalf("tool_choice.name = %q, want %q", got, "lookup")
	}
}

func TestConvertOpenAIResponsesRequestToClaudeExtractsSystemInputOnce(t *testing.T) {
	input := []byte(`{
		"input":[
			{"role":"system","content":[{"text":"sys1"},{"text":"sys2"}]},
			{"role":"user","content":[{"type":"input_text","text":"hello"}]}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToClaude("claude-test", input, false)
	root := mustParseClaudeResponsesObject(t, output)

	messages, ok := jsonutil.Array(root, "messages")
	if !ok || len(messages) != 2 {
		t.Fatalf("messages = %#v, want 2-item []any", messages)
	}
	if got, _ := jsonutil.String(root, "messages.0.content"); got != "sys1\nsys2" {
		t.Fatalf("messages.0.content = %q, want %q", got, "sys1\nsys2")
	}
	if got, _ := jsonutil.String(root, "messages.1.content"); got != "hello" {
		t.Fatalf("messages.1.content = %q, want %q", got, "hello")
	}
}

func TestConvertOpenAIResponsesRequestToClaudeBuildsImageFileAndRequiredToolChoice(t *testing.T) {
	input := []byte(`{
		"input":[
			{"role":"user","content":[
				{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="},
				{"type":"input_file","file_data":"data:application/pdf;base64,UEQ="}
			]}
		],
		"tool_choice":"required"
	}`)

	output := ConvertOpenAIResponsesRequestToClaude("claude-test", input, true)
	root := mustParseClaudeResponsesObject(t, output)

	if got, ok := jsonutil.Bool(root, "stream"); !ok || !got {
		t.Fatalf("stream = (%v, %v), want (true, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "messages.0.content.0.type"); got != "image" {
		t.Fatalf("messages.0.content.0.type = %q, want %q", got, "image")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.0.source.media_type"); got != "image/png" {
		t.Fatalf("messages.0.content.0.source.media_type = %q, want %q", got, "image/png")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.1.type"); got != "document" {
		t.Fatalf("messages.0.content.1.type = %q, want %q", got, "document")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.1.source.media_type"); got != "application/pdf" {
		t.Fatalf("messages.0.content.1.source.media_type = %q, want %q", got, "application/pdf")
	}
	if got, _ := jsonutil.String(root, "tool_choice.type"); got != "any" {
		t.Fatalf("tool_choice.type = %q, want %q", got, "any")
	}
}

func mustParseClaudeResponsesObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
