package gemini

import (
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiRequestToClaudeBuildsMessagesAndPairsToolResults(t *testing.T) {
	input := []byte(`{
		"system_instruction":{"parts":[{"text":"sys"}]},
		"generationConfig":{
			"maxOutputTokens":321,
			"temperature":0.4,
			"stopSequences":["a"],
			"thinkingConfig":{"includeThoughts":true}
		},
		"contents":[
			{"role":"user","parts":[
				{"text":"hello"},
				{"inline_data":{"mime_type":"image/png","data":"aGVsbG8="}}
			]},
			{"role":"model","parts":[
				{"functionCall":{"name":"lookup","args":{"q":"hi"}}}
			]},
			{"role":"function","parts":[
				{"functionResponse":{"response":{"result":"done"}}}
			]}
		],
		"tools":[
			{"functionDeclarations":[
				{"name":"lookup","description":"desc","parametersJsonSchema":{"type":"OBJECT","properties":{"q":{"type":"STRING"}}}}
			]}
		],
		"tool_config":{"function_calling_config":{"mode":"ANY"}}
	}`)

	output := ConvertGeminiRequestToClaude("claude-test", input, true)
	root := mustParseClaudeGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "claude-test" {
		t.Fatalf("model = %q, want %q", got, "claude-test")
	}
	if got, ok := jsonutil.Bool(root, "stream"); !ok || !got {
		t.Fatalf("stream = (%v, %v), want (true, true)", got, ok)
	}
	if got, ok := jsonutil.Int64(root, "max_tokens"); !ok || got != 321 {
		t.Fatalf("max_tokens = (%d, %v), want (321, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "temperature"); got != "0.4" {
		t.Fatalf("temperature = %q, want %q", got, "0.4")
	}
	if got, _ := jsonutil.String(root, "thinking.type"); got != "enabled" {
		t.Fatalf("thinking.type = %q, want %q", got, "enabled")
	}
	if got, _ := jsonutil.String(root, "metadata.user_id"); !strings.HasPrefix(got, "user_") {
		t.Fatalf("metadata.user_id = %q, want prefix %q", got, "user_")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.0.text"); got != "sys" {
		t.Fatalf("messages.0.content.0.text = %q, want %q", got, "sys")
	}
	if got, _ := jsonutil.String(root, "messages.1.role"); got != "user" {
		t.Fatalf("messages.1.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "messages.1.content.0.text"); got != "hello" {
		t.Fatalf("messages.1.content.0.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "messages.1.content.1.type"); got != "image" {
		t.Fatalf("messages.1.content.1.type = %q, want %q", got, "image")
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
	toolUseID, _ := jsonutil.String(root, "messages.2.content.0.id")
	if !strings.HasPrefix(toolUseID, "toolu_") {
		t.Fatalf("messages.2.content.0.id = %q, want prefix %q", toolUseID, "toolu_")
	}
	if got, _ := jsonutil.String(root, "messages.3.content.0.type"); got != "tool_result" {
		t.Fatalf("messages.3.content.0.type = %q, want %q", got, "tool_result")
	}
	if got, _ := jsonutil.String(root, "messages.3.content.0.tool_use_id"); got != toolUseID {
		t.Fatalf("messages.3.content.0.tool_use_id = %q, want %q", got, toolUseID)
	}
	if got, _ := jsonutil.String(root, "messages.3.content.0.content"); got != "done" {
		t.Fatalf("messages.3.content.0.content = %q, want %q", got, "done")
	}
	if got, _ := jsonutil.String(root, "tools.0.input_schema.type"); got != "object" {
		t.Fatalf("tools.0.input_schema.type = %q, want %q", got, "object")
	}
	if got, _ := jsonutil.String(root, "tools.0.input_schema.properties.q.type"); got != "string" {
		t.Fatalf("tools.0.input_schema.properties.q.type = %q, want %q", got, "string")
	}
	if got, _ := jsonutil.String(root, "tool_choice.type"); got != "any" {
		t.Fatalf("tool_choice.type = %q, want %q", got, "any")
	}
}

func TestConvertGeminiRequestToClaudeSpecificToolChoiceAndFileData(t *testing.T) {
	input := []byte(`{
		"contents":[
			{"role":"user","parts":[
				{"file_data":{"file_uri":"file:///tmp/a.txt","mime_type":"text/plain"}}
			]}
		],
		"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["lookup"]}}
	}`)

	output := ConvertGeminiRequestToClaude("claude-test", input, false)
	root := mustParseClaudeGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "messages.0.content.0.text"); got != "File: file:///tmp/a.txt (Type: text/plain)" {
		t.Fatalf("messages.0.content.0.text = %q, want file info text", got)
	}
	if got, _ := jsonutil.String(root, "tool_choice.type"); got != "tool" {
		t.Fatalf("tool_choice.type = %q, want %q", got, "tool")
	}
	if got, _ := jsonutil.String(root, "tool_choice.name"); got != "lookup" {
		t.Fatalf("tool_choice.name = %q, want %q", got, "lookup")
	}
}

func mustParseClaudeGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
