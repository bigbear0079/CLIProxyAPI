package gemini

import (
	"fmt"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiRequestToCodexBuildsInputToolsAndReasoning(t *testing.T) {
	longToolName := "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"
	input := []byte(fmt.Sprintf(`{
		"system_instruction":{"parts":[{"text":"sys1"},{"text":"sys2"}]},
		"contents":[
			{"role":"user","parts":[{"text":"hello"}]},
			{"role":"model","parts":[
				{"text":"working"},
				{"functionCall":{"name":"%s","args":{"q":"hi"}}}
			]},
			{"role":"user","parts":[
				{"functionResponse":{"response":{"result":"done"}}}
			]}
		],
		"tools":[
			{"functionDeclarations":[
				{"name":"%s","description":"Lookup data","parametersJsonSchema":{"type":"OBJECT","properties":{"q":{"type":"STRING"}},"$schema":"x"}}
			]}
		],
		"generationConfig":{"thinkingConfig":{"thinkingBudget":0}}
	}`, longToolName, longToolName))

	output := ConvertGeminiRequestToCodex("codex-test", input, false)
	root := mustParseCodexGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "codex-test" {
		t.Fatalf("model = %q, want %q", got, "codex-test")
	}
	if got, _ := jsonutil.String(root, "input.0.role"); got != "developer" {
		t.Fatalf("input.0.role = %q, want %q", got, "developer")
	}
	if got, _ := jsonutil.String(root, "input.0.content.0.text"); got != "sys1" {
		t.Fatalf("input.0.content.0.text = %q, want %q", got, "sys1")
	}
	if got, _ := jsonutil.String(root, "input.0.content.1.text"); got != "sys2" {
		t.Fatalf("input.0.content.1.text = %q, want %q", got, "sys2")
	}
	if got, _ := jsonutil.String(root, "input.1.role"); got != "user" {
		t.Fatalf("input.1.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "input.2.role"); got != "assistant" {
		t.Fatalf("input.2.role = %q, want %q", got, "assistant")
	}
	if got, _ := jsonutil.String(root, "input.2.content.0.type"); got != "output_text" {
		t.Fatalf("input.2.content.0.type = %q, want %q", got, "output_text")
	}
	if got, _ := jsonutil.String(root, "input.3.type"); got != "function_call" {
		t.Fatalf("input.3.type = %q, want %q", got, "function_call")
	}
	if got, _ := jsonutil.String(root, "input.3.name"); got != "mcp__lookup" {
		t.Fatalf("input.3.name = %q, want %q", got, "mcp__lookup")
	}
	if got, _ := jsonutil.String(root, "input.3.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("input.3.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	callID, _ := jsonutil.String(root, "input.3.call_id")
	if !strings.HasPrefix(callID, "call_") {
		t.Fatalf("input.3.call_id = %q, want prefix %q", callID, "call_")
	}
	if got, _ := jsonutil.String(root, "input.4.type"); got != "function_call_output" {
		t.Fatalf("input.4.type = %q, want %q", got, "function_call_output")
	}
	if got, _ := jsonutil.String(root, "input.4.call_id"); got != callID {
		t.Fatalf("input.4.call_id = %q, want %q", got, callID)
	}
	if got, _ := jsonutil.String(root, "input.4.output"); got != "done" {
		t.Fatalf("input.4.output = %q, want %q", got, "done")
	}
	if got, _ := jsonutil.String(root, "tools.0.name"); got != "mcp__lookup" {
		t.Fatalf("tools.0.name = %q, want %q", got, "mcp__lookup")
	}
	if got, _ := jsonutil.String(root, "tools.0.parameters.type"); got != "object" {
		t.Fatalf("tools.0.parameters.type = %q, want %q", got, "object")
	}
	if got, _ := jsonutil.String(root, "tools.0.parameters.properties.q.type"); got != "string" {
		t.Fatalf("tools.0.parameters.properties.q.type = %q, want %q", got, "string")
	}
	if got, ok := jsonutil.Bool(root, "tools.0.parameters.additionalProperties"); !ok || got {
		t.Fatalf("tools.0.parameters.additionalProperties = (%v, %v), want (false, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "reasoning.effort"); got != "none" {
		t.Fatalf("reasoning.effort = %q, want %q", got, "none")
	}
	if got, ok := jsonutil.Bool(root, "parallel_tool_calls"); !ok || !got {
		t.Fatalf("parallel_tool_calls = (%v, %v), want (true, true)", got, ok)
	}
}

func TestConvertGeminiRequestToCodexDefaultsReasoningToMedium(t *testing.T) {
	output := ConvertGeminiRequestToCodex("codex-test", []byte(`{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`), false)
	root := mustParseCodexGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "reasoning.effort"); got != "medium" {
		t.Fatalf("reasoning.effort = %q, want %q", got, "medium")
	}
}

func mustParseCodexGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
