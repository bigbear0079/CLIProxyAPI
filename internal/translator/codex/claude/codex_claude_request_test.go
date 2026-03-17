package claude

import (
	"fmt"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertClaudeRequestToCodex_SystemMessageScenarios(t *testing.T) {
	tests := []struct {
		name             string
		inputJSON        string
		wantHasDeveloper bool
		wantTexts        []string
	}{
		{
			name: "No system field",
			inputJSON: `{
				"model": "claude-3-opus",
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantHasDeveloper: false,
		},
		{
			name: "Empty string system field",
			inputJSON: `{
				"model": "claude-3-opus",
				"system": "",
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantHasDeveloper: false,
		},
		{
			name: "String system field",
			inputJSON: `{
				"model": "claude-3-opus",
				"system": "Be helpful",
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantHasDeveloper: true,
			wantTexts:        []string{"Be helpful"},
		},
		{
			name: "Array system field with filtered billing header",
			inputJSON: `{
				"model": "claude-3-opus",
				"system": [
					{"type": "text", "text": "x-anthropic-billing-header: tenant-123"},
					{"type": "text", "text": "Block 1"},
					{"type": "text", "text": "Block 2"}
				],
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantHasDeveloper: true,
			wantTexts:        []string{"Block 1", "Block 2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertClaudeRequestToCodex("test-model", []byte(tt.inputJSON), false)
			root := mustParseCodexClaudeObject(t, result)
			inputs, ok := jsonutil.Array(root, "input")
			if !ok {
				t.Fatalf("input missing in output: %s", string(result))
			}

			hasDeveloper := false
			if len(inputs) > 0 {
				if first, ok := inputs[0].(map[string]any); ok {
					role, _ := jsonutil.String(first, "role")
					hasDeveloper = role == "developer"
				}
			}
			if hasDeveloper != tt.wantHasDeveloper {
				t.Fatalf("got hasDeveloper = %v, want %v. Output: %v", hasDeveloper, tt.wantHasDeveloper, inputs)
			}

			if !tt.wantHasDeveloper {
				return
			}

			content, ok := jsonutil.Array(inputs[0], "content")
			if !ok {
				t.Fatalf("content missing in first input: %#v", inputs[0])
			}
			if len(content) != len(tt.wantTexts) {
				t.Fatalf("got %d system content items, want %d. Content: %#v", len(content), len(tt.wantTexts), content)
			}

			for i, wantText := range tt.wantTexts {
				if gotType, _ := jsonutil.String(content[i], "type"); gotType != "input_text" {
					t.Fatalf("content[%d] type = %q, want %q", i, gotType, "input_text")
				}
				if gotText, _ := jsonutil.String(content[i], "text"); gotText != wantText {
					t.Fatalf("content[%d] text = %q, want %q", i, gotText, wantText)
				}
			}
		})
	}
}

func TestConvertClaudeRequestToCodex_ParallelToolCalls(t *testing.T) {
	tests := []struct {
		name                  string
		inputJSON             string
		wantParallelToolCalls bool
	}{
		{
			name: "Default to true when tool_choice.disable_parallel_tool_use is absent",
			inputJSON: `{
				"model": "claude-3-opus",
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantParallelToolCalls: true,
		},
		{
			name: "Disable parallel tool calls when client opts out",
			inputJSON: `{
				"model": "claude-3-opus",
				"tool_choice": {"disable_parallel_tool_use": true},
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantParallelToolCalls: false,
		},
		{
			name: "Keep parallel tool calls enabled when client explicitly allows them",
			inputJSON: `{
				"model": "claude-3-opus",
				"tool_choice": {"disable_parallel_tool_use": false},
				"messages": [{"role": "user", "content": "hello"}]
			}`,
			wantParallelToolCalls: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertClaudeRequestToCodex("test-model", []byte(tt.inputJSON), false)
			root := mustParseCodexClaudeObject(t, result)

			if got, _ := jsonutil.Bool(root, "parallel_tool_calls"); got != tt.wantParallelToolCalls {
				t.Fatalf("parallel_tool_calls = %v, want %v. Output: %s", got, tt.wantParallelToolCalls, string(result))
			}
		})
	}
}

func TestConvertClaudeRequestToCodexTransformsToolUseAndToolResult(t *testing.T) {
	longToolName := "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"
	input := []byte(fmt.Sprintf(`{
		"tools": [
			{
				"name": "%s",
				"description": "Lookup data",
				"input_schema": {
					"type": "object",
					"properties": {
						"q": {"type": "string"}
					}
				}
			}
		],
		"messages": [
			{
				"role": "assistant",
				"content": [
					{"type": "text", "text": "Working"},
					{"type": "tool_use", "id": "toolu_1", "name": "%s", "input": {"q": "hi"}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "done"}]}
				]
			}
		]
	}`, longToolName, longToolName))

	output := ConvertClaudeRequestToCodex("test-model", input, false)
	root := mustParseCodexClaudeObject(t, output)

	if got, _ := jsonutil.String(root, "input.0.role"); got != "assistant" {
		t.Fatalf("input.0.role = %q, want %q", got, "assistant")
	}
	if got, _ := jsonutil.String(root, "input.0.content.0.type"); got != "output_text" {
		t.Fatalf("input.0.content.0.type = %q, want %q", got, "output_text")
	}
	if got, _ := jsonutil.String(root, "input.1.type"); got != "function_call" {
		t.Fatalf("input.1.type = %q, want %q", got, "function_call")
	}
	if got, _ := jsonutil.String(root, "input.1.call_id"); got != "toolu_1" {
		t.Fatalf("input.1.call_id = %q, want %q", got, "toolu_1")
	}
	if got, _ := jsonutil.String(root, "input.1.name"); got != "mcp__lookup" {
		t.Fatalf("input.1.name = %q, want %q", got, "mcp__lookup")
	}
	if got, _ := jsonutil.String(root, "input.1.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("input.1.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(root, "input.2.type"); got != "function_call_output" {
		t.Fatalf("input.2.type = %q, want %q", got, "function_call_output")
	}
	if got, _ := jsonutil.String(root, "input.2.output.0.type"); got != "input_text" {
		t.Fatalf("input.2.output.0.type = %q, want %q", got, "input_text")
	}
	if got, _ := jsonutil.String(root, "input.2.output.0.text"); got != "done" {
		t.Fatalf("input.2.output.0.text = %q, want %q", got, "done")
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
	if got, _ := jsonutil.String(root, "reasoning.effort"); got != "medium" {
		t.Fatalf("reasoning.effort = %q, want %q", got, "medium")
	}
}

func mustParseCodexClaudeObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
