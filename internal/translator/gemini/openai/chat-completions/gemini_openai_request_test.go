package chat_completions

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
)

func TestConvertOpenAIRequestToGeminiBuildsSystemInstructionAndSafety(t *testing.T) {
	input := []byte(`{
		"messages": [
			{"role":"system","content":"system prompt"},
			{"role":"user","content":"hello"}
		]
	}`)

	output := ConvertOpenAIRequestToGemini("gemini-test", input, false)
	root := mustParseGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "gemini-test" {
		t.Fatalf("model = %q, want %q", got, "gemini-test")
	}
	if got, _ := jsonutil.String(root, "systemInstruction.role"); got != "user" {
		t.Fatalf("systemInstruction.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "systemInstruction.parts.0.text"); got != "system prompt" {
		t.Fatalf("systemInstruction.parts.0.text = %q, want %q", got, "system prompt")
	}
	if got, _ := jsonutil.String(root, "contents.0.role"); got != "user" {
		t.Fatalf("contents.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.text"); got != "hello" {
		t.Fatalf("contents.0.parts.0.text = %q, want %q", got, "hello")
	}

	safetySettings, ok := jsonutil.Array(root, "safetySettings")
	if !ok {
		t.Fatal("safetySettings missing")
	}
	if len(safetySettings) != len(common.DefaultSafetySettings()) {
		t.Fatalf("len(safetySettings) = %d, want %d", len(safetySettings), len(common.DefaultSafetySettings()))
	}
}

func TestConvertOpenAIRequestToGeminiBuildsToolCallsAndResponses(t *testing.T) {
	input := []byte(`{
		"messages": [
			{
				"role":"assistant",
				"tool_calls":[
					{
						"id":"call_1",
						"type":"function",
						"function":{
							"name":"lookup",
							"arguments":"{\"q\":\"golang\"}"
						}
					}
				]
			},
			{
				"role":"tool",
				"tool_call_id":"call_1",
				"content":"{\"ok\":true}"
			}
		]
	}`)

	output := ConvertOpenAIRequestToGemini("gemini-test", input, false)
	root := mustParseGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "contents.0.role"); got != "model" {
		t.Fatalf("contents.0.role = %q, want %q", got, "model")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.functionCall.name"); got != "lookup" {
		t.Fatalf("contents.0.parts.0.functionCall.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.functionCall.args.q"); got != "golang" {
		t.Fatalf("contents.0.parts.0.functionCall.args.q = %q, want %q", got, "golang")
	}
	if got, _ := jsonutil.String(root, "contents.1.role"); got != "user" {
		t.Fatalf("contents.1.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "contents.1.parts.0.functionResponse.name"); got != "lookup" {
		t.Fatalf("contents.1.parts.0.functionResponse.name = %q, want %q", got, "lookup")
	}
	if got, ok := jsonutil.Bool(root, "contents.1.parts.0.functionResponse.response.result.ok"); !ok || !got {
		t.Fatalf("contents.1.parts.0.functionResponse.response.result.ok = (%v, %v), want (true, true)", got, ok)
	}
}

func TestConvertOpenAIRequestToGeminiBuildsToolsAndSchemas(t *testing.T) {
	input := []byte(`{
		"messages": [
			{"role":"user","content":"hello"}
		],
		"tools": [
			{
				"type":"function",
				"function":{
					"name":"lookup",
					"description":"Find data",
					"parameters":{
						"type":"object",
						"properties":{
							"q":{"type":"string"}
						}
					},
					"strict":true
				}
			},
			{
				"type":"function",
				"function":{
					"name":"ping"
				}
			},
			{
				"google_search":{}
			}
		]
	}`)

	output := ConvertOpenAIRequestToGemini("gemini-test", input, false)
	root := mustParseGeminiObject(t, output)

	functionDeclarations, ok := jsonutil.Array(root, "tools.0.functionDeclarations")
	if !ok {
		t.Fatal("tools.0.functionDeclarations missing")
	}
	if len(functionDeclarations) != 2 {
		t.Fatalf("len(tools.0.functionDeclarations) = %d, want %d", len(functionDeclarations), 2)
	}
	if got, _ := jsonutil.String(root, "tools.0.functionDeclarations.0.name"); got != "lookup" {
		t.Fatalf("tools.0.functionDeclarations.0.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "tools.0.functionDeclarations.0.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("tools.0.functionDeclarations.0.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if jsonutil.Exists(root, "tools.0.functionDeclarations.0.strict") {
		t.Fatal("tools.0.functionDeclarations.0.strict should be removed")
	}
	if got, _ := jsonutil.String(root, "tools.0.functionDeclarations.1.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("tools.0.functionDeclarations.1.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if !jsonutil.IsEmptyObject(root, "tools.0.functionDeclarations.1.parametersJsonSchema.properties") {
		t.Fatal("tools.0.functionDeclarations.1.parametersJsonSchema.properties should be an empty object")
	}
	if !jsonutil.Exists(root, "tools.1.googleSearch") {
		t.Fatal("tools.1.googleSearch missing")
	}
}

func mustParseGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
