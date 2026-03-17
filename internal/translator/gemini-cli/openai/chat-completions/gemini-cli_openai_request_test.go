package chat_completions

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
)

func TestConvertOpenAIRequestToGeminiCLIBuildsSystemInstructionAndSafety(t *testing.T) {
	input := []byte(`{
		"messages": [
			{"role":"system","content":"system prompt"},
			{"role":"user","content":"hello"}
		]
	}`)

	output := ConvertOpenAIRequestToGeminiCLI("gemini-cli-test", input, false)
	root := mustParseGeminiCLIObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "gemini-cli-test" {
		t.Fatalf("model = %q, want %q", got, "gemini-cli-test")
	}
	if got, _ := jsonutil.String(root, "request.systemInstruction.role"); got != "user" {
		t.Fatalf("request.systemInstruction.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "request.systemInstruction.parts.0.text"); got != "system prompt" {
		t.Fatalf("request.systemInstruction.parts.0.text = %q, want %q", got, "system prompt")
	}
	if got, _ := jsonutil.String(root, "request.contents.0.role"); got != "user" {
		t.Fatalf("request.contents.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "request.contents.0.parts.0.text"); got != "hello" {
		t.Fatalf("request.contents.0.parts.0.text = %q, want %q", got, "hello")
	}

	safetySettings, ok := jsonutil.Array(root, "request.safetySettings")
	if !ok {
		t.Fatal("request.safetySettings missing")
	}
	if len(safetySettings) != len(common.DefaultSafetySettings()) {
		t.Fatalf("len(request.safetySettings) = %d, want %d", len(safetySettings), len(common.DefaultSafetySettings()))
	}
}

func TestConvertOpenAIRequestToGeminiCLIBuildsToolCallsAndResponses(t *testing.T) {
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

	output := ConvertOpenAIRequestToGeminiCLI("gemini-cli-test", input, false)
	root := mustParseGeminiCLIObject(t, output)

	if got, _ := jsonutil.String(root, "request.contents.0.role"); got != "model" {
		t.Fatalf("request.contents.0.role = %q, want %q", got, "model")
	}
	if got, _ := jsonutil.String(root, "request.contents.0.parts.0.functionCall.name"); got != "lookup" {
		t.Fatalf("request.contents.0.parts.0.functionCall.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "request.contents.0.parts.0.functionCall.args.q"); got != "golang" {
		t.Fatalf("request.contents.0.parts.0.functionCall.args.q = %q, want %q", got, "golang")
	}
	if got, _ := jsonutil.String(root, "request.contents.1.role"); got != "user" {
		t.Fatalf("request.contents.1.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "request.contents.1.parts.0.functionResponse.name"); got != "lookup" {
		t.Fatalf("request.contents.1.parts.0.functionResponse.name = %q, want %q", got, "lookup")
	}
	if got, ok := jsonutil.Bool(root, "request.contents.1.parts.0.functionResponse.response.result.ok"); !ok || !got {
		t.Fatalf("request.contents.1.parts.0.functionResponse.response.result.ok = (%v, %v), want (true, true)", got, ok)
	}
}

func TestConvertOpenAIRequestToGeminiCLIBuildsToolsAndSchemas(t *testing.T) {
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

	output := ConvertOpenAIRequestToGeminiCLI("gemini-cli-test", input, false)
	root := mustParseGeminiCLIObject(t, output)

	functionDeclarations, ok := jsonutil.Array(root, "request.tools.0.functionDeclarations")
	if !ok {
		t.Fatal("request.tools.0.functionDeclarations missing")
	}
	if len(functionDeclarations) != 2 {
		t.Fatalf("len(request.tools.0.functionDeclarations) = %d, want %d", len(functionDeclarations), 2)
	}
	if got, _ := jsonutil.String(root, "request.tools.0.functionDeclarations.0.name"); got != "lookup" {
		t.Fatalf("request.tools.0.functionDeclarations.0.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "request.tools.0.functionDeclarations.0.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("request.tools.0.functionDeclarations.0.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if jsonutil.Exists(root, "request.tools.0.functionDeclarations.0.strict") {
		t.Fatal("request.tools.0.functionDeclarations.0.strict should be removed")
	}
	if got, _ := jsonutil.String(root, "request.tools.0.functionDeclarations.1.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("request.tools.0.functionDeclarations.1.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if !jsonutil.IsEmptyObject(root, "request.tools.0.functionDeclarations.1.parametersJsonSchema.properties") {
		t.Fatal("request.tools.0.functionDeclarations.1.parametersJsonSchema.properties should be an empty object")
	}
	if !jsonutil.Exists(root, "request.tools.1.googleSearch") {
		t.Fatal("request.tools.1.googleSearch missing")
	}
}

func mustParseGeminiCLIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
