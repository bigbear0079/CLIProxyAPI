package responses

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
)

func TestConvertOpenAIResponsesRequestToGeminiBuildsSystemInstructionsAndMixedRoles(t *testing.T) {
	input := []byte(`{
		"instructions":"base",
		"input":[
			{"role":"system","content":[{"text":"extra"}]},
			{"role":"user","content":[
				{"type":"input_text","text":"hello"},
				{"type":"output_text","text":"done"}
			]}
		],
		"temperature":0.3,
		"reasoning":{"effort":"high"}
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-test", input, false)
	root := mustParseGeminiResponsesObject(t, output)

	if got, _ := jsonutil.String(root, "systemInstruction.parts.0.text"); got != "base" {
		t.Fatalf("systemInstruction.parts.0.text = %q, want %q", got, "base")
	}
	if got, _ := jsonutil.String(root, "systemInstruction.parts.1.text"); got != "extra" {
		t.Fatalf("systemInstruction.parts.1.text = %q, want %q", got, "extra")
	}
	if got, _ := jsonutil.String(root, "contents.0.role"); got != "user" {
		t.Fatalf("contents.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.text"); got != "hello" {
		t.Fatalf("contents.0.parts.0.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "contents.1.role"); got != "model" {
		t.Fatalf("contents.1.role = %q, want %q", got, "model")
	}
	if got, _ := jsonutil.String(root, "contents.1.parts.0.text"); got != "done" {
		t.Fatalf("contents.1.parts.0.text = %q, want %q", got, "done")
	}
	if got, _ := jsonutil.String(root, "generationConfig.thinkingConfig.thinkingLevel"); got != "high" {
		t.Fatalf("generationConfig.thinkingConfig.thinkingLevel = %q, want %q", got, "high")
	}
	if got, ok := jsonutil.Bool(root, "generationConfig.thinkingConfig.includeThoughts"); !ok || !got {
		t.Fatalf("generationConfig.thinkingConfig.includeThoughts = (%v, %v), want (true, true)", got, ok)
	}

	safetySettings, ok := jsonutil.Array(root, "safetySettings")
	if !ok {
		t.Fatal("safetySettings missing")
	}
	if len(safetySettings) != len(common.DefaultSafetySettings()) {
		t.Fatalf("len(safetySettings) = %d, want %d", len(safetySettings), len(common.DefaultSafetySettings()))
	}
}

func TestConvertOpenAIResponsesRequestToGeminiNormalizesFunctionCallsAndOutputs(t *testing.T) {
	input := []byte(`{
		"input":[
			{"type":"function_call","call_id":"call_1","name":"alpha","arguments":"{\"q\":\"one\"}"},
			{"type":"function_call","call_id":"call_2","name":"beta","arguments":"{\"q\":\"two\"}"},
			{"type":"function_call_output","call_id":"call_2","output":"{\"ok\":true}"},
			{"type":"function_call_output","call_id":"call_1","output":"plain"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-test", input, false)
	root := mustParseGeminiResponsesObject(t, output)

	if got, _ := jsonutil.String(root, "contents.0.role"); got != "model" {
		t.Fatalf("contents.0.role = %q, want %q", got, "model")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.functionCall.name"); got != "alpha" {
		t.Fatalf("contents.0.parts.0.functionCall.name = %q, want %q", got, "alpha")
	}
	if got, _ := jsonutil.String(root, "contents.1.role"); got != "function" {
		t.Fatalf("contents.1.role = %q, want %q", got, "function")
	}
	if got, _ := jsonutil.String(root, "contents.1.parts.0.functionResponse.name"); got != "alpha" {
		t.Fatalf("contents.1.parts.0.functionResponse.name = %q, want %q", got, "alpha")
	}
	if got, _ := jsonutil.String(root, "contents.1.parts.0.functionResponse.response.result"); got != "plain" {
		t.Fatalf("contents.1.parts.0.functionResponse.response.result = %q, want %q", got, "plain")
	}
	if got, _ := jsonutil.String(root, "contents.2.parts.0.functionCall.name"); got != "beta" {
		t.Fatalf("contents.2.parts.0.functionCall.name = %q, want %q", got, "beta")
	}
	if got, _ := jsonutil.String(root, "contents.3.parts.0.functionResponse.name"); got != "beta" {
		t.Fatalf("contents.3.parts.0.functionResponse.name = %q, want %q", got, "beta")
	}
	if got, ok := jsonutil.Bool(root, "contents.3.parts.0.functionResponse.response.result.ok"); !ok || !got {
		t.Fatalf("contents.3.parts.0.functionResponse.response.result.ok = (%v, %v), want (true, true)", got, ok)
	}
}

func TestConvertOpenAIResponsesRequestToGeminiBuildsToolsAndGenerationConfig(t *testing.T) {
	input := []byte(`{
		"input":"hello",
		"tools":[
			{"type":"function","name":"lookup","description":"desc","parameters":{"type":"object"}}
		],
		"max_output_tokens":123,
		"stop_sequences":["a","b"]
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-test", input, false)
	root := mustParseGeminiResponsesObject(t, output)

	if got, _ := jsonutil.String(root, "contents.0.role"); got != "user" {
		t.Fatalf("contents.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.text"); got != "hello" {
		t.Fatalf("contents.0.parts.0.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "tools.0.functionDeclarations.0.name"); got != "lookup" {
		t.Fatalf("tools.0.functionDeclarations.0.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "tools.0.functionDeclarations.0.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("tools.0.functionDeclarations.0.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if got, ok := jsonutil.Int64(root, "generationConfig.maxOutputTokens"); !ok || got != 123 {
		t.Fatalf("generationConfig.maxOutputTokens = (%d, %v), want (123, true)", got, ok)
	}
	stopSequences, ok := jsonutil.Array(root, "generationConfig.stopSequences")
	if !ok || len(stopSequences) != 2 {
		t.Fatalf("generationConfig.stopSequences = %#v, want 2-item []any", stopSequences)
	}
}

func mustParseGeminiResponsesObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
