package gemini

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiRequestToGeminiNormalizesRequestTree(t *testing.T) {
	input := []byte(`{
		"tools":[
			{"functionDeclarations":[
				{"name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}
			]}
		],
		"contents":[
			{"parts":[{"text":"hello"}]},
			{"parts":[
				{"functionCall":{"name":"lookup"}},
				{"thoughtSignature":"original"}
			]},
			{"role":"user","parts":[
				{"functionResponse":{"name":"","response":{"result":"done"}}}
			]}
		],
		"generationConfig":{"responseSchema":{"type":"object"}}
	}`)

	output := ConvertGeminiRequestToGemini("", input, false)
	root := mustParseGeminiGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "contents.0.role"); got != "user" {
		t.Fatalf("contents.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "contents.1.role"); got != "model" {
		t.Fatalf("contents.1.role = %q, want %q", got, "model")
	}
	if got, _ := jsonutil.String(root, "contents.1.parts.0.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("contents.1.parts.0.thoughtSignature = %q, want validator skip marker", got)
	}
	if got, _ := jsonutil.String(root, "contents.1.parts.1.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("contents.1.parts.1.thoughtSignature = %q, want validator skip marker", got)
	}
	if got, _ := jsonutil.String(root, "contents.2.parts.0.functionResponse.name"); got != "lookup" {
		t.Fatalf("contents.2.parts.0.functionResponse.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "tools.0.function_declarations.0.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("tools.0.function_declarations.0.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if !jsonutil.Exists(root, "generationConfig.responseJsonSchema") {
		t.Fatalf("generationConfig.responseJsonSchema missing in output: %s", string(output))
	}
	if jsonutil.Exists(root, "generationConfig.responseSchema") {
		t.Fatalf("generationConfig.responseSchema should be removed: %s", string(output))
	}
	if !jsonutil.Exists(root, "safetySettings") {
		t.Fatalf("safetySettings missing in output: %s", string(output))
	}
}

func mustParseGeminiGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
