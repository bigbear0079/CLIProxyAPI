package geminiCLI

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiCLIRequestToGeminiUnwrapsAndNormalizes(t *testing.T) {
	input := []byte(`{
		"model":"gemini-2.5-pro",
		"request":{
			"systemInstruction":{"parts":[{"text":"sys"}]},
			"contents":[
				{"role":"model","parts":[
					{"functionCall":{"name":"lookup"}},
					{"thoughtSignature":"original"}
				]}
			],
			"tools":[
				{"function_declarations":[
					{"name":"lookup","parameters":{"type":"object"}}
				]}
			]
		}
	}`)

	output := ConvertGeminiCLIRequestToGemini("", input, false)
	root := mustParseGeminiCLIToGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "gemini-2.5-pro" {
		t.Fatalf("model = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(root, "system_instruction.parts.0.text"); got != "sys" {
		t.Fatalf("system_instruction.parts.0.text = %q, want %q", got, "sys")
	}
	if jsonutil.Exists(root, "systemInstruction") {
		t.Fatalf("systemInstruction should be removed: %s", string(output))
	}
	if got, _ := jsonutil.String(root, "tools.0.function_declarations.0.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("tools.0.function_declarations.0.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.0.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("contents.0.parts.0.thoughtSignature = %q, want validator skip marker", got)
	}
	if got, _ := jsonutil.String(root, "contents.0.parts.1.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("contents.0.parts.1.thoughtSignature = %q, want validator skip marker", got)
	}
	if !jsonutil.Exists(root, "safetySettings") {
		t.Fatalf("safetySettings missing in output: %s", string(output))
	}
}

func mustParseGeminiCLIToGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
