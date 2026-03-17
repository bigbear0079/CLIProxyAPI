package gemini

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiRequestToGeminiCLIWrapsAndGroupsToolResponses(t *testing.T) {
	input := []byte(`{
		"model":"gemini-2.5-pro",
		"system_instruction":{"parts":[{"text":"sys"}]},
		"contents":[
			{"parts":[{"text":"hello"}]},
			{"role":"model","parts":[
				{"functionCall":{"name":"lookup"}},
				{"thoughtSignature":"original"}
			]},
			{"role":"user","parts":[
				{"functionResponse":{"name":"","response":{"result":"done"}}}
			]},
			{"role":"user","parts":[]}
		],
		"tools":[
			{"function_declarations":[
				{"name":"lookup","parameters":{"type":"object"}}
			]}
		]
	}`)

	output := ConvertGeminiRequestToGeminiCLI("", input, false)
	root := mustParseGeminiToGeminiCLIObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "gemini-2.5-pro" {
		t.Fatalf("model = %q, want %q", got, "gemini-2.5-pro")
	}
	if jsonutil.Exists(root, "request.model") {
		t.Fatalf("request.model should be removed: %s", string(output))
	}
	if got, _ := jsonutil.String(root, "request.systemInstruction.parts.0.text"); got != "sys" {
		t.Fatalf("request.systemInstruction.parts.0.text = %q, want %q", got, "sys")
	}
	if jsonutil.Exists(root, "request.system_instruction") {
		t.Fatalf("request.system_instruction should be removed: %s", string(output))
	}
	if got, _ := jsonutil.String(root, "request.contents.0.role"); got != "user" {
		t.Fatalf("request.contents.0.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "request.contents.1.role"); got != "model" {
		t.Fatalf("request.contents.1.role = %q, want %q", got, "model")
	}
	if got, _ := jsonutil.String(root, "request.contents.1.parts.0.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("request.contents.1.parts.0.thoughtSignature = %q, want validator skip marker", got)
	}
	if got, _ := jsonutil.String(root, "request.contents.1.parts.1.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("request.contents.1.parts.1.thoughtSignature = %q, want validator skip marker", got)
	}
	if got, _ := jsonutil.String(root, "request.contents.2.role"); got != "user" {
		t.Fatalf("request.contents.2.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "request.contents.2.parts.0.functionResponse.name"); got != "lookup" {
		t.Fatalf("request.contents.2.parts.0.functionResponse.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "request.tools.0.function_declarations.0.parametersJsonSchema.type"); got != "object" {
		t.Fatalf("request.tools.0.function_declarations.0.parametersJsonSchema.type = %q, want %q", got, "object")
	}
	contents, ok := jsonutil.Array(root, "request.contents")
	if !ok || len(contents) != 3 {
		t.Fatalf("request.contents length = (%d, %v), want (3, true)", len(contents), ok)
	}
	if !jsonutil.Exists(root, "request.safetySettings") {
		t.Fatalf("request.safetySettings missing in output: %s", string(output))
	}
}

func mustParseGeminiToGeminiCLIObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
