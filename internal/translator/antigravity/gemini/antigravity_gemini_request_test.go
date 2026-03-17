package gemini

import (
	"fmt"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiRequestToAntigravity_PreserveValidSignature(t *testing.T) {
	validSignature := "abc123validSignature1234567890123456789012345678901234567890"
	inputJSON := []byte(fmt.Sprintf(`{
		"model": "gemini-3-pro-preview",
		"contents": [
			{
				"role": "model",
				"parts": [
					{"functionCall": {"name": "test_tool", "args": {}}, "thoughtSignature": "%s"}
				]
			}
		]
	}`, validSignature))

	output := ConvertGeminiRequestToAntigravity("gemini-3-pro-preview", inputJSON, false)
	root := mustParseAntigravityGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "request.contents.0.parts.0.thoughtSignature"); got != validSignature {
		t.Fatalf("thoughtSignature = %q, want %q", got, validSignature)
	}
}

func TestConvertGeminiRequestToAntigravity_AddSkipSentinelToFunctionCall(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-pro-preview",
		"contents": [
			{
				"role": "model",
				"parts": [
					{"functionCall": {"name": "test_tool", "args": {}}}
				]
			}
		]
	}`)

	output := ConvertGeminiRequestToAntigravity("gemini-3-pro-preview", inputJSON, false)
	root := mustParseAntigravityGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "request.contents.0.parts.0.thoughtSignature"); got != "skip_thought_signature_validator" {
		t.Fatalf("thoughtSignature = %q, want %q", got, "skip_thought_signature_validator")
	}
}

func TestConvertGeminiRequestToAntigravity_ParallelFunctionCalls(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-pro-preview",
		"contents": [
			{
				"role": "model",
				"parts": [
					{"functionCall": {"name": "tool_one", "args": {"a": "1"}}},
					{"functionCall": {"name": "tool_two", "args": {"b": "2"}}}
				]
			}
		]
	}`)

	output := ConvertGeminiRequestToAntigravity("gemini-3-pro-preview", inputJSON, false)
	root := mustParseAntigravityGeminiObject(t, output)

	parts, ok := jsonutil.Array(root, "request.contents.0.parts")
	if !ok || len(parts) != 2 {
		t.Fatalf("request.contents.0.parts = (%d, %v), want (2, true)", len(parts), ok)
	}
	for i := range parts {
		if got, _ := jsonutil.String(root, fmt.Sprintf("request.contents.0.parts.%d.thoughtSignature", i)); got != "skip_thought_signature_validator" {
			t.Fatalf("part %d thoughtSignature = %q, want %q", i, got, "skip_thought_signature_validator")
		}
	}
}

func TestFixCLIToolResponse_PreservesFunctionResponseParts(t *testing.T) {
	input := `{
		"model": "claude-opus-4-6-thinking",
		"request": {
			"contents": [
				{
					"role": "model",
					"parts": [
						{
							"functionCall": {"name": "screenshot", "args": {}}
						}
					]
				},
				{
					"role": "function",
					"parts": [
						{
							"functionResponse": {
								"id": "tool-001",
								"name": "screenshot",
								"response": {"result": "Screenshot taken"},
								"parts": [
									{"inlineData": {"mimeType": "image/png", "data": "iVBOR"}}
								]
							}
						}
					]
				}
			]
		}
	}`

	result, err := fixCLIToolResponse(input)
	if err != nil {
		t.Fatalf("fixCLIToolResponse failed: %v", err)
	}
	root := mustParseAntigravityGeminiString(t, result)

	funcContent := findAntigravityGeminiContentByRole(t, root, "function")
	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.parts.0.inlineData.mimeType"); got != "image/png" {
		t.Fatalf("inlineData.mimeType = %q, want %q", got, "image/png")
	}
	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.parts.0.inlineData.data"); got != "iVBOR" {
		t.Fatalf("inlineData.data = %q, want %q", got, "iVBOR")
	}
	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.response.result"); got != "Screenshot taken" {
		t.Fatalf("response.result = %q, want %q", got, "Screenshot taken")
	}
}

func TestFixCLIToolResponse_BackfillsEmptyFunctionResponseName(t *testing.T) {
	input := `{
		"model": "gemini-3-pro-preview",
		"request": {
			"contents": [
				{
					"role": "model",
					"parts": [
						{"functionCall": {"name": "Bash", "args": {"cmd": "ls"}}}
					]
				},
				{
					"role": "function",
					"parts": [
						{"functionResponse": {"name": "", "response": {"output": "file1.txt"}}}
					]
				}
			]
		}
	}`

	result, err := fixCLIToolResponse(input)
	if err != nil {
		t.Fatalf("fixCLIToolResponse failed: %v", err)
	}
	root := mustParseAntigravityGeminiString(t, result)
	funcContent := findAntigravityGeminiContentByRole(t, root, "function")

	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.name"); got != "Bash" {
		t.Fatalf("functionResponse.name = %q, want %q", got, "Bash")
	}
}

func TestFixCLIToolResponse_BackfillsMultipleEmptyNames(t *testing.T) {
	input := `{
		"model": "gemini-3-pro-preview",
		"request": {
			"contents": [
				{
					"role": "model",
					"parts": [
						{"functionCall": {"name": "Read", "args": {"path": "/a"}}},
						{"functionCall": {"name": "Grep", "args": {"pattern": "x"}}}
					]
				},
				{
					"role": "function",
					"parts": [
						{"functionResponse": {"name": "", "response": {"result": "content a"}}},
						{"functionResponse": {"name": "", "response": {"result": "match x"}}}
					]
				}
			]
		}
	}`

	result, err := fixCLIToolResponse(input)
	if err != nil {
		t.Fatalf("fixCLIToolResponse failed: %v", err)
	}
	root := mustParseAntigravityGeminiString(t, result)
	funcContent := findAntigravityGeminiContentByRole(t, root, "function")

	parts, ok := jsonutil.Array(funcContent, "parts")
	if !ok || len(parts) != 2 {
		t.Fatalf("parts = (%d, %v), want (2, true)", len(parts), ok)
	}
	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.name"); got != "Read" {
		t.Fatalf("parts.0.functionResponse.name = %q, want %q", got, "Read")
	}
	if got, _ := jsonutil.String(funcContent, "parts.1.functionResponse.name"); got != "Grep" {
		t.Fatalf("parts.1.functionResponse.name = %q, want %q", got, "Grep")
	}
}

func TestFixCLIToolResponse_PreservesExistingName(t *testing.T) {
	input := `{
		"model": "gemini-3-pro-preview",
		"request": {
			"contents": [
				{
					"role": "model",
					"parts": [
						{"functionCall": {"name": "Bash", "args": {}}}
					]
				},
				{
					"role": "function",
					"parts": [
						{"functionResponse": {"name": "Bash", "response": {"result": "ok"}}}
					]
				}
			]
		}
	}`

	result, err := fixCLIToolResponse(input)
	if err != nil {
		t.Fatalf("fixCLIToolResponse failed: %v", err)
	}
	root := mustParseAntigravityGeminiString(t, result)
	funcContent := findAntigravityGeminiContentByRole(t, root, "function")

	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.name"); got != "Bash" {
		t.Fatalf("functionResponse.name = %q, want %q", got, "Bash")
	}
}

func TestFixCLIToolResponse_MoreResponsesThanCalls(t *testing.T) {
	input := `{
		"model": "gemini-3-pro-preview",
		"request": {
			"contents": [
				{
					"role": "model",
					"parts": [
						{"functionCall": {"name": "Bash", "args": {}}}
					]
				},
				{
					"role": "function",
					"parts": [
						{"functionResponse": {"name": "", "response": {"result": "ok"}}},
						{"functionResponse": {"name": "", "response": {"result": "extra"}}}
					]
				}
			]
		}
	}`

	result, err := fixCLIToolResponse(input)
	if err != nil {
		t.Fatalf("fixCLIToolResponse failed: %v", err)
	}
	root := mustParseAntigravityGeminiString(t, result)
	funcContent := findAntigravityGeminiContentByRole(t, root, "function")

	if got, _ := jsonutil.String(funcContent, "parts.0.functionResponse.name"); got != "Bash" {
		t.Fatalf("functionResponse.name = %q, want %q", got, "Bash")
	}
}

func TestFixCLIToolResponse_MultipleGroupsFIFO(t *testing.T) {
	input := `{
		"model": "gemini-3-pro-preview",
		"request": {
			"contents": [
				{
					"role": "model",
					"parts": [
						{"functionCall": {"name": "Read", "args": {}}}
					]
				},
				{
					"role": "function",
					"parts": [
						{"functionResponse": {"name": "", "response": {"result": "file content"}}}
					]
				},
				{
					"role": "model",
					"parts": [
						{"functionCall": {"name": "Grep", "args": {}}}
					]
				},
				{
					"role": "function",
					"parts": [
						{"functionResponse": {"name": "", "response": {"result": "match"}}}
					]
				}
			]
		}
	}`

	result, err := fixCLIToolResponse(input)
	if err != nil {
		t.Fatalf("fixCLIToolResponse failed: %v", err)
	}
	root := mustParseAntigravityGeminiString(t, result)
	funcContents := findAntigravityGeminiContentsByRole(t, root, "function")

	if len(funcContents) != 2 {
		t.Fatalf("function content count = %d, want %d", len(funcContents), 2)
	}
	if got, _ := jsonutil.String(funcContents[0], "parts.0.functionResponse.name"); got != "Read" {
		t.Fatalf("first group name = %q, want %q", got, "Read")
	}
	if got, _ := jsonutil.String(funcContents[1], "parts.0.functionResponse.name"); got != "Grep" {
		t.Fatalf("second group name = %q, want %q", got, "Grep")
	}
}

func mustParseAntigravityGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}

func mustParseAntigravityGeminiString(t *testing.T, payload string) map[string]any {
	t.Helper()
	return mustParseAntigravityGeminiObject(t, []byte(payload))
}

func findAntigravityGeminiContentByRole(t *testing.T, root map[string]any, role string) map[string]any {
	t.Helper()

	contents, ok := jsonutil.Array(root, "request.contents")
	if !ok {
		t.Fatalf("request.contents missing in output: %#v", root)
	}
	for _, contentValue := range contents {
		content, ok := contentValue.(map[string]any)
		if !ok {
			continue
		}
		if gotRole, _ := jsonutil.String(content, "role"); gotRole == role {
			return content
		}
	}
	t.Fatalf("content with role %q not found in output: %#v", role, contents)
	return nil
}

func findAntigravityGeminiContentsByRole(t *testing.T, root map[string]any, role string) []map[string]any {
	t.Helper()

	contents, ok := jsonutil.Array(root, "request.contents")
	if !ok {
		t.Fatalf("request.contents missing in output: %#v", root)
	}
	matches := make([]map[string]any, 0)
	for _, contentValue := range contents {
		content, ok := contentValue.(map[string]any)
		if !ok {
			continue
		}
		if gotRole, _ := jsonutil.String(content, "role"); gotRole == role {
			matches = append(matches, content)
		}
	}
	return matches
}
