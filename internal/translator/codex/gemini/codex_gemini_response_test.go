package gemini

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertCodexResponseToGeminiStreamAccumulatesFunctionCall(t *testing.T) {
	originalRequest := []byte(`{
		"tools":[
			{"functionDeclarations":[
				{"name":"mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"}
			]}
		]
	}`)
	shortName := buildShortNameMap([]string{"mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"})["mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"]

	var param any
	chunks := ConvertCodexResponseToGemini(context.Background(), "gemini-test", originalRequest, nil, []byte(`data: {"type":"response.created","response":{"id":"resp_1","created_at":1700000000,"model":"gemini-real"}}`), &param)
	if len(chunks) != 1 {
		t.Fatalf("response.created chunk count = %d, want %d", len(chunks), 1)
	}

	createdRoot := mustParseCodexGeminiResponseObject(t, []byte(chunks[0]))
	if got, _ := jsonutil.String(createdRoot, "modelVersion"); got != "gemini-real" {
		t.Fatalf("modelVersion = %q, want %q", got, "gemini-real")
	}
	if got, _ := jsonutil.String(createdRoot, "responseId"); got != "resp_1" {
		t.Fatalf("responseId = %q, want %q", got, "resp_1")
	}

	chunks = ConvertCodexResponseToGemini(context.Background(), "gemini-test", originalRequest, nil, []byte(`data: {"type":"response.output_item.done","item":{"type":"function_call","name":"`+shortName+`","arguments":"{\"q\":\"hi\"}"}}`), &param)
	if len(chunks) != 0 {
		t.Fatalf("function_call output_item.done chunk count = %d, want %d", len(chunks), 0)
	}

	chunks = ConvertCodexResponseToGemini(context.Background(), "gemini-test", originalRequest, nil, []byte(`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":5}}}`), &param)
	if len(chunks) != 2 {
		t.Fatalf("response.completed chunk count = %d, want %d", len(chunks), 2)
	}

	storedRoot := mustParseCodexGeminiResponseObject(t, []byte(chunks[0]))
	if got, _ := jsonutil.String(storedRoot, "candidates.0.content.parts.0.functionCall.name"); got != "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup" {
		t.Fatalf("functionCall.name = %q, want original tool name", got)
	}
	if got, _ := jsonutil.String(storedRoot, "candidates.0.content.parts.0.functionCall.args.q"); got != "hi" {
		t.Fatalf("functionCall.args.q = %q, want %q", got, "hi")
	}

	completedRoot := mustParseCodexGeminiResponseObject(t, []byte(chunks[1]))
	if got, _ := jsonutil.String(completedRoot, "usageMetadata.promptTokenCount"); got != "10" {
		t.Fatalf("usageMetadata.promptTokenCount = %q, want %q", got, "10")
	}
	if got, _ := jsonutil.String(completedRoot, "usageMetadata.totalTokenCount"); got != "15" {
		t.Fatalf("usageMetadata.totalTokenCount = %q, want %q", got, "15")
	}
}

func TestConvertCodexResponseToGeminiNonStreamBuildsParts(t *testing.T) {
	originalRequest := []byte(`{
		"tools":[
			{"functionDeclarations":[
				{"name":"mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"}
			]}
		]
	}`)
	shortName := buildShortNameMap([]string{"mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"})["mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup"]

	output := ConvertCodexResponseToGeminiNonStream(context.Background(), "gemini-test", originalRequest, nil, []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_2",
			"created_at":1700000000,
			"usage":{"input_tokens":7,"output_tokens":3},
			"output":[
				{"type":"reasoning","summary":[{"text":"think"}]},
				{"type":"message","content":[{"type":"output_text","text":"hello"}]},
				{"type":"function_call","name":"`+shortName+`","arguments":"{\"q\":\"hi\"}"}
			]
		}
	}`), nil)

	root := mustParseCodexGeminiResponseObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "responseId"); got != "resp_2" {
		t.Fatalf("responseId = %q, want %q", got, "resp_2")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.text"); got != "think" {
		t.Fatalf("parts.0.text = %q, want %q", got, "think")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.1.text"); got != "hello" {
		t.Fatalf("parts.1.text = %q, want %q", got, "hello")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.2.functionCall.name"); got != "mcp__server__abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz__lookup" {
		t.Fatalf("parts.2.functionCall.name = %q, want original tool name", got)
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.2.functionCall.args.q"); got != "hi" {
		t.Fatalf("parts.2.functionCall.args.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "usageMetadata.totalTokenCount"); got != "10" {
		t.Fatalf("usageMetadata.totalTokenCount = %q, want %q", got, "10")
	}
}

func mustParseCodexGeminiResponseObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
