package gemini

import (
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertClaudeResponseToGemini_StreamAssemblesToolUseAndUsage(t *testing.T) {
	ctx := context.Background()
	var param any

	out := ConvertClaudeResponseToGemini(ctx, "gemini-2.5-pro", nil, nil, []byte(`data: {"type":"message_start","message":{"id":"msg_1","model":"gemini-2.5-pro"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("message_start chunk count = %d, want %d", len(out), 0)
	}

	out = ConvertClaudeResponseToGemini(ctx, "gemini-2.5-pro", nil, nil, []byte(`data: {"type":"content_block_start","index":3,"content_block":{"type":"tool_use","name":"lookup"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("content_block_start chunk count = %d, want %d", len(out), 0)
	}

	out = ConvertClaudeResponseToGemini(ctx, "gemini-2.5-pro", nil, nil, []byte(`data: {"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"hi\"}"}}`), &param)
	if len(out) != 0 {
		t.Fatalf("input_json_delta chunk count = %d, want %d", len(out), 0)
	}

	out = ConvertClaudeResponseToGemini(ctx, "gemini-2.5-pro", nil, nil, []byte(`data: {"type":"content_block_stop","index":3}`), &param)
	if len(out) != 1 {
		t.Fatalf("content_block_stop chunk count = %d, want %d", len(out), 1)
	}
	root := mustParseClaudeGeminiResponseObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(root, "modelVersion"); got != "gemini-2.5-pro" {
		t.Fatalf("modelVersion = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(root, "responseId"); got != "msg_1" {
		t.Fatalf("responseId = %q, want %q", got, "msg_1")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.functionCall.name"); got != "lookup" {
		t.Fatalf("functionCall.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.functionCall.args.q"); got != "hi" {
		t.Fatalf("functionCall.args.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "candidates.0.finishReason"); got != "STOP" {
		t.Fatalf("finishReason = %q, want %q", got, "STOP")
	}

	out = ConvertClaudeResponseToGemini(ctx, "gemini-2.5-pro", nil, nil, []byte(`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"input_tokens":9,"output_tokens":4,"cache_creation_input_tokens":2,"cache_read_input_tokens":1,"thinking_tokens":3}}`), &param)
	if len(out) != 1 {
		t.Fatalf("message_delta chunk count = %d, want %d", len(out), 1)
	}
	usageRoot := mustParseClaudeGeminiResponseObject(t, []byte(out[0]))
	if got, _ := jsonutil.String(usageRoot, "candidates.0.finishReason"); got != "STOP" {
		t.Fatalf("finishReason = %q, want %q", got, "STOP")
	}
	if got, _ := jsonutil.String(usageRoot, "usageMetadata.promptTokenCount"); got != "9" {
		t.Fatalf("usageMetadata.promptTokenCount = %q, want %q", got, "9")
	}
	if got, _ := jsonutil.String(usageRoot, "usageMetadata.cachedContentTokenCount"); got != "3" {
		t.Fatalf("usageMetadata.cachedContentTokenCount = %q, want %q", got, "3")
	}
	if got, _ := jsonutil.String(usageRoot, "usageMetadata.thoughtsTokenCount"); got != "3" {
		t.Fatalf("usageMetadata.thoughtsTokenCount = %q, want %q", got, "3")
	}
}

func TestConvertClaudeResponseToGeminiNonStreamConsolidatesParts(t *testing.T) {
	output := ConvertClaudeResponseToGeminiNonStream(context.Background(), "gemini-2.5-pro", nil, nil, []byte(
		"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"model\":\"gemini-2.5-pro\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan-1\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"plan-2\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n"+
			"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"name\":\"lookup\"}}\n"+
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"hi\\\"}\"}}\n"+
			"data: {\"type\":\"content_block_stop\",\"index\":0}\n"+
			"data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":7,\"output_tokens\":5,\"cache_creation_input_tokens\":2,\"cache_read_input_tokens\":1}}\n",
	), nil)

	root := mustParseClaudeGeminiResponseObject(t, []byte(output))
	if got, _ := jsonutil.String(root, "modelVersion"); got != "gemini-2.5-pro" {
		t.Fatalf("modelVersion = %q, want %q", got, "gemini-2.5-pro")
	}
	if got, _ := jsonutil.String(root, "responseId"); got != "msg_2" {
		t.Fatalf("responseId = %q, want %q", got, "msg_2")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.text"); got != "plan-1plan-2" {
		t.Fatalf("parts.0.text = %q, want %q", got, "plan-1plan-2")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.0.thought"); got != "true" {
		t.Fatalf("parts.0.thought = %q, want %q", got, "true")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.1.text"); got != "hello world" {
		t.Fatalf("parts.1.text = %q, want %q", got, "hello world")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.2.functionCall.name"); got != "lookup" {
		t.Fatalf("functionCall.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "candidates.0.content.parts.2.functionCall.args.q"); got != "hi" {
		t.Fatalf("functionCall.args.q = %q, want %q", got, "hi")
	}
	if got, _ := jsonutil.String(root, "usageMetadata.cachedContentTokenCount"); got != "3" {
		t.Fatalf("usageMetadata.cachedContentTokenCount = %q, want %q", got, "3")
	}
}

func mustParseClaudeGeminiResponseObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
