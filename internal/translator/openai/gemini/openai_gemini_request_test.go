package gemini

import (
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func TestConvertGeminiRequestToOpenAIBuildsSystemAndGenerationConfig(t *testing.T) {
	input := []byte(`{
		"system_instruction":{
			"parts":[
				{"text":"sys"}
			]
		},
		"generationConfig":{
			"temperature":0.3,
			"maxOutputTokens":123,
			"topP":0.8,
			"topK":10,
			"stopSequences":["a","b"],
			"candidateCount":2,
			"thinkingConfig":{"thinkingLevel":"high"}
		},
		"contents":[
			{"role":"user","parts":[{"text":"hello"}]}
		]
	}`)

	output := ConvertGeminiRequestToOpenAI("gpt-test", input, true)
	root := mustParseOpenAIGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "model"); got != "gpt-test" {
		t.Fatalf("model = %q, want %q", got, "gpt-test")
	}
	if got, ok := jsonutil.Bool(root, "stream"); !ok || !got {
		t.Fatalf("stream = (%v, %v), want (true, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "temperature"); got != "0.3" {
		t.Fatalf("temperature = %q, want %q", got, "0.3")
	}
	if got, ok := jsonutil.Int64(root, "max_tokens"); !ok || got != 123 {
		t.Fatalf("max_tokens = (%d, %v), want (123, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "top_p"); got != "0.8" {
		t.Fatalf("top_p = %q, want %q", got, "0.8")
	}
	if got, ok := jsonutil.Int64(root, "top_k"); !ok || got != 10 {
		t.Fatalf("top_k = (%d, %v), want (10, true)", got, ok)
	}
	if got, ok := jsonutil.Int64(root, "n"); !ok || got != 2 {
		t.Fatalf("n = (%d, %v), want (2, true)", got, ok)
	}
	if got, _ := jsonutil.String(root, "reasoning_effort"); got != "high" {
		t.Fatalf("reasoning_effort = %q, want %q", got, "high")
	}
	if got, _ := jsonutil.String(root, "messages.0.role"); got != "system" {
		t.Fatalf("messages.0.role = %q, want %q", got, "system")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.0.text"); got != "sys" {
		t.Fatalf("messages.0.content.0.text = %q, want %q", got, "sys")
	}
	if got, _ := jsonutil.String(root, "messages.1.role"); got != "user" {
		t.Fatalf("messages.1.role = %q, want %q", got, "user")
	}
	if got, _ := jsonutil.String(root, "messages.1.content"); got != "hello" {
		t.Fatalf("messages.1.content = %q, want %q", got, "hello")
	}
	stopValues, ok := jsonutil.Array(root, "stop")
	if !ok || len(stopValues) != 2 {
		t.Fatalf("stop = %#v, want 2-item []any", stopValues)
	}
}

func TestConvertGeminiRequestToOpenAIBuildsToolCallsAndToolMessages(t *testing.T) {
	input := []byte(`{
		"contents":[
			{"role":"model","parts":[
				{"functionCall":{"id":"call_1","name":"lookup","args":{"q":"hi"}}}
			]},
			{"role":"function","parts":[
				{"functionResponse":{"id":"call_1","name":"lookup","response":{"result":{"ok":true}}}}
			]}
		]
	}`)

	output := ConvertGeminiRequestToOpenAI("gpt-test", input, false)
	root := mustParseOpenAIGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "messages.0.role"); got != "assistant" {
		t.Fatalf("messages.0.role = %q, want %q", got, "assistant")
	}
	if got, _ := jsonutil.String(root, "messages.0.tool_calls.0.id"); got != "call_1" {
		t.Fatalf("messages.0.tool_calls.0.id = %q, want %q", got, "call_1")
	}
	if got, _ := jsonutil.String(root, "messages.0.tool_calls.0.function.name"); got != "lookup" {
		t.Fatalf("messages.0.tool_calls.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "messages.0.tool_calls.0.function.arguments"); got != `{"q":"hi"}` {
		t.Fatalf("messages.0.tool_calls.0.function.arguments = %q, want %q", got, `{"q":"hi"}`)
	}
	if got, _ := jsonutil.String(root, "messages.1.role"); got != "tool" {
		t.Fatalf("messages.1.role = %q, want %q", got, "tool")
	}
	if got, _ := jsonutil.String(root, "messages.1.tool_call_id"); got != "call_1" {
		t.Fatalf("messages.1.tool_call_id = %q, want %q", got, "call_1")
	}
	if got, _ := jsonutil.String(root, "messages.1.content"); got != `{"result":{"ok":true}}` {
		t.Fatalf("messages.1.content = %q, want %q", got, `{"result":{"ok":true}}`)
	}
}

func TestConvertGeminiRequestToOpenAIBuildsToolsAndSpecificToolChoice(t *testing.T) {
	input := []byte(`{
		"tools":[
			{"functionDeclarations":[
				{"name":"lookup","description":"desc","parametersJsonSchema":{"type":"object"}}
			]}
		],
		"toolConfig":{
			"functionCallingConfig":{
				"mode":"ANY",
				"allowedFunctionNames":["lookup"]
			}
		},
		"contents":[
			{"role":"user","parts":[{"text":"hello"}]}
		]
	}`)

	output := ConvertGeminiRequestToOpenAI("gpt-test", input, false)
	root := mustParseOpenAIGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "tools.0.function.name"); got != "lookup" {
		t.Fatalf("tools.0.function.name = %q, want %q", got, "lookup")
	}
	if got, _ := jsonutil.String(root, "tools.0.function.parameters.type"); got != "object" {
		t.Fatalf("tools.0.function.parameters.type = %q, want %q", got, "object")
	}
	if got, _ := jsonutil.String(root, "tool_choice.type"); got != "function" {
		t.Fatalf("tool_choice.type = %q, want %q", got, "function")
	}
	if got, _ := jsonutil.String(root, "tool_choice.function.name"); got != "lookup" {
		t.Fatalf("tool_choice.function.name = %q, want %q", got, "lookup")
	}
}

func TestConvertGeminiRequestToOpenAIConvertsInlineDataToImageURL(t *testing.T) {
	input := []byte(`{
		"contents":[
			{"role":"user","parts":[
				{"inlineData":{"mimeType":"image/png","data":"aGVsbG8="}}
			]}
		]
	}`)

	output := ConvertGeminiRequestToOpenAI("gpt-test", input, false)
	root := mustParseOpenAIGeminiObject(t, output)

	if got, _ := jsonutil.String(root, "messages.0.content.0.type"); got != "image_url" {
		t.Fatalf("messages.0.content.0.type = %q, want %q", got, "image_url")
	}
	if got, _ := jsonutil.String(root, "messages.0.content.0.image_url.url"); !strings.HasPrefix(got, "data:image/png;base64,") {
		t.Fatalf("messages.0.content.0.image_url.url = %q, want data URL with image/png prefix", got)
	}
}

func mustParseOpenAIGeminiObject(t *testing.T, payload []byte) map[string]any {
	t.Helper()

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		t.Fatalf("ParseObjectBytes returned error: %v\npayload: %s", errParse, string(payload))
	}
	return root
}
