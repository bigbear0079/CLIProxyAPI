package responses

import (
	"encoding/json"
	"testing"
)

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletionsPreservesToolChoiceObject(t *testing.T) {
	input := []byte(`{
		"instructions":"be helpful",
		"input":[{"role":"user","content":"hello"}],
		"tool_choice":{"type":"function","function":{"name":"lookup"}},
		"reasoning":{"effort":"high"}
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-test", input, true)

	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}

	toolChoice, ok := root["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %#v, want map[string]any", root["tool_choice"])
	}
	if toolChoice["type"] != "function" {
		t.Fatalf("tool_choice.type = %#v, want %q", toolChoice["type"], "function")
	}
	function, ok := toolChoice["function"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice.function = %#v, want map[string]any", toolChoice["function"])
	}
	if function["name"] != "lookup" {
		t.Fatalf("tool_choice.function.name = %#v, want %q", function["name"], "lookup")
	}
	if root["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %#v, want %q", root["reasoning_effort"], "high")
	}
}

func TestConvertOpenAIResponsesRequestToOpenAIChatCompletionsBuildsMessagesAndTools(t *testing.T) {
	input := []byte(`{
		"input":[
			{"role":"user","content":[{"type":"input_text","text":"hello"},{"type":"input_image","image_url":"https://example.com/a.png"}]},
			{"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{\"q\":\"hi\"}"},
			{"type":"function_call_output","call_id":"call-1","output":"done"}
		],
		"tools":[
			{"type":"web_search"},
			{"type":"function","name":"lookup","description":"desc","parameters":{"type":"object"}}
		]
	}`)

	out := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-test", input, false)

	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}

	if root["model"] != "gpt-test" {
		t.Fatalf("model = %#v, want %q", root["model"], "gpt-test")
	}
	if root["stream"] != false {
		t.Fatalf("stream = %#v, want false", root["stream"])
	}
	messages, ok := root["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("messages = %#v, want 3-item []any", root["messages"])
	}
	userMessage, ok := messages[0].(map[string]any)
	if !ok {
		t.Fatalf("messages[0] = %#v, want map[string]any", messages[0])
	}
	if userMessage["role"] != "user" {
		t.Fatalf("messages[0].role = %#v, want %q", userMessage["role"], "user")
	}
	content, ok := userMessage["content"].([]any)
	if !ok || len(content) != 2 {
		t.Fatalf("messages[0].content = %#v, want 2-item []any", userMessage["content"])
	}
	assistantMessage, ok := messages[1].(map[string]any)
	if !ok {
		t.Fatalf("messages[1] = %#v, want map[string]any", messages[1])
	}
	toolCalls, ok := assistantMessage["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("messages[1].tool_calls = %#v, want 1-item []any", assistantMessage["tool_calls"])
	}
	tools, ok := root["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v, want 1-item []any", root["tools"])
	}
}
