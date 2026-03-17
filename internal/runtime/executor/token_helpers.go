package executor

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/tiktoken-go/tokenizer"
)

// tokenizerForModel returns a tokenizer codec suitable for an OpenAI-style model id.
func tokenizerForModel(model string) (tokenizer.Codec, error) {
	sanitized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case sanitized == "":
		return tokenizer.Get(tokenizer.Cl100kBase)
	case strings.HasPrefix(sanitized, "gpt-5"):
		return tokenizer.ForModel(tokenizer.GPT5)
	case strings.HasPrefix(sanitized, "gpt-5.1"):
		return tokenizer.ForModel(tokenizer.GPT5)
	case strings.HasPrefix(sanitized, "gpt-4.1"):
		return tokenizer.ForModel(tokenizer.GPT41)
	case strings.HasPrefix(sanitized, "gpt-4o"):
		return tokenizer.ForModel(tokenizer.GPT4o)
	case strings.HasPrefix(sanitized, "gpt-4"):
		return tokenizer.ForModel(tokenizer.GPT4)
	case strings.HasPrefix(sanitized, "gpt-3.5"), strings.HasPrefix(sanitized, "gpt-3"):
		return tokenizer.ForModel(tokenizer.GPT35Turbo)
	case strings.HasPrefix(sanitized, "o1"):
		return tokenizer.ForModel(tokenizer.O1)
	case strings.HasPrefix(sanitized, "o3"):
		return tokenizer.ForModel(tokenizer.O3)
	case strings.HasPrefix(sanitized, "o4"):
		return tokenizer.ForModel(tokenizer.O4Mini)
	default:
		return tokenizer.Get(tokenizer.O200kBase)
	}
}

// countOpenAIChatTokens approximates prompt tokens for OpenAI chat completions payloads.
func countOpenAIChatTokens(enc tokenizer.Codec, payload []byte) (int64, error) {
	if enc == nil {
		return 0, fmt.Errorf("encoder is nil")
	}
	if len(payload) == 0 {
		return 0, nil
	}

	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return 0, nil
	}
	segments := make([]string, 0, 32)

	collectOpenAIMessages(root["messages"], &segments)
	collectOpenAITools(root["tools"], &segments)
	collectOpenAIFunctions(root["functions"], &segments)
	collectOpenAIToolChoice(root["tool_choice"], &segments)
	collectOpenAIResponseFormat(root["response_format"], &segments)
	addIfNotEmpty(&segments, jsonTextValue(root["input"]))
	addIfNotEmpty(&segments, jsonTextValue(root["prompt"]))

	joined := strings.TrimSpace(strings.Join(segments, "\n"))
	if joined == "" {
		return 0, nil
	}

	count, err := enc.Count(joined)
	if err != nil {
		return 0, err
	}
	return int64(count), nil
}

// buildOpenAIUsageJSON returns a minimal usage structure understood by downstream translators.
func buildOpenAIUsageJSON(count int64) []byte {
	return []byte(fmt.Sprintf(`{"usage":{"prompt_tokens":%d,"completion_tokens":0,"total_tokens":%d}}`, count, count))
}

func collectOpenAIMessages(messages any, segments *[]string) {
	messageArray, ok := messages.([]any)
	if !ok {
		return
	}
	for _, messageValue := range messageArray {
		message, ok := messageValue.(map[string]any)
		if !ok {
			continue
		}
		addIfNotEmpty(segments, jsonTextValue(message["role"]))
		addIfNotEmpty(segments, jsonTextValue(message["name"]))
		collectOpenAIContent(message["content"], segments)
		collectOpenAIToolCalls(message["tool_calls"], segments)
		collectOpenAIFunctionCall(message["function_call"], segments)
	}
}

func collectOpenAIContent(content any, segments *[]string) {
	if content == nil {
		return
	}
	if contentString, ok := content.(string); ok {
		addIfNotEmpty(segments, contentString)
		return
	}
	if contentArray, ok := content.([]any); ok {
		for _, partValue := range contentArray {
			part, ok := partValue.(map[string]any)
			if !ok {
				collectOpenAIContent(partValue, segments)
				continue
			}
			partType := jsonTextValue(part["type"])
			switch partType {
			case "text", "input_text", "output_text":
				addIfNotEmpty(segments, jsonTextValue(part["text"]))
			case "image_url":
				if imageURL, ok := part["image_url"].(map[string]any); ok {
					addIfNotEmpty(segments, jsonTextValue(imageURL["url"]))
				}
			case "input_audio", "output_audio", "audio":
				addIfNotEmpty(segments, jsonTextValue(part["id"]))
			case "tool_result":
				addIfNotEmpty(segments, jsonTextValue(part["name"]))
				collectOpenAIContent(part["content"], segments)
			default:
				addIfNotEmpty(segments, jsonTextValue(part))
			}
		}
		return
	}
	addIfNotEmpty(segments, jsonTextValue(content))
}

func collectOpenAIToolCalls(calls any, segments *[]string) {
	callArray, ok := calls.([]any)
	if !ok {
		return
	}
	for _, callValue := range callArray {
		call, ok := callValue.(map[string]any)
		if !ok {
			continue
		}
		addIfNotEmpty(segments, jsonTextValue(call["id"]))
		addIfNotEmpty(segments, jsonTextValue(call["type"]))
		if function, ok := call["function"].(map[string]any); ok {
			addIfNotEmpty(segments, jsonTextValue(function["name"]))
			addIfNotEmpty(segments, jsonTextValue(function["description"]))
			addIfNotEmpty(segments, jsonTextValue(function["arguments"]))
			if params, ok := function["parameters"]; ok {
				addIfNotEmpty(segments, jsonTextValue(params))
			}
		}
	}
}

func collectOpenAIFunctionCall(call any, segments *[]string) {
	functionCall, ok := call.(map[string]any)
	if !ok {
		return
	}
	addIfNotEmpty(segments, jsonTextValue(functionCall["name"]))
	addIfNotEmpty(segments, jsonTextValue(functionCall["arguments"]))
}

func collectOpenAITools(tools any, segments *[]string) {
	if tools == nil {
		return
	}
	if toolArray, ok := tools.([]any); ok {
		for _, toolValue := range toolArray {
			appendToolPayload(toolValue, segments)
		}
		return
	}
	appendToolPayload(tools, segments)
}

func collectOpenAIFunctions(functions any, segments *[]string) {
	functionArray, ok := functions.([]any)
	if !ok {
		return
	}
	for _, functionValue := range functionArray {
		function, ok := functionValue.(map[string]any)
		if !ok {
			continue
		}
		addIfNotEmpty(segments, jsonTextValue(function["name"]))
		addIfNotEmpty(segments, jsonTextValue(function["description"]))
		if params, ok := function["parameters"]; ok {
			addIfNotEmpty(segments, jsonTextValue(params))
		}
	}
}

func collectOpenAIToolChoice(choice any, segments *[]string) {
	if choice == nil {
		return
	}
	if choiceString, ok := choice.(string); ok {
		addIfNotEmpty(segments, choiceString)
		return
	}
	addIfNotEmpty(segments, jsonTextValue(choice))
}

func collectOpenAIResponseFormat(format any, segments *[]string) {
	formatObject, ok := format.(map[string]any)
	if !ok {
		return
	}
	addIfNotEmpty(segments, jsonTextValue(formatObject["type"]))
	addIfNotEmpty(segments, jsonTextValue(formatObject["name"]))
	if schema, ok := formatObject["json_schema"]; ok {
		addIfNotEmpty(segments, jsonTextValue(schema))
	}
	if schema, ok := formatObject["schema"]; ok {
		addIfNotEmpty(segments, jsonTextValue(schema))
	}
}

func appendToolPayload(tool any, segments *[]string) {
	toolObject, ok := tool.(map[string]any)
	if !ok {
		return
	}
	addIfNotEmpty(segments, jsonTextValue(toolObject["type"]))
	addIfNotEmpty(segments, jsonTextValue(toolObject["name"]))
	addIfNotEmpty(segments, jsonTextValue(toolObject["description"]))
	if function, ok := toolObject["function"].(map[string]any); ok {
		addIfNotEmpty(segments, jsonTextValue(function["name"]))
		addIfNotEmpty(segments, jsonTextValue(function["description"]))
		if params, ok := function["parameters"]; ok {
			addIfNotEmpty(segments, jsonTextValue(params))
		}
	}
}

func jsonTextValue(value any) string {
	if value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	case json.Number:
		return typed.String()
	case bool:
		if typed {
			return "true"
		}
		return "false"
	default:
		out, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return fmt.Sprint(typed)
		}
		return string(out)
	}
}

func addIfNotEmpty(segments *[]string, value string) {
	if segments == nil {
		return
	}
	if trimmed := strings.TrimSpace(value); trimmed != "" {
		*segments = append(*segments, trimmed)
	}
}
