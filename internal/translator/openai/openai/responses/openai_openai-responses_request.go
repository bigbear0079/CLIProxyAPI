package responses

import (
	"encoding/json"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertOpenAIResponsesRequestToOpenAIChatCompletions converts OpenAI responses format to OpenAI chat completions format.
// It transforms the OpenAI responses API format (with instructions and input array) into the standard
// OpenAI chat completions format (with messages array and system content).
//
// The conversion handles:
// 1. Model name and streaming configuration
// 2. Instructions to system message conversion
// 3. Input array to messages array transformation
// 4. Tool definitions and tool choice conversion
// 5. Function calls and function results handling
// 6. Generation parameters mapping (max_tokens, reasoning, etc.)
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data in OpenAI responses format
//   - stream: A boolean indicating if the request is for a streaming response
//
// Returns:
//   - []byte: The transformed request data in OpenAI chat completions format
func ConvertOpenAIResponsesRequestToOpenAIChatCompletions(modelName string, inputRawJSON []byte, stream bool) []byte {
	root, errParse := jsonutil.ParseObjectBytes(inputRawJSON)
	if errParse != nil {
		return marshalOpenAIChatCompletionsRequest(map[string]any{
			"model":    modelName,
			"messages": make([]any, 0),
			"stream":   stream,
		})
	}
	return ConvertOpenAIResponsesRequestObjectToOpenAIChatCompletions(modelName, root, stream)
}

// ConvertOpenAIResponsesRequestObjectToOpenAIChatCompletions converts a pre-parsed
// OpenAI responses request object into the OpenAI chat completions format.
func ConvertOpenAIResponsesRequestObjectToOpenAIChatCompletions(modelName string, root map[string]any, stream bool) []byte {
	out := map[string]any{
		"model":    modelName,
		"messages": make([]any, 0),
		"stream":   stream,
	}

	getString := func(value any) (string, bool) {
		if value == nil {
			return "", false
		}
		switch typed := value.(type) {
		case string:
			return typed, true
		case json.Number:
			return typed.String(), true
		case bool:
			if typed {
				return "true", true
			}
			return "false", true
		default:
			marshaled, errMarshal := json.Marshal(typed)
			if errMarshal != nil {
				return "", false
			}
			return string(marshaled), true
		}
	}

	getObject := func(value any) (map[string]any, bool) {
		object, ok := value.(map[string]any)
		return object, ok
	}

	getArray := func(value any) ([]any, bool) {
		array, ok := value.([]any)
		return array, ok
	}

	getBool := func(value any) (bool, bool) {
		boolean, ok := value.(bool)
		return boolean, ok
	}

	getInt64 := func(value any) (int64, bool) {
		switch typed := value.(type) {
		case json.Number:
			intValue, errInt := typed.Int64()
			if errInt != nil {
				return 0, false
			}
			return intValue, true
		default:
			return 0, false
		}
	}

	messages := out["messages"].([]any)

	// Map generation parameters from responses format to chat completions format.
	if maxTokens, ok := getInt64(root["max_output_tokens"]); ok {
		out["max_tokens"] = maxTokens
	}

	if parallelToolCalls, ok := getBool(root["parallel_tool_calls"]); ok {
		out["parallel_tool_calls"] = parallelToolCalls
	}

	// Convert instructions to system message.
	if instructions, ok := getString(root["instructions"]); ok {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": instructions,
		})
	}

	// Convert input array to messages.
	if inputArray, ok := getArray(root["input"]); ok {
		for _, itemValue := range inputArray {
			item, ok := getObject(itemValue)
			if !ok {
				continue
			}

			itemType, _ := getString(item["type"])
			role, _ := getString(item["role"])
			if itemType == "" && role != "" {
				itemType = "message"
			}

			switch itemType {
			case "message", "":
				if role == "developer" {
					role = "user"
				}

				message := map[string]any{
					"role": role,
				}

				if contentArray, ok := getArray(item["content"]); ok {
					parts := make([]any, 0, len(contentArray))
					for _, contentValue := range contentArray {
						contentItem, ok := getObject(contentValue)
						if !ok {
							continue
						}

						contentType, _ := getString(contentItem["type"])
						if contentType == "" {
							contentType = "input_text"
						}

						switch contentType {
						case "input_text", "output_text":
							text, ok := getString(contentItem["text"])
							if !ok {
								continue
							}
							parts = append(parts, map[string]any{
								"type": "text",
								"text": text,
							})
						case "input_image":
							imageURL, ok := getString(contentItem["image_url"])
							if !ok {
								continue
							}
							parts = append(parts, map[string]any{
								"type": "image_url",
								"image_url": map[string]any{
									"url": imageURL,
								},
							})
						}
					}
					message["content"] = parts
				} else if content, ok := getString(item["content"]); ok {
					message["content"] = content
				}

				messages = append(messages, message)
			case "function_call":
				function := map[string]any{}
				if name, ok := getString(item["name"]); ok {
					function["name"] = name
				}
				if arguments, ok := getString(item["arguments"]); ok {
					function["arguments"] = arguments
				}

				toolCall := map[string]any{
					"type":     "function",
					"function": function,
				}
				if callID, ok := getString(item["call_id"]); ok {
					toolCall["id"] = callID
				}

				messages = append(messages, map[string]any{
					"role":       "assistant",
					"tool_calls": []any{toolCall},
				})
			case "function_call_output":
				toolMessage := map[string]any{
					"role": "tool",
				}
				if callID, ok := getString(item["call_id"]); ok {
					toolMessage["tool_call_id"] = callID
				}
				if output, ok := getString(item["output"]); ok {
					toolMessage["content"] = output
				}

				messages = append(messages, toolMessage)
			}
		}
	} else if inputText, ok := getString(root["input"]); ok {
		messages = append(messages, map[string]any{
			"role":    "user",
			"content": inputText,
		})
	}

	out["messages"] = messages

	// Convert tools from responses format to chat completions format.
	if tools, ok := getArray(root["tools"]); ok {
		chatCompletionsTools := make([]any, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := getObject(toolValue)
			if !ok {
				continue
			}

			// Built-in tools are intentionally ignored because most upstream chat-completions providers do not support them.
			toolType, _ := getString(tool["type"])
			if toolType != "" && toolType != "function" {
				continue
			}

			function := map[string]any{}
			if name, ok := getString(tool["name"]); ok {
				function["name"] = name
			}
			if description, ok := getString(tool["description"]); ok {
				function["description"] = description
			}
			if parameters, ok := tool["parameters"]; ok {
				function["parameters"] = parameters
			}

			chatCompletionsTools = append(chatCompletionsTools, map[string]any{
				"type":     "function",
				"function": function,
			})
		}

		if len(chatCompletionsTools) > 0 {
			out["tools"] = chatCompletionsTools
		}
	}

	if reasoning, ok := getObject(root["reasoning"]); ok {
		if effort, ok := getString(reasoning["effort"]); ok {
			effort = strings.ToLower(strings.TrimSpace(effort))
			if effort != "" {
				out["reasoning_effort"] = effort
			}
		}
	}

	// Preserve tool_choice shape so object values stay objects instead of turning into JSON strings.
	if toolChoice, ok := root["tool_choice"]; ok {
		out["tool_choice"] = toolChoice
	}

	return marshalOpenAIChatCompletionsRequest(out)
}

func marshalOpenAIChatCompletionsRequest(payload map[string]any) []byte {
	out, errMarshal := json.Marshal(payload)
	if errMarshal != nil {
		return []byte(`{"model":"","messages":[],"stream":false}`)
	}
	return out
}
