// Package claude provides request translation functionality for Anthropic to OpenAI API.
// It handles parsing and transforming Anthropic API requests into OpenAI Chat Completions API format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between Anthropic API format and OpenAI API's expected format.
package claude

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// ConvertClaudeRequestToOpenAI parses and transforms an Anthropic API request into OpenAI Chat Completions API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the OpenAI API.
func ConvertClaudeRequestToOpenAI(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"model":    modelName,
		"messages": []any{},
		"stream":   stream,
	}
	messages := make([]any, 0)

	if maxTokens, ok := jsonutil.Get(root, "max_tokens"); ok {
		outRoot["max_tokens"] = maxTokens
	}

	if temp, ok := jsonutil.Get(root, "temperature"); ok {
		outRoot["temperature"] = temp
	} else if topP, ok := jsonutil.Get(root, "top_p"); ok {
		outRoot["top_p"] = topP
	}

	if stopSequences, ok := jsonutil.Array(root, "stop_sequences"); ok {
		stops := make([]string, 0, len(stopSequences))
		for _, value := range stopSequences {
			switch typed := value.(type) {
			case string:
				stops = append(stops, typed)
			}
		}
		if len(stops) == 1 {
			outRoot["stop"] = stops[0]
		} else if len(stops) > 1 {
			outRoot["stop"] = stops
		}
	}

	if thinkingConfig, ok := jsonutil.Object(root, "thinking"); ok {
		if thinkingType, ok := jsonutil.String(thinkingConfig, "type"); ok {
			switch thinkingType {
			case "enabled":
				if budgetTokens, ok := jsonutil.Int64(thinkingConfig, "budget_tokens"); ok {
					budget := int(budgetTokens)
					if effort, ok := thinking.ConvertBudgetToLevel(budget); ok && effort != "" {
						outRoot["reasoning_effort"] = effort
					}
				} else if effort, ok := thinking.ConvertBudgetToLevel(-1); ok && effort != "" {
					outRoot["reasoning_effort"] = effort
				}
			case "adaptive", "auto":
				effort := ""
				if outputConfig, ok := jsonutil.Object(root, "output_config"); ok {
					if value, ok := jsonutil.String(outputConfig, "effort"); ok {
						effort = strings.ToLower(strings.TrimSpace(value))
					}
				}
				if effort != "" {
					outRoot["reasoning_effort"] = effort
				} else {
					outRoot["reasoning_effort"] = string(thinking.LevelXHigh)
				}
			case "disabled":
				if effort, ok := thinking.ConvertBudgetToLevel(0); ok && effort != "" {
					outRoot["reasoning_effort"] = effort
				}
			}
		}
	}

	systemMessage := map[string]any{
		"role":    "system",
		"content": []any{},
	}
	systemContent := make([]any, 0)
	if systemText, ok := jsonutil.String(root, "system"); ok {
		if systemText != "" {
			systemContent = append(systemContent, map[string]any{
				"type": "text",
				"text": systemText,
			})
		}
	} else if systemArray, ok := jsonutil.Array(root, "system"); ok {
		for _, partValue := range systemArray {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}
			if contentItem, ok := convertClaudeContentPart(part); ok {
				systemContent = append(systemContent, contentItem)
			}
		}
	}
	if len(systemContent) > 0 {
		systemMessage["content"] = systemContent
		messages = append(messages, systemMessage)
	}

	if anthropicMessages, ok := jsonutil.Array(root, "messages"); ok {
		for _, messageValue := range anthropicMessages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(message, "role")
			if contentArray, ok := jsonutil.Array(message, "content"); ok {
				contentItems := make([]any, 0)
				reasoningParts := make([]string, 0)
				toolCalls := make([]any, 0)
				toolResults := make([]any, 0)

				for _, partValue := range contentArray {
					part, ok := partValue.(map[string]any)
					if !ok {
						continue
					}
					partType, _ := jsonutil.String(part, "type")

					switch partType {
					case "thinking":
						if role == "assistant" {
							thinkingText := extractThinkingText(part)
							if strings.TrimSpace(thinkingText) != "" {
								reasoningParts = append(reasoningParts, thinkingText)
							}
						}
					case "redacted_thinking":
						// Ignore redacted thinking.
					case "text", "image":
						if contentItem, ok := convertClaudeContentPart(part); ok {
							contentItems = append(contentItems, contentItem)
						}
					case "tool_use":
						if role == "assistant" {
							toolCall := map[string]any{
								"id":   jsonString(part, "id"),
								"type": "function",
								"function": map[string]any{
									"name":      jsonString(part, "name"),
									"arguments": "{}",
								},
							}
							if input, ok := jsonutil.Get(part, "input"); ok {
								toolCall["function"].(map[string]any)["arguments"] = string(jsonutil.MarshalOrOriginal(nil, input))
							}
							toolCalls = append(toolCalls, toolCall)
						}
					case "tool_result":
						contentValue, raw := convertClaudeToolResultContentValue(part["content"])
						toolResult := map[string]any{
							"role":         "tool",
							"tool_call_id": jsonString(part, "tool_use_id"),
						}
						if raw {
							toolResult["content"] = contentValue
						} else {
							toolResult["content"] = contentValue
						}
						toolResults = append(toolResults, toolResult)
					}
				}

				reasoningContent := ""
				if len(reasoningParts) > 0 {
					reasoningContent = strings.Join(reasoningParts, "\n\n")
				}

				for _, toolResult := range toolResults {
					messages = append(messages, toolResult)
				}

				hasContent := len(contentItems) > 0
				hasReasoning := reasoningContent != ""
				hasToolCalls := len(toolCalls) > 0
				hasToolResults := len(toolResults) > 0

				if role == "assistant" {
					if hasContent || hasReasoning || hasToolCalls {
						msg := map[string]any{
							"role": "assistant",
						}
						if hasContent {
							msg["content"] = contentItems
						} else {
							msg["content"] = ""
						}
						if hasReasoning {
							msg["reasoning_content"] = reasoningContent
						}
						if hasToolCalls {
							msg["tool_calls"] = toolCalls
						}
						messages = append(messages, msg)
					}
				} else {
					if hasContent {
						messages = append(messages, map[string]any{
							"role":    role,
							"content": contentItems,
						})
					} else if hasToolResults && !hasContent {
						// Tool results were already emitted.
					}
				}
			} else if contentText, ok := jsonutil.String(message, "content"); ok {
				messages = append(messages, map[string]any{
					"role":    role,
					"content": contentText,
				})
			}
		}
	}

	if len(messages) > 0 {
		outRoot["messages"] = messages
	}

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		outTools := make([]any, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			openAITool := map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        jsonString(tool, "name"),
					"description": jsonString(tool, "description"),
				},
			}
			if inputSchema, ok := jsonutil.Get(tool, "input_schema"); ok {
				openAITool["function"].(map[string]any)["parameters"] = inputSchema
			}
			outTools = append(outTools, openAITool)
		}
		if len(outTools) > 0 {
			outRoot["tools"] = outTools
		}
	}

	if toolChoice, ok := jsonutil.Object(root, "tool_choice"); ok {
		switch jsonString(toolChoice, "type") {
		case "auto":
			outRoot["tool_choice"] = "auto"
		case "any":
			outRoot["tool_choice"] = "required"
		case "tool":
			outRoot["tool_choice"] = map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": jsonString(toolChoice, "name"),
				},
			}
		default:
			outRoot["tool_choice"] = "auto"
		}
	}

	if user, ok := jsonutil.String(root, "user"); ok {
		outRoot["user"] = user
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func jsonString(root map[string]any, path string) string {
	value, _ := jsonutil.String(root, path)
	return value
}

func extractThinkingText(part map[string]any) string {
	if text, ok := jsonutil.String(part, "text"); ok {
		return text
	}

	thinkingValue, ok := jsonutil.Get(part, "thinking")
	if !ok {
		return ""
	}

	switch typed := thinkingValue.(type) {
	case string:
		return typed
	case map[string]any:
		if text, ok := jsonutil.String(typed, "text"); ok {
			return text
		}
		if text, ok := jsonutil.String(typed, "thinking"); ok {
			return text
		}
	}

	return ""
}

func convertClaudeContentPart(part map[string]any) (map[string]any, bool) {
	partType, _ := jsonutil.String(part, "type")

	switch partType {
	case "text":
		text, _ := jsonutil.String(part, "text")
		if strings.TrimSpace(text) == "" {
			return nil, false
		}
		return map[string]any{
			"type": "text",
			"text": text,
		}, true

	case "image":
		imageURL := ""
		if source, ok := jsonutil.Object(part, "source"); ok {
			switch jsonString(source, "type") {
			case "base64":
				mediaType := jsonString(source, "media_type")
				if mediaType == "" {
					mediaType = "application/octet-stream"
				}
				data := jsonString(source, "data")
				if data != "" {
					imageURL = "data:" + mediaType + ";base64," + data
				}
			case "url":
				imageURL = jsonString(source, "url")
			}
		}
		if imageURL == "" {
			imageURL = jsonString(part, "url")
		}
		if imageURL == "" {
			return nil, false
		}
		return map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url": imageURL,
			},
		}, true
	}

	return nil, false
}

func convertClaudeToolResultContentValue(content any) (any, bool) {
	if content == nil {
		return "", false
	}

	switch typed := content.(type) {
	case string:
		return typed, false

	case []any:
		parts := make([]string, 0)
		contentArray := make([]any, 0)
		hasImagePart := false

		for _, itemValue := range typed {
			switch item := itemValue.(type) {
			case string:
				parts = append(parts, item)
				contentArray = append(contentArray, map[string]any{
					"type": "text",
					"text": item,
				})
			case map[string]any:
				itemType, _ := jsonutil.String(item, "type")
				switch itemType {
				case "text":
					text := jsonString(item, "text")
					parts = append(parts, text)
					contentArray = append(contentArray, map[string]any{
						"type": "text",
						"text": text,
					})
				case "image":
					if contentItem, ok := convertClaudeContentPart(item); ok {
						contentArray = append(contentArray, contentItem)
						hasImagePart = true
					} else {
						parts = append(parts, string(jsonutil.MarshalOrOriginal(nil, item)))
					}
				default:
					if text, ok := jsonutil.String(item, "text"); ok {
						parts = append(parts, text)
					} else {
						parts = append(parts, string(jsonutil.MarshalOrOriginal(nil, item)))
					}
				}
			default:
				parts = append(parts, string(jsonutil.MarshalOrOriginal(nil, item)))
			}
		}

		if hasImagePart {
			return contentArray, true
		}

		joined := strings.Join(parts, "\n\n")
		if strings.TrimSpace(joined) != "" {
			return joined, false
		}
		return string(jsonutil.MarshalOrOriginal(nil, typed)), false

	case map[string]any:
		if jsonString(typed, "type") == "image" {
			if contentItem, ok := convertClaudeContentPart(typed); ok {
				return []any{contentItem}, true
			}
		}
		if text, ok := jsonutil.String(typed, "text"); ok {
			return text, false
		}
		return string(jsonutil.MarshalOrOriginal(nil, typed)), false
	}

	return string(jsonutil.MarshalOrOriginal(nil, content)), false
}
