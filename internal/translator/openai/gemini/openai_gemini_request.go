// Package gemini provides request translation functionality for Gemini to
// OpenAI API using standard JSON trees.
package gemini

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// ConvertGeminiRequestToOpenAI parses and transforms a Gemini API request into
// OpenAI Chat Completions API format.
func ConvertGeminiRequestToOpenAI(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"model":    modelName,
		"messages": []any{},
		"stream":   stream,
	}

	if generationConfig, ok := jsonutil.Object(root, "generationConfig"); ok {
		if value, ok := jsonutil.Get(generationConfig, "temperature"); ok {
			outRoot["temperature"] = value
		}
		if value, ok := jsonutil.Get(generationConfig, "maxOutputTokens"); ok {
			outRoot["max_tokens"] = value
		}
		if value, ok := jsonutil.Get(generationConfig, "topP"); ok {
			outRoot["top_p"] = value
		}
		if value, ok := jsonutil.Get(generationConfig, "topK"); ok {
			outRoot["top_k"] = value
		}
		if stopSequences, ok := jsonutil.Array(generationConfig, "stopSequences"); ok {
			stops := make([]string, 0, len(stopSequences))
			for _, stopValue := range stopSequences {
				if stop, ok := stopValue.(string); ok {
					stops = append(stops, stop)
				}
			}
			if len(stops) > 0 {
				outRoot["stop"] = stops
			}
		}
		if value, ok := jsonutil.Get(generationConfig, "candidateCount"); ok {
			outRoot["n"] = value
		}

		if thinkingConfig, ok := jsonutil.Object(generationConfig, "thinkingConfig"); ok {
			geminiThinkingToOpenAIReasoning(outRoot, thinkingConfig)
		} else if thinkingConfig, ok := jsonutil.Object(generationConfig, "thinking_config"); ok {
			geminiThinkingToOpenAIReasoning(outRoot, thinkingConfig)
		}
	}

	messages := make([]any, 0)
	if systemInstruction, ok := jsonutil.Object(root, "systemInstruction"); ok {
		if systemMessage, ok := geminiSystemInstructionToOpenAIMessage(systemInstruction); ok {
			messages = append(messages, systemMessage)
		}
	} else if systemInstruction, ok := jsonutil.Object(root, "system_instruction"); ok {
		if systemMessage, ok := geminiSystemInstructionToOpenAIMessage(systemInstruction); ok {
			messages = append(messages, systemMessage)
		}
	}

	pendingToolIDs := make([]string, 0)
	if contents, ok := jsonutil.Array(root, "contents"); ok {
		for _, contentValue := range contents {
			content, ok := contentValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(content, "role")
			if role == "model" {
				role = "assistant"
			}

			contentParts := make([]any, 0)
			toolCalls := make([]any, 0)
			toolMessages := make([]any, 0)
			var textBuilder strings.Builder
			onlyTextContent := true

			if parts, ok := jsonutil.Array(content, "parts"); ok {
				for _, partValue := range parts {
					part, ok := partValue.(map[string]any)
					if !ok {
						continue
					}

					if text, ok := jsonutil.String(part, "text"); ok {
						textBuilder.WriteString(text)
						contentParts = append(contentParts, map[string]any{
							"type": "text",
							"text": text,
						})
						continue
					}

					if imagePart, ok := geminiInlineDataToOpenAIImagePart(part); ok {
						onlyTextContent = false
						contentParts = append(contentParts, imagePart)
						continue
					}

					if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
						toolCallID, _ := jsonutil.String(functionCall, "id")
						if toolCallID == "" {
							toolCallID = geminiOpenAIToolCallID()
						}
						pendingToolIDs = append(pendingToolIDs, toolCallID)

						arguments := "{}"
						if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
							arguments = string(jsonutil.MarshalOrOriginal(nil, argsValue))
						}

						toolCall := map[string]any{
							"id":   toolCallID,
							"type": "function",
							"function": map[string]any{
								"name":      geminiString(functionCall, "name"),
								"arguments": arguments,
							},
						}
						toolCalls = append(toolCalls, toolCall)
						continue
					}

					if functionResponse, ok := jsonutil.Object(part, "functionResponse"); ok {
						toolCallID := geminiFunctionResponseToolCallID(functionResponse, pendingToolIDs)
						if len(pendingToolIDs) > 0 {
							pendingToolIDs = pendingToolIDs[:len(pendingToolIDs)-1]
						}

						toolMessages = append(toolMessages, map[string]any{
							"role":         "tool",
							"tool_call_id": toolCallID,
							"content":      geminiFunctionResponseContent(functionResponse),
						})
						continue
					}
				}
			}

			message := map[string]any{
				"role": role,
			}
			hasMessageContent := false
			if len(contentParts) > 0 {
				hasMessageContent = true
				if onlyTextContent && len(toolCalls) == 0 {
					message["content"] = textBuilder.String()
				} else {
					message["content"] = contentParts
				}
			}
			if len(toolCalls) > 0 {
				hasMessageContent = true
				message["tool_calls"] = toolCalls
			}

			if hasMessageContent {
				messages = append(messages, message)
			}
			messages = append(messages, toolMessages...)
		}
	}

	outRoot["messages"] = messages

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		openAITools := make([]any, 0)
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			functionDeclarations, ok := jsonutil.Array(tool, "functionDeclarations")
			if !ok {
				continue
			}
			for _, declarationValue := range functionDeclarations {
				declaration, ok := declarationValue.(map[string]any)
				if !ok {
					continue
				}
				function := map[string]any{}
				if name, ok := jsonutil.String(declaration, "name"); ok {
					function["name"] = name
				}
				if description, ok := jsonutil.String(declaration, "description"); ok {
					function["description"] = description
				}
				if parameters, ok := jsonutil.Get(declaration, "parameters"); ok {
					function["parameters"] = parameters
				} else if parameters, ok := jsonutil.Get(declaration, "parametersJsonSchema"); ok {
					function["parameters"] = parameters
				}

				openAITools = append(openAITools, map[string]any{
					"type":     "function",
					"function": function,
				})
			}
		}
		if len(openAITools) > 0 {
			outRoot["tools"] = openAITools
		}
	}

	if toolConfig, ok := jsonutil.Object(root, "toolConfig"); ok {
		applyGeminiToolChoice(outRoot, toolConfig)
	} else if toolConfig, ok := jsonutil.Object(root, "tool_config"); ok {
		applyGeminiToolChoice(outRoot, toolConfig)
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func geminiThinkingToOpenAIReasoning(outRoot map[string]any, thinkingConfig map[string]any) {
	if level, ok := jsonutil.String(thinkingConfig, "thinkingLevel"); ok {
		level = strings.ToLower(strings.TrimSpace(level))
		if level != "" {
			outRoot["reasoning_effort"] = level
			return
		}
	}
	if level, ok := jsonutil.String(thinkingConfig, "thinking_level"); ok {
		level = strings.ToLower(strings.TrimSpace(level))
		if level != "" {
			outRoot["reasoning_effort"] = level
			return
		}
	}

	if budget, ok := jsonutil.Int64(thinkingConfig, "thinkingBudget"); ok {
		if effort, ok := thinking.ConvertBudgetToLevel(int(budget)); ok {
			outRoot["reasoning_effort"] = effort
		}
		return
	}
	if budget, ok := jsonutil.Int64(thinkingConfig, "thinking_budget"); ok {
		if effort, ok := thinking.ConvertBudgetToLevel(int(budget)); ok {
			outRoot["reasoning_effort"] = effort
		}
	}
}

func geminiSystemInstructionToOpenAIMessage(systemInstruction map[string]any) (map[string]any, bool) {
	parts, ok := jsonutil.Array(systemInstruction, "parts")
	if !ok {
		return nil, false
	}

	content := make([]any, 0)
	for _, partValue := range parts {
		part, ok := partValue.(map[string]any)
		if !ok {
			continue
		}
		if text, ok := jsonutil.String(part, "text"); ok {
			content = append(content, map[string]any{
				"type": "text",
				"text": text,
			})
			continue
		}
		if imagePart, ok := geminiInlineDataToOpenAIImagePart(part); ok {
			content = append(content, imagePart)
		}
	}
	if len(content) == 0 {
		return nil, false
	}
	return map[string]any{
		"role":    "system",
		"content": content,
	}, true
}

func geminiInlineDataToOpenAIImagePart(part map[string]any) (map[string]any, bool) {
	inlineData, ok := jsonutil.Object(part, "inlineData")
	if !ok {
		inlineData, ok = jsonutil.Object(part, "inline_data")
		if !ok {
			return nil, false
		}
	}

	mimeType, _ := jsonutil.String(inlineData, "mimeType")
	if mimeType == "" {
		mimeType, _ = jsonutil.String(inlineData, "mime_type")
	}
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	data, _ := jsonutil.String(inlineData, "data")
	if data == "" {
		return nil, false
	}

	return map[string]any{
		"type": "image_url",
		"image_url": map[string]any{
			"url": fmt.Sprintf("data:%s;base64,%s", mimeType, data),
		},
	}, true
}

func geminiOpenAIToolCallID() string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	var builder strings.Builder
	for index := 0; index < 24; index++ {
		number, _ := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
		builder.WriteByte(letters[number.Int64()])
	}
	return "call_" + builder.String()
}

func geminiString(root map[string]any, path string) string {
	value, _ := jsonutil.String(root, path)
	return value
}

func geminiFunctionResponseToolCallID(functionResponse map[string]any, pendingToolIDs []string) string {
	if toolCallID, ok := jsonutil.String(functionResponse, "id"); ok && toolCallID != "" {
		return toolCallID
	}
	if len(pendingToolIDs) > 0 {
		return pendingToolIDs[len(pendingToolIDs)-1]
	}
	return geminiOpenAIToolCallID()
}

func geminiFunctionResponseContent(functionResponse map[string]any) string {
	if contentValue, ok := jsonutil.Get(functionResponse, "response.content"); ok {
		return geminiJSONString(contentValue)
	}
	if responseValue, ok := jsonutil.Get(functionResponse, "response"); ok {
		return geminiJSONString(responseValue)
	}
	return ""
}

func geminiJSONString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	default:
		bytes, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return ""
		}
		return string(bytes)
	}
}

func applyGeminiToolChoice(outRoot map[string]any, toolConfig map[string]any) {
	functionCallingConfig, ok := jsonutil.Object(toolConfig, "functionCallingConfig")
	if !ok {
		functionCallingConfig, ok = jsonutil.Object(toolConfig, "function_calling_config")
		if !ok {
			return
		}
	}

	mode, _ := jsonutil.String(functionCallingConfig, "mode")
	switch mode {
	case "NONE":
		outRoot["tool_choice"] = "none"
	case "AUTO":
		outRoot["tool_choice"] = "auto"
	case "ANY":
		if allowed, ok := jsonutil.Array(functionCallingConfig, "allowedFunctionNames"); ok && len(allowed) == 1 {
			if functionName, ok := allowed[0].(string); ok && functionName != "" {
				outRoot["tool_choice"] = map[string]any{
					"type": "function",
					"function": map[string]any{
						"name": functionName,
					},
				}
				return
			}
		}
		outRoot["tool_choice"] = "required"
	}
}
