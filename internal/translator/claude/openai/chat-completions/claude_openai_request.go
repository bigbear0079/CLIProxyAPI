// Package openai provides request translation functionality for OpenAI to Claude Code API compatibility.
// It handles parsing and transforming OpenAI Chat Completions API requests into Claude Code API format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between OpenAI API format and Claude Code API's expected format.
package chat_completions

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

var (
	user    = ""
	account = ""
	session = ""
)

// ConvertOpenAIRequestToClaude parses and transforms an OpenAI Chat Completions API request into Claude Code API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the Claude Code API.
// The function performs comprehensive transformation including:
// 1. Model name mapping and parameter extraction (max_tokens, temperature, top_p, etc.)
// 2. Message content conversion from OpenAI to Claude Code format
// 3. Tool call and tool result handling with proper ID mapping
// 4. Image data conversion from OpenAI data URLs to Claude Code base64 format
// 5. Stop sequence and streaming configuration handling
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the OpenAI API
//   - stream: A boolean indicating if the request is for a streaming response
//
// Returns:
//   - []byte: The transformed request data in Claude Code API format
func ConvertOpenAIRequestToClaude(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	if account == "" {
		u, _ := uuid.NewRandom()
		account = u.String()
	}
	if session == "" {
		u, _ := uuid.NewRandom()
		session = u.String()
	}
	if user == "" {
		sum := sha256.Sum256([]byte(account + session))
		user = hex.EncodeToString(sum[:])
	}
	userID := fmt.Sprintf("user_%s_account_%s_session_%s", user, account, session)

	outRoot := map[string]any{
		"model":      modelName,
		"max_tokens": 32000,
		"messages":   []any{},
		"metadata": map[string]any{
			"user_id": userID,
		},
		"stream": stream,
	}

	if effortValue, ok := jsonutil.String(root, "reasoning_effort"); ok {
		effort := strings.ToLower(strings.TrimSpace(effortValue))
		if effort != "" {
			mi := registry.LookupModelInfo(modelName, "claude")
			supportsAdaptive := mi != nil && mi.Thinking != nil && len(mi.Thinking.Levels) > 0
			supportsMax := supportsAdaptive && thinking.HasLevel(mi.Thinking.Levels, string(thinking.LevelMax))

			if supportsAdaptive {
				switch effort {
				case "none":
					outRoot["thinking"] = map[string]any{"type": "disabled"}
				case "auto":
					outRoot["thinking"] = map[string]any{"type": "adaptive"}
				default:
					if mapped, ok := thinking.MapToClaudeEffort(effort, supportsMax); ok {
						effort = mapped
					}
					outRoot["thinking"] = map[string]any{"type": "adaptive"}
					outRoot["output_config"] = map[string]any{"effort": effort}
				}
			} else {
				budget, ok := thinking.ConvertLevelToBudget(effort)
				if ok {
					switch budget {
					case 0:
						outRoot["thinking"] = map[string]any{"type": "disabled"}
					case -1:
						outRoot["thinking"] = map[string]any{"type": "enabled"}
					default:
						if budget > 0 {
							outRoot["thinking"] = map[string]any{
								"type":          "enabled",
								"budget_tokens": budget,
							}
						}
					}
				}
			}
		}
	}

	genToolCallID := func() string {
		const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		var builder strings.Builder
		for i := 0; i < 24; i++ {
			n, _ := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
			builder.WriteByte(letters[n.Int64()])
		}
		return "toolu_" + builder.String()
	}

	if maxTokens, ok := jsonutil.Get(root, "max_tokens"); ok {
		outRoot["max_tokens"] = maxTokens
	}

	if temp, ok := jsonutil.Get(root, "temperature"); ok {
		outRoot["temperature"] = temp
	} else if topP, ok := jsonutil.Get(root, "top_p"); ok {
		outRoot["top_p"] = topP
	}

	if stopValue, ok := jsonutil.Get(root, "stop"); ok {
		switch typed := stopValue.(type) {
		case []any:
			stopSequences := make([]string, 0, len(typed))
			for _, value := range typed {
				if stop, ok := value.(string); ok {
					stopSequences = append(stopSequences, stop)
				}
			}
			if len(stopSequences) > 0 {
				outRoot["stop_sequences"] = stopSequences
			}
		case string:
			outRoot["stop_sequences"] = []string{typed}
		}
	}

	messages := make([]any, 0)
	systemMessageIndex := -1
	if sourceMessages, ok := jsonutil.Array(root, "messages"); ok {
		for _, messageValue := range sourceMessages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(message, "role")
			switch role {
			case "system":
				if systemMessageIndex == -1 {
					messages = append(messages, map[string]any{
						"role":    "user",
						"content": []any{},
					})
					systemMessageIndex = len(messages) - 1
				}
				systemMessage := messages[systemMessageIndex].(map[string]any)
				systemContent := systemMessage["content"].([]any)
				if contentText, ok := jsonutil.String(message, "content"); ok && contentText != "" {
					systemContent = append(systemContent, map[string]any{
						"type": "text",
						"text": contentText,
					})
				} else if contentArray, ok := jsonutil.Array(message, "content"); ok {
					for _, partValue := range contentArray {
						part, ok := partValue.(map[string]any)
						if !ok {
							continue
						}
						if partType, _ := jsonutil.String(part, "type"); partType == "text" {
							systemContent = append(systemContent, map[string]any{
								"type": "text",
								"text": jsonString(part, "text"),
							})
						}
					}
				}
				systemMessage["content"] = systemContent

			case "user", "assistant":
				msg := map[string]any{
					"role":    role,
					"content": []any{},
				}
				contentParts := make([]any, 0)

				if contentText, ok := jsonutil.String(message, "content"); ok && contentText != "" {
					contentParts = append(contentParts, map[string]any{
						"type": "text",
						"text": contentText,
					})
				} else if contentArray, ok := jsonutil.Array(message, "content"); ok {
					for _, partValue := range contentArray {
						part, ok := partValue.(map[string]any)
						if !ok {
							continue
						}
						if claudePart, ok := convertOpenAIContentPartToClaudePart(part); ok {
							contentParts = append(contentParts, claudePart)
						}
					}
				}

				if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok && role == "assistant" {
					for _, toolCallValue := range toolCalls {
						toolCall, ok := toolCallValue.(map[string]any)
						if !ok {
							continue
						}
						if toolType, _ := jsonutil.String(toolCall, "type"); toolType != "function" {
							continue
						}
						function, ok := jsonutil.Object(toolCall, "function")
						if !ok {
							continue
						}
						toolCallID, _ := jsonutil.String(toolCall, "id")
						if toolCallID == "" {
							toolCallID = genToolCallID()
						}
						input := map[string]any{}
						if arguments, ok := jsonutil.String(function, "arguments"); ok && arguments != "" {
							if parsed, ok := parseObjectString(arguments); ok {
								input = parsed
							}
						}
						contentParts = append(contentParts, map[string]any{
							"type":  "tool_use",
							"id":    toolCallID,
							"name":  jsonString(function, "name"),
							"input": input,
						})
					}
				}

				msg["content"] = contentParts
				messages = append(messages, msg)

			case "tool":
				contentValue, ok := jsonutil.Get(message, "content")
				if !ok {
					contentValue = ""
				}
				toolContent, raw := convertOpenAIToolResultContent(contentValue)
				content := map[string]any{
					"type":        "tool_result",
					"tool_use_id": jsonString(message, "tool_call_id"),
				}
				if raw {
					content["content"] = toolContent
				} else {
					content["content"] = toolContent
				}
				messages = append(messages, map[string]any{
					"role": "user",
					"content": []any{
						content,
					},
				})
			}
		}
	}
	outRoot["messages"] = messages

	if tools, ok := jsonutil.Array(root, "tools"); ok && len(tools) > 0 {
		outTools := make([]any, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			if toolType, _ := jsonutil.String(tool, "type"); toolType != "function" {
				continue
			}
			function, ok := jsonutil.Object(tool, "function")
			if !ok {
				continue
			}
			anthropicTool := map[string]any{
				"name":        jsonString(function, "name"),
				"description": jsonString(function, "description"),
			}
			if parameters, ok := jsonutil.Get(function, "parameters"); ok {
				anthropicTool["input_schema"] = parameters
			} else if parameters, ok := jsonutil.Get(function, "parametersJsonSchema"); ok {
				anthropicTool["input_schema"] = parameters
			}
			outTools = append(outTools, anthropicTool)
		}
		if len(outTools) > 0 {
			outRoot["tools"] = outTools
		}
	}

	if toolChoiceValue, ok := jsonutil.Get(root, "tool_choice"); ok {
		switch typed := toolChoiceValue.(type) {
		case string:
			switch typed {
			case "auto":
				outRoot["tool_choice"] = map[string]any{"type": "auto"}
			case "required":
				outRoot["tool_choice"] = map[string]any{"type": "any"}
			case "none":
				// Leave unset.
			}
		case map[string]any:
			if toolChoiceType, _ := jsonutil.String(typed, "type"); toolChoiceType == "function" {
				toolChoice := map[string]any{
					"type": "tool",
				}
				if function, ok := jsonutil.Object(typed, "function"); ok {
					toolChoice["name"] = jsonString(function, "name")
				}
				outRoot["tool_choice"] = toolChoice
			}
		}
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func jsonString(root map[string]any, path string) string {
	value, _ := jsonutil.String(root, path)
	return value
}

func parseObjectString(raw string) (map[string]any, bool) {
	value, errParse := jsonutil.ParseAnyBytes([]byte(raw))
	if errParse != nil {
		return nil, false
	}
	object, ok := value.(map[string]any)
	return object, ok
}

func convertOpenAIContentPartToClaudePart(part map[string]any) (map[string]any, bool) {
	switch jsonString(part, "type") {
	case "text":
		return map[string]any{
			"type": "text",
			"text": jsonString(part, "text"),
		}, true
	case "image_url":
		imageURLValue, ok := jsonutil.Object(part, "image_url")
		if ok {
			return convertOpenAIImageURLToClaudePart(jsonString(imageURLValue, "url"))
		}
		if imageURL, ok := jsonutil.String(part, "image_url"); ok {
			return convertOpenAIImageURLToClaudePart(imageURL)
		}
	case "file":
		if fileObject, ok := jsonutil.Object(part, "file"); ok {
			fileData := jsonString(fileObject, "file_data")
			if strings.HasPrefix(fileData, "data:") {
				semicolonIndex := strings.Index(fileData, ";")
				commaIndex := strings.Index(fileData, ",")
				if semicolonIndex != -1 && commaIndex != -1 && commaIndex > semicolonIndex {
					mediaType := strings.TrimPrefix(fileData[:semicolonIndex], "data:")
					data := fileData[commaIndex+1:]
					return map[string]any{
						"type": "document",
						"source": map[string]any{
							"type":       "base64",
							"media_type": mediaType,
							"data":       data,
						},
					}, true
				}
			}
		}
	}

	return nil, false
}

func convertOpenAIImageURLToClaudePart(imageURL string) (map[string]any, bool) {
	if imageURL == "" {
		return nil, false
	}

	if strings.HasPrefix(imageURL, "data:") {
		parts := strings.SplitN(imageURL, ",", 2)
		if len(parts) != 2 {
			return nil, false
		}
		mediaTypePart := strings.SplitN(parts[0], ";", 2)[0]
		mediaType := strings.TrimPrefix(mediaTypePart, "data:")
		if mediaType == "" {
			mediaType = "application/octet-stream"
		}
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": mediaType,
				"data":       parts[1],
			},
		}, true
	}

	return map[string]any{
		"type": "image",
		"source": map[string]any{
			"type": "url",
			"url":  imageURL,
		},
	}, true
}

func convertOpenAIToolResultContent(content any) (any, bool) {
	if content == nil {
		return "", false
	}

	switch typed := content.(type) {
	case string:
		return typed, false
	case []any:
		claudeContent := make([]any, 0)
		partCount := 0
		for _, partValue := range typed {
			switch part := partValue.(type) {
			case string:
				claudeContent = append(claudeContent, map[string]any{
					"type": "text",
					"text": part,
				})
				partCount++
			case map[string]any:
				if claudePart, ok := convertOpenAIContentPartToClaudePart(part); ok {
					claudeContent = append(claudeContent, claudePart)
					partCount++
				}
			}
		}
		if partCount > 0 || len(typed) == 0 {
			return claudeContent, true
		}
		return string(jsonutil.MarshalOrOriginal(nil, typed)), false
	case map[string]any:
		if claudePart, ok := convertOpenAIContentPartToClaudePart(typed); ok {
			return []any{claudePart}, true
		}
		return string(jsonutil.MarshalOrOriginal(nil, typed)), false
	default:
		return string(jsonutil.MarshalOrOriginal(nil, content)), false
	}
}
