// Package gemini provides request translation functionality for Gemini to
// Claude Code API compatibility using standard JSON trees.
package gemini

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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

// ConvertGeminiRequestToClaude parses and transforms a Gemini API request into
// Claude Code API format.
func ConvertGeminiRequestToClaude(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"model":      modelName,
		"max_tokens": 32000,
		"messages":   []any{},
		"metadata": map[string]any{
			"user_id": geminiClaudeUserID(),
		},
		"stream": stream,
	}

	if generationConfig, ok := jsonutil.Object(root, "generationConfig"); ok {
		applyGeminiClaudeGenerationConfig(outRoot, modelName, generationConfig)
	}

	messages := make([]any, 0)
	if systemInstruction, ok := jsonutil.Object(root, "system_instruction"); ok {
		if systemMessage, ok := geminiClaudeSystemMessage(systemInstruction); ok {
			messages = append(messages, systemMessage)
		}
	} else if systemInstruction, ok := jsonutil.Object(root, "systemInstruction"); ok {
		if systemMessage, ok := geminiClaudeSystemMessage(systemInstruction); ok {
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

			role := geminiClaudeRole(content)
			message := map[string]any{
				"role":    role,
				"content": []any{},
			}
			contentParts := make([]any, 0)

			if parts, ok := jsonutil.Array(content, "parts"); ok {
				for _, partValue := range parts {
					part, ok := partValue.(map[string]any)
					if !ok {
						continue
					}

					if text, ok := jsonutil.String(part, "text"); ok {
						contentParts = append(contentParts, map[string]any{
							"type": "text",
							"text": text,
						})
						continue
					}

					if role == "assistant" {
						if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
							toolID := geminiClaudeToolCallID()
							pendingToolIDs = append(pendingToolIDs, toolID)

							toolUse := map[string]any{
								"type":  "tool_use",
								"id":    toolID,
								"name":  geminiClaudeString(functionCall, "name"),
								"input": map[string]any{},
							}
							if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
								if argsObject, ok := argsValue.(map[string]any); ok {
									toolUse["input"] = argsObject
								}
							}
							contentParts = append(contentParts, toolUse)
							continue
						}
					}

					if functionResponse, ok := jsonutil.Object(part, "functionResponse"); ok {
						toolID := geminiClaudeNextToolID(&pendingToolIDs)
						toolResult := map[string]any{
							"type":        "tool_result",
							"tool_use_id": toolID,
						}
						if resultValue, ok := jsonutil.Get(functionResponse, "response.result"); ok {
							toolResult["content"] = geminiClaudeContentString(resultValue)
						} else if responseValue, ok := jsonutil.Get(functionResponse, "response"); ok {
							toolResult["content"] = geminiClaudeContentString(responseValue)
						} else {
							toolResult["content"] = ""
						}
						contentParts = append(contentParts, toolResult)
						continue
					}

					if imagePart, ok := geminiClaudeImagePart(part); ok {
						contentParts = append(contentParts, imagePart)
						continue
					}

					if filePart, ok := geminiClaudeFilePart(part); ok {
						contentParts = append(contentParts, filePart)
					}
				}
			}

			if len(contentParts) == 0 {
				continue
			}
			message["content"] = contentParts
			messages = append(messages, message)
		}
	}

	outRoot["messages"] = messages

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		anthropicTools := make([]any, 0)
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

				anthropicTool := map[string]any{
					"name":         geminiClaudeString(declaration, "name"),
					"description":  geminiClaudeString(declaration, "description"),
					"input_schema": map[string]any{},
				}

				if parameters, ok := jsonutil.Get(declaration, "parameters"); ok {
					if cleaned, ok := geminiClaudeCleanSchema(parameters); ok {
						anthropicTool["input_schema"] = cleaned
					}
				} else if parameters, ok := jsonutil.Get(declaration, "parametersJsonSchema"); ok {
					if cleaned, ok := geminiClaudeCleanSchema(parameters); ok {
						anthropicTool["input_schema"] = cleaned
					}
				}

				anthropicTools = append(anthropicTools, anthropicTool)
			}
		}
		if len(anthropicTools) > 0 {
			outRoot["tools"] = anthropicTools
		}
	}

	if toolConfig, ok := jsonutil.Object(root, "tool_config"); ok {
		applyGeminiClaudeToolChoice(outRoot, toolConfig)
	} else if toolConfig, ok := jsonutil.Object(root, "toolConfig"); ok {
		applyGeminiClaudeToolChoice(outRoot, toolConfig)
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func geminiClaudeUserID() string {
	if account == "" {
		uuidValue, _ := uuid.NewRandom()
		account = uuidValue.String()
	}
	if session == "" {
		uuidValue, _ := uuid.NewRandom()
		session = uuidValue.String()
	}
	if user == "" {
		sum := sha256.Sum256([]byte(account + session))
		user = hex.EncodeToString(sum[:])
	}
	return fmt.Sprintf("user_%s_account_%s_session_%s", user, account, session)
}

func applyGeminiClaudeGenerationConfig(outRoot map[string]any, modelName string, generationConfig map[string]any) {
	if value, ok := jsonutil.Get(generationConfig, "maxOutputTokens"); ok {
		outRoot["max_tokens"] = value
	}
	if value, ok := jsonutil.Get(generationConfig, "temperature"); ok {
		outRoot["temperature"] = value
	} else if value, ok := jsonutil.Get(generationConfig, "topP"); ok {
		outRoot["top_p"] = value
	}
	if stopSequences, ok := jsonutil.Array(generationConfig, "stopSequences"); ok {
		sequences := make([]string, 0, len(stopSequences))
		for _, sequenceValue := range stopSequences {
			if sequence, ok := sequenceValue.(string); ok {
				sequences = append(sequences, sequence)
			}
		}
		if len(sequences) > 0 {
			outRoot["stop_sequences"] = sequences
		}
	}

	thinkingConfig, ok := jsonutil.Object(generationConfig, "thinkingConfig")
	if !ok {
		return
	}

	modelInfo := registry.LookupModelInfo(modelName, "claude")
	supportsAdaptive := modelInfo != nil && modelInfo.Thinking != nil && len(modelInfo.Thinking.Levels) > 0
	supportsMax := supportsAdaptive && thinking.HasLevel(modelInfo.Thinking.Levels, string(thinking.LevelMax))

	if thinkingLevel, ok := jsonutil.String(thinkingConfig, "thinkingLevel"); ok {
		geminiClaudeApplyThinkingLevel(outRoot, thinkingLevel, supportsAdaptive, supportsMax)
		return
	}
	if thinkingLevel, ok := jsonutil.String(thinkingConfig, "thinking_level"); ok {
		geminiClaudeApplyThinkingLevel(outRoot, thinkingLevel, supportsAdaptive, supportsMax)
		return
	}

	if thinkingBudget, ok := jsonutil.Int64(thinkingConfig, "thinkingBudget"); ok {
		geminiClaudeApplyThinkingBudget(outRoot, int(thinkingBudget), supportsAdaptive, supportsMax)
		return
	}
	if thinkingBudget, ok := jsonutil.Int64(thinkingConfig, "thinking_budget"); ok {
		geminiClaudeApplyThinkingBudget(outRoot, int(thinkingBudget), supportsAdaptive, supportsMax)
		return
	}

	if includeThoughts, ok := jsonutil.Bool(thinkingConfig, "includeThoughts"); ok && includeThoughts {
		outRoot["thinking"] = map[string]any{"type": "enabled"}
		return
	}
	if includeThoughts, ok := jsonutil.Bool(thinkingConfig, "include_thoughts"); ok && includeThoughts {
		outRoot["thinking"] = map[string]any{"type": "enabled"}
	}
}

func geminiClaudeApplyThinkingLevel(outRoot map[string]any, level string, supportsAdaptive, supportsMax bool) {
	level = strings.ToLower(strings.TrimSpace(level))
	if supportsAdaptive {
		switch level {
		case "", "none":
			outRoot["thinking"] = map[string]any{"type": "disabled"}
			_ = jsonutil.Delete(outRoot, "output_config.effort")
		default:
			if mapped, ok := thinking.MapToClaudeEffort(level, supportsMax); ok {
				level = mapped
			}
			outRoot["thinking"] = map[string]any{"type": "adaptive"}
			_ = jsonutil.Set(outRoot, "output_config.effort", level)
		}
		return
	}

	switch level {
	case "", "none":
		outRoot["thinking"] = map[string]any{"type": "disabled"}
	case "auto":
		outRoot["thinking"] = map[string]any{"type": "enabled"}
	default:
		if budget, ok := thinking.ConvertLevelToBudget(level); ok {
			outRoot["thinking"] = map[string]any{
				"type":          "enabled",
				"budget_tokens": budget,
			}
		}
	}
}

func geminiClaudeApplyThinkingBudget(outRoot map[string]any, budget int, supportsAdaptive, supportsMax bool) {
	if supportsAdaptive {
		switch budget {
		case 0:
			outRoot["thinking"] = map[string]any{"type": "disabled"}
			_ = jsonutil.Delete(outRoot, "output_config.effort")
		default:
			if level, ok := thinking.ConvertBudgetToLevel(budget); ok {
				if mapped, ok := thinking.MapToClaudeEffort(level, supportsMax); ok {
					level = mapped
				}
				outRoot["thinking"] = map[string]any{"type": "adaptive"}
				_ = jsonutil.Set(outRoot, "output_config.effort", level)
			}
		}
		return
	}

	switch budget {
	case 0:
		outRoot["thinking"] = map[string]any{"type": "disabled"}
	case -1:
		outRoot["thinking"] = map[string]any{"type": "enabled"}
	default:
		outRoot["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": budget,
		}
	}
}

func geminiClaudeSystemMessage(systemInstruction map[string]any) (map[string]any, bool) {
	parts, ok := jsonutil.Array(systemInstruction, "parts")
	if !ok {
		return nil, false
	}

	var systemText strings.Builder
	for _, partValue := range parts {
		part, ok := partValue.(map[string]any)
		if !ok {
			continue
		}
		text, _ := jsonutil.String(part, "text")
		if text == "" {
			continue
		}
		if systemText.Len() > 0 {
			systemText.WriteByte('\n')
		}
		systemText.WriteString(text)
	}

	if systemText.Len() == 0 {
		return nil, false
	}

	return map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{
				"type": "text",
				"text": systemText.String(),
			},
		},
	}, true
}

func geminiClaudeRole(content map[string]any) string {
	role, _ := jsonutil.String(content, "role")
	switch role {
	case "model":
		return "assistant"
	case "function", "tool":
		return "user"
	default:
		return role
	}
}

func geminiClaudeToolCallID() string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	var builder strings.Builder
	for index := 0; index < 24; index++ {
		number, _ := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
		builder.WriteByte(letters[number.Int64()])
	}
	return "toolu_" + builder.String()
}

func geminiClaudeString(root map[string]any, path string) string {
	value, _ := jsonutil.String(root, path)
	return value
}

func geminiClaudeNextToolID(pendingToolIDs *[]string) string {
	if len(*pendingToolIDs) > 0 {
		toolID := (*pendingToolIDs)[0]
		*pendingToolIDs = (*pendingToolIDs)[1:]
		return toolID
	}
	return geminiClaudeToolCallID()
}

func geminiClaudeContentString(value any) string {
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

func geminiClaudeImagePart(part map[string]any) (map[string]any, bool) {
	inlineData, ok := jsonutil.Object(part, "inline_data")
	if !ok {
		inlineData, ok = jsonutil.Object(part, "inlineData")
		if !ok {
			return nil, false
		}
	}

	mimeType, _ := jsonutil.String(inlineData, "mime_type")
	if mimeType == "" {
		mimeType, _ = jsonutil.String(inlineData, "mimeType")
	}
	data, _ := jsonutil.String(inlineData, "data")
	if mimeType == "" || data == "" {
		return nil, false
	}

	return map[string]any{
		"type": "image",
		"source": map[string]any{
			"type":       "base64",
			"media_type": mimeType,
			"data":       data,
		},
	}, true
}

func geminiClaudeFilePart(part map[string]any) (map[string]any, bool) {
	fileData, ok := jsonutil.Object(part, "file_data")
	if !ok {
		fileData, ok = jsonutil.Object(part, "fileData")
		if !ok {
			return nil, false
		}
	}

	fileURI, _ := jsonutil.String(fileData, "file_uri")
	if fileURI == "" {
		fileURI, _ = jsonutil.String(fileData, "fileUri")
	}
	fileInfo := "File: " + fileURI
	if mimeType, ok := jsonutil.String(fileData, "mime_type"); ok && mimeType != "" {
		fileInfo += " (Type: " + mimeType + ")"
	} else if mimeType, ok := jsonutil.String(fileData, "mimeType"); ok && mimeType != "" {
		fileInfo += " (Type: " + mimeType + ")"
	}

	return map[string]any{
		"type": "text",
		"text": fileInfo,
	}, fileURI != ""
}

func geminiClaudeCleanSchema(value any) (map[string]any, bool) {
	bytes, errMarshal := json.Marshal(value)
	if errMarshal != nil {
		return nil, false
	}
	parsed, errParse := jsonutil.ParseAnyBytes(bytes)
	if errParse != nil {
		return nil, false
	}
	object, ok := parsed.(map[string]any)
	if !ok {
		return nil, false
	}
	object["additionalProperties"] = false
	object["$schema"] = "http://json-schema.org/draft-07/schema#"
	geminiClaudeLowercaseTypeFields(object)
	return object, true
}

func geminiClaudeLowercaseTypeFields(value any) {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if key == "type" {
				if typeString, ok := child.(string); ok {
					typed[key] = strings.ToLower(typeString)
				}
			}
			geminiClaudeLowercaseTypeFields(child)
		}
	case []any:
		for _, child := range typed {
			geminiClaudeLowercaseTypeFields(child)
		}
	}
}

func applyGeminiClaudeToolChoice(outRoot map[string]any, toolConfig map[string]any) {
	functionCallingConfig, ok := jsonutil.Object(toolConfig, "function_calling_config")
	if !ok {
		functionCallingConfig, ok = jsonutil.Object(toolConfig, "functionCallingConfig")
		if !ok {
			return
		}
	}

	mode, _ := jsonutil.String(functionCallingConfig, "mode")
	switch mode {
	case "AUTO":
		outRoot["tool_choice"] = map[string]any{"type": "auto"}
	case "NONE":
		outRoot["tool_choice"] = map[string]any{"type": "none"}
	case "ANY":
		if allowed, ok := jsonutil.Array(functionCallingConfig, "allowedFunctionNames"); ok && len(allowed) == 1 {
			if functionName, ok := allowed[0].(string); ok && functionName != "" {
				outRoot["tool_choice"] = map[string]any{
					"type": "tool",
					"name": functionName,
				}
				return
			}
		}
		outRoot["tool_choice"] = map[string]any{"type": "any"}
	}
}
