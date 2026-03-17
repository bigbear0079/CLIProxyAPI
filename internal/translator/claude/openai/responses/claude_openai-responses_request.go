package responses

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

// ConvertOpenAIResponsesRequestToClaude transforms an OpenAI Responses API
// request into a Claude Messages API request.
func ConvertOpenAIResponsesRequestToClaude(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"model":      modelName,
		"max_tokens": 32000,
		"messages":   []any{},
		"metadata": map[string]any{
			"user_id": claudeResponsesUserID(),
		},
		"stream": stream,
	}

	if maxOutputTokens, ok := jsonutil.Get(root, "max_output_tokens"); ok {
		outRoot["max_tokens"] = maxOutputTokens
	}

	if reasoningEffort, ok := jsonutil.String(root, "reasoning.effort"); ok {
		applyClaudeResponsesReasoning(outRoot, modelName, reasoningEffort)
	}

	messages := make([]any, 0)
	instructionsText := ""
	extractedFromSystem := false

	if instructions, ok := jsonutil.String(root, "instructions"); ok {
		instructionsText = instructions
		if instructionsText != "" {
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": instructionsText,
			})
		}
	}

	if instructionsText == "" {
		if inputItems, ok := jsonutil.Array(root, "input"); ok {
			for _, itemValue := range inputItems {
				item, ok := itemValue.(map[string]any)
				if !ok {
					continue
				}
				role, _ := jsonutil.String(item, "role")
				if !strings.EqualFold(role, "system") {
					continue
				}
				instructionsText = claudeResponsesSystemInstruction(item)
				if instructionsText != "" {
					messages = append(messages, map[string]any{
						"role":    "user",
						"content": instructionsText,
					})
					extractedFromSystem = true
				}
				break
			}
		}
	}

	if inputItems, ok := jsonutil.Array(root, "input"); ok {
		for _, itemValue := range inputItems {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(item, "role")
			if extractedFromSystem && strings.EqualFold(role, "system") {
				continue
			}

			itemType := claudeResponsesItemType(item)
			switch itemType {
			case "message":
				if message, ok := buildClaudeResponsesMessage(item); ok {
					messages = append(messages, message)
				}
			case "function_call":
				if message, ok := buildClaudeResponsesFunctionCallMessage(item); ok {
					messages = append(messages, message)
				}
			case "function_call_output":
				if message, ok := buildClaudeResponsesFunctionOutputMessage(item); ok {
					messages = append(messages, message)
				}
			}
		}
	}

	outRoot["messages"] = messages

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		claudeTools := make([]any, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			claudeTool := map[string]any{}
			if name, ok := jsonutil.String(tool, "name"); ok {
				claudeTool["name"] = name
			}
			if description, ok := jsonutil.String(tool, "description"); ok {
				claudeTool["description"] = description
			}
			if parameters, ok := jsonutil.Get(tool, "parameters"); ok {
				claudeTool["input_schema"] = parameters
			} else if parameters, ok := jsonutil.Get(tool, "parametersJsonSchema"); ok {
				claudeTool["input_schema"] = parameters
			}
			claudeTools = append(claudeTools, claudeTool)
		}
		if len(claudeTools) > 0 {
			outRoot["tools"] = claudeTools
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
			}
		case map[string]any:
			toolChoiceType, _ := jsonutil.String(typed, "type")
			if toolChoiceType == "function" {
				functionName, _ := jsonutil.String(typed, "function.name")
				outRoot["tool_choice"] = map[string]any{
					"name": functionName,
					"type": "tool",
				}
			}
		}
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func claudeResponsesUserID() string {
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

func applyClaudeResponsesReasoning(outRoot map[string]any, modelName, effort string) {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort == "" {
		return
	}

	modelInfo := registry.LookupModelInfo(modelName, "claude")
	supportsAdaptive := modelInfo != nil && modelInfo.Thinking != nil && len(modelInfo.Thinking.Levels) > 0
	supportsMax := supportsAdaptive && thinking.HasLevel(modelInfo.Thinking.Levels, string(thinking.LevelMax))

	if supportsAdaptive {
		switch effort {
		case "none":
			_ = jsonutil.Set(outRoot, "thinking.type", "disabled")
			_ = jsonutil.Delete(outRoot, "thinking.budget_tokens")
			_ = jsonutil.Delete(outRoot, "output_config.effort")
		case "auto":
			_ = jsonutil.Set(outRoot, "thinking.type", "adaptive")
			_ = jsonutil.Delete(outRoot, "thinking.budget_tokens")
			_ = jsonutil.Delete(outRoot, "output_config.effort")
		default:
			if mapped, ok := thinking.MapToClaudeEffort(effort, supportsMax); ok {
				effort = mapped
			}
			_ = jsonutil.Set(outRoot, "thinking.type", "adaptive")
			_ = jsonutil.Delete(outRoot, "thinking.budget_tokens")
			_ = jsonutil.Set(outRoot, "output_config.effort", effort)
		}
		return
	}

	budget, ok := thinking.ConvertLevelToBudget(effort)
	if !ok {
		return
	}
	switch budget {
	case 0:
		_ = jsonutil.Set(outRoot, "thinking.type", "disabled")
	case -1:
		_ = jsonutil.Set(outRoot, "thinking.type", "enabled")
	default:
		if budget > 0 {
			_ = jsonutil.Set(outRoot, "thinking.type", "enabled")
			_ = jsonutil.Set(outRoot, "thinking.budget_tokens", budget)
		}
	}
}

func claudeResponsesSystemInstruction(item map[string]any) string {
	contentValue, ok := jsonutil.Get(item, "content")
	if !ok {
		return ""
	}

	switch typed := contentValue.(type) {
	case []any:
		var builder strings.Builder
		for _, partValue := range typed {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}
			text, _ := jsonutil.String(part, "text")
			if builder.Len() > 0 && text != "" {
				builder.WriteByte('\n')
			}
			builder.WriteString(text)
		}
		return builder.String()
	case string:
		return typed
	default:
		return ""
	}
}

func claudeResponsesItemType(item map[string]any) string {
	itemType, _ := jsonutil.String(item, "type")
	if itemType == "" {
		if role, ok := jsonutil.String(item, "role"); ok && role != "" {
			return "message"
		}
	}
	return itemType
}

func buildClaudeResponsesMessage(item map[string]any) (map[string]any, bool) {
	role := ""
	var textBuilder strings.Builder
	contentParts := make([]any, 0)
	hasImage := false
	hasFile := false

	contentValue, ok := jsonutil.Get(item, "content")
	if ok {
		switch typed := contentValue.(type) {
		case []any:
			for _, partValue := range typed {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}

				partType, _ := jsonutil.String(part, "type")
				switch partType {
				case "input_text", "output_text":
					text, ok := jsonutil.String(part, "text")
					if !ok {
						continue
					}
					textBuilder.WriteString(text)
					contentParts = append(contentParts, map[string]any{
						"type": "text",
						"text": text,
					})
					if partType == "input_text" {
						role = "user"
					} else {
						role = "assistant"
					}
				case "input_image":
					imageURL := ""
					if value, ok := jsonutil.String(part, "image_url"); ok {
						imageURL = value
					}
					if imageURL == "" {
						imageURL, _ = jsonutil.String(part, "url")
					}
					if contentPart, ok := buildClaudeResponsesImagePart(imageURL); ok {
						contentParts = append(contentParts, contentPart)
						if role == "" {
							role = "user"
						}
						hasImage = true
					}
				case "input_file":
					fileData, _ := jsonutil.String(part, "file_data")
					if contentPart, ok := buildClaudeResponsesFilePart(fileData); ok {
						contentParts = append(contentParts, contentPart)
						if role == "" {
							role = "user"
						}
						hasFile = true
					}
				}
			}
		case string:
			textBuilder.WriteString(typed)
		}
	}

	if role == "" {
		itemRole, _ := jsonutil.String(item, "role")
		switch itemRole {
		case "user", "assistant", "system":
			role = itemRole
		default:
			role = "user"
		}
	}

	if len(contentParts) > 0 {
		message := map[string]any{
			"role": role,
		}
		if len(contentParts) == 1 && !hasImage && !hasFile {
			if textPart, ok := contentParts[0].(map[string]any); ok {
				message["content"] = textPart["text"]
			}
		} else {
			message["content"] = contentParts
		}
		return message, true
	}

	if textBuilder.Len() > 0 || role == "system" {
		return map[string]any{
			"role":    role,
			"content": textBuilder.String(),
		}, true
	}

	return nil, false
}

func buildClaudeResponsesImagePart(imageURL string) (map[string]any, bool) {
	if imageURL == "" {
		return nil, false
	}

	if strings.HasPrefix(imageURL, "data:") {
		mediaType := "application/octet-stream"
		data := ""
		trimmed := strings.TrimPrefix(imageURL, "data:")
		mediaAndData := strings.SplitN(trimmed, ";base64,", 2)
		if len(mediaAndData) == 2 {
			if mediaAndData[0] != "" {
				mediaType = mediaAndData[0]
			}
			data = mediaAndData[1]
		}
		if data == "" {
			return nil, false
		}
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": mediaType,
				"data":       data,
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

func buildClaudeResponsesFilePart(fileData string) (map[string]any, bool) {
	if fileData == "" {
		return nil, false
	}

	mediaType := "application/octet-stream"
	data := fileData
	if strings.HasPrefix(fileData, "data:") {
		trimmed := strings.TrimPrefix(fileData, "data:")
		mediaAndData := strings.SplitN(trimmed, ";base64,", 2)
		if len(mediaAndData) == 2 {
			if mediaAndData[0] != "" {
				mediaType = mediaAndData[0]
			}
			data = mediaAndData[1]
		}
	}

	return map[string]any{
		"type": "document",
		"source": map[string]any{
			"type":       "base64",
			"media_type": mediaType,
			"data":       data,
		},
	}, true
}

func buildClaudeResponsesFunctionCallMessage(item map[string]any) (map[string]any, bool) {
	callID, _ := jsonutil.String(item, "call_id")
	if callID == "" {
		callID = claudeResponsesToolCallID()
	}

	toolUse := map[string]any{
		"type": "tool_use",
		"id":   callID,
	}
	if name, ok := jsonutil.String(item, "name"); ok {
		toolUse["name"] = name
	}
	toolUse["input"] = map[string]any{}

	arguments, _ := jsonutil.String(item, "arguments")
	if arguments != "" && json.Valid([]byte(arguments)) {
		if parsedArguments, errParse := jsonutil.ParseAnyBytes([]byte(arguments)); errParse == nil {
			if object, ok := parsedArguments.(map[string]any); ok {
				toolUse["input"] = object
			}
		}
	}

	return map[string]any{
		"role":    "assistant",
		"content": []any{toolUse},
	}, true
}

func buildClaudeResponsesFunctionOutputMessage(item map[string]any) (map[string]any, bool) {
	callID, _ := jsonutil.String(item, "call_id")
	if callID == "" {
		return nil, false
	}

	toolResult := map[string]any{
		"type":        "tool_result",
		"tool_use_id": callID,
		"content":     claudeResponsesStringValue(item["output"]),
	}

	return map[string]any{
		"role":    "user",
		"content": []any{toolResult},
	}, true
}

func claudeResponsesStringValue(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
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
		marshaled, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return ""
		}
		return string(marshaled)
	}
}

func claudeResponsesToolCallID() string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	var builder strings.Builder
	for index := 0; index < 24; index++ {
		number, _ := rand.Int(rand.Reader, big.NewInt(int64(len(letters))))
		builder.WriteByte(letters[number.Int64()])
	}
	return "toolu_" + builder.String()
}
