// Package claude provides request translation functionality for Claude API
// using standard JSON trees.
package claude

import (
	"bytes"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
)

const geminiClaudeThoughtSignature = "skip_thought_signature_validator"

// ConvertClaudeRequestToGemini parses a Claude API request and returns a
// Gemini request body ready to be sent upstream.
func ConvertClaudeRequestToGemini(modelName string, inputRawJSON []byte, _ bool) []byte {
	sanitizedRawJSON := bytes.Replace(inputRawJSON, []byte(`"url":{"type":"string","format":"uri",`), []byte(`"url":{"type":"string",`), -1)
	root := jsonutil.ParseObjectBytesOrEmpty(sanitizedRawJSON)

	outRoot := map[string]any{
		"contents":       []any{},
		"safetySettings": common.DefaultSafetySettings(),
		"model":          modelName,
	}

	if systemInstruction, ok := buildGeminiClaudeSystemInstruction(root); ok {
		outRoot["system_instruction"] = systemInstruction
	}

	if contents := buildGeminiClaudeContents(root); len(contents) > 0 {
		outRoot["contents"] = contents
	}

	if tools := buildGeminiClaudeTools(root); len(tools) > 0 {
		outRoot["tools"] = tools
	}

	if toolChoiceValue, ok := jsonutil.Get(root, "tool_choice"); ok {
		toolChoiceType := ""
		toolChoiceName := ""
		switch typed := toolChoiceValue.(type) {
		case map[string]any:
			toolChoiceType, _ = jsonutil.String(typed, "type")
			toolChoiceName, _ = jsonutil.String(typed, "name")
		case string:
			toolChoiceType = typed
		}

		switch toolChoiceType {
		case "auto":
			_ = jsonutil.Set(outRoot, "toolConfig.functionCallingConfig.mode", "AUTO")
		case "none":
			_ = jsonutil.Set(outRoot, "toolConfig.functionCallingConfig.mode", "NONE")
		case "any":
			_ = jsonutil.Set(outRoot, "toolConfig.functionCallingConfig.mode", "ANY")
		case "tool":
			_ = jsonutil.Set(outRoot, "toolConfig.functionCallingConfig.mode", "ANY")
			if toolChoiceName != "" {
				_ = jsonutil.Set(outRoot, "toolConfig.functionCallingConfig.allowedFunctionNames", []string{toolChoiceName})
			}
		}
	}

	if thinkingConfig, ok := jsonutil.Object(root, "thinking"); ok {
		thinkingType, _ := jsonutil.String(thinkingConfig, "type")
		switch thinkingType {
		case "enabled":
			if budget, ok := jsonutil.Get(thinkingConfig, "budget_tokens"); ok {
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.thinkingBudget", budget)
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.includeThoughts", true)
			}
		case "adaptive", "auto":
			effort := ""
			if outputConfig, ok := jsonutil.Object(root, "output_config"); ok {
				if value, ok := jsonutil.String(outputConfig, "effort"); ok {
					effort = strings.ToLower(strings.TrimSpace(value))
				}
			}
			if effort != "" {
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.thinkingLevel", effort)
			} else {
				maxBudget := 0
				if modelInfo := registry.LookupModelInfo(modelName, "gemini"); modelInfo != nil && modelInfo.Thinking != nil {
					maxBudget = modelInfo.Thinking.Max
				}
				if maxBudget > 0 {
					_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.thinkingBudget", maxBudget)
				} else {
					_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.thinkingLevel", "high")
				}
			}
			_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.includeThoughts", true)
		}
	}

	if value, ok := jsonutil.Get(root, "temperature"); ok {
		_ = jsonutil.Set(outRoot, "generationConfig.temperature", value)
	}
	if value, ok := jsonutil.Get(root, "top_p"); ok {
		_ = jsonutil.Set(outRoot, "generationConfig.topP", value)
	}
	if value, ok := jsonutil.Get(root, "top_k"); ok {
		_ = jsonutil.Set(outRoot, "generationConfig.topK", value)
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func buildGeminiClaudeSystemInstruction(root map[string]any) (map[string]any, bool) {
	if systemEntries, ok := jsonutil.Array(root, "system"); ok {
		parts := make([]any, 0, len(systemEntries))
		for _, entryValue := range systemEntries {
			entry, ok := entryValue.(map[string]any)
			if !ok {
				continue
			}
			entryType, _ := jsonutil.String(entry, "type")
			if entryType != "text" {
				continue
			}
			text, _ := jsonutil.String(entry, "text")
			parts = append(parts, map[string]any{"text": text})
		}
		if len(parts) > 0 {
			return map[string]any{
				"role":  "user",
				"parts": parts,
			}, true
		}
	}

	if systemText, ok := jsonutil.String(root, "system"); ok {
		return map[string]any{
			"role": "user",
			"parts": []any{
				map[string]any{"text": systemText},
			},
		}, true
	}

	return nil, false
}

func buildGeminiClaudeContents(root map[string]any) []any {
	messages, ok := jsonutil.Array(root, "messages")
	if !ok {
		return nil
	}

	contents := make([]any, 0, len(messages))
	for _, messageValue := range messages {
		message, ok := messageValue.(map[string]any)
		if !ok {
			continue
		}

		role, ok := jsonutil.String(message, "role")
		if !ok || role == "" {
			continue
		}
		if role == "assistant" {
			role = "model"
		}

		parts := make([]any, 0)
		if contentArray, ok := jsonutil.Array(message, "content"); ok {
			for _, contentValue := range contentArray {
				content, ok := contentValue.(map[string]any)
				if !ok {
					continue
				}
				if part, ok := buildGeminiClaudeContentPart(content); ok {
					parts = append(parts, part)
				}
			}
		} else if contentText, ok := jsonutil.String(message, "content"); ok {
			parts = append(parts, map[string]any{"text": contentText})
		}

		if len(parts) == 0 {
			continue
		}

		contents = append(contents, map[string]any{
			"role":  role,
			"parts": parts,
		})
	}

	return contents
}

func buildGeminiClaudeContentPart(content map[string]any) (map[string]any, bool) {
	contentType, _ := jsonutil.String(content, "type")
	switch contentType {
	case "text":
		text, _ := jsonutil.String(content, "text")
		return map[string]any{"text": text}, true
	case "tool_use":
		functionName, _ := jsonutil.String(content, "name")
		if toolUseID, ok := jsonutil.String(content, "id"); ok && toolUseID != "" {
			if derived := toolNameFromClaudeToolUseID(toolUseID); derived != "" {
				functionName = derived
			}
		}
		functionArgs, ok := geminiClaudeToolInput(content)
		if !ok {
			return nil, false
		}
		return map[string]any{
			"thoughtSignature": geminiClaudeThoughtSignature,
			"functionCall": map[string]any{
				"name": functionName,
				"args": functionArgs,
			},
		}, true
	case "tool_result":
		toolCallID, _ := jsonutil.String(content, "tool_use_id")
		if toolCallID == "" {
			return nil, false
		}
		functionName := toolNameFromClaudeToolUseID(toolCallID)
		if functionName == "" {
			functionName = toolCallID
		}
		response := map[string]any{}
		if value, ok := jsonutil.Get(content, "content"); ok {
			response["result"] = value
		}
		return map[string]any{
			"functionResponse": map[string]any{
				"name":     functionName,
				"response": response,
			},
		}, true
	case "image":
		source, ok := jsonutil.Object(content, "source")
		if !ok {
			return nil, false
		}
		sourceType, _ := jsonutil.String(source, "type")
		if sourceType != "base64" {
			return nil, false
		}
		mimeType, _ := jsonutil.String(source, "media_type")
		data, _ := jsonutil.String(source, "data")
		if mimeType == "" || data == "" {
			return nil, false
		}
		return map[string]any{
			"inline_data": map[string]any{
				"mime_type": mimeType,
				"data":      data,
			},
		}, true
	default:
		return nil, false
	}
}

func geminiClaudeToolInput(content map[string]any) (map[string]any, bool) {
	inputValue, ok := jsonutil.Get(content, "input")
	if !ok || inputValue == nil {
		return nil, false
	}
	switch typed := inputValue.(type) {
	case map[string]any:
		return typed, true
	case string:
		parsed, errParse := jsonutil.ParseAnyBytes([]byte(typed))
		if errParse != nil {
			return nil, false
		}
		object, ok := parsed.(map[string]any)
		return object, ok
	default:
		return nil, false
	}
}

func buildGeminiClaudeTools(root map[string]any) []any {
	tools, ok := jsonutil.Array(root, "tools")
	if !ok {
		return nil
	}

	functionDeclarations := make([]any, 0)
	for _, toolValue := range tools {
		tool, ok := toolValue.(map[string]any)
		if !ok {
			continue
		}

		inputSchema, ok := jsonutil.Get(tool, "input_schema")
		if !ok {
			continue
		}

		functionDeclaration := make(map[string]any)
		for key, value := range tool {
			switch key {
			case "strict", "input_examples", "type", "cache_control", "defer_loading", "input_schema":
				continue
			default:
				functionDeclaration[key] = value
			}
		}
		functionDeclaration["parametersJsonSchema"] = inputSchema
		functionDeclarations = append(functionDeclarations, functionDeclaration)
	}

	if len(functionDeclarations) == 0 {
		return nil
	}

	return []any{
		map[string]any{
			"functionDeclarations": functionDeclarations,
		},
	}
}

func toolNameFromClaudeToolUseID(toolUseID string) string {
	parts := strings.Split(toolUseID, "-")
	if len(parts) <= 1 {
		return ""
	}
	return strings.Join(parts[:len(parts)-1], "-")
}
