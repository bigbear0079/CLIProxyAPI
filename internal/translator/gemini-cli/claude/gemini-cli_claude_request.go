// Package claude provides request translation functionality for Claude Code API
// compatibility using standard JSON trees.
package claude

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

const geminiCLIClaudeThoughtSignature = "skip_thought_signature_validator"

// ConvertClaudeRequestToCLI parses and transforms a Claude Code API request
// into Gemini CLI API format.
func ConvertClaudeRequestToCLI(modelName string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	requestRoot := map[string]any{
		"contents":       []any{},
		"safetySettings": common.DefaultSafetySettings(),
	}
	outRoot := map[string]any{
		"model":   modelName,
		"request": requestRoot,
	}

	if systemInstruction, ok := buildGeminiCLIClaudeSystemInstruction(root); ok {
		requestRoot["systemInstruction"] = systemInstruction
	}

	if contents := buildGeminiCLIClaudeContents(root); len(contents) > 0 {
		requestRoot["contents"] = contents
	}

	if tools := buildGeminiCLIClaudeTools(root); len(tools) > 0 {
		requestRoot["tools"] = tools
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
			_ = jsonutil.Set(outRoot, "request.toolConfig.functionCallingConfig.mode", "AUTO")
		case "none":
			_ = jsonutil.Set(outRoot, "request.toolConfig.functionCallingConfig.mode", "NONE")
		case "any":
			_ = jsonutil.Set(outRoot, "request.toolConfig.functionCallingConfig.mode", "ANY")
		case "tool":
			_ = jsonutil.Set(outRoot, "request.toolConfig.functionCallingConfig.mode", "ANY")
			if toolChoiceName != "" {
				_ = jsonutil.Set(outRoot, "request.toolConfig.functionCallingConfig.allowedFunctionNames", []string{toolChoiceName})
			}
		}
	}

	if thinkingConfig, ok := jsonutil.Object(root, "thinking"); ok {
		thinkingType, _ := jsonutil.String(thinkingConfig, "type")
		switch thinkingType {
		case "enabled":
			if budget, ok := jsonutil.Get(thinkingConfig, "budget_tokens"); ok {
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.thinkingBudget", budget)
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.includeThoughts", true)
			}
		case "adaptive", "auto":
			effort := ""
			if outputConfig, ok := jsonutil.Object(root, "output_config"); ok {
				if value, ok := jsonutil.String(outputConfig, "effort"); ok {
					effort = strings.ToLower(strings.TrimSpace(value))
				}
			}
			if effort != "" {
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.thinkingLevel", effort)
			} else {
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.thinkingLevel", "high")
			}
			_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.includeThoughts", true)
		}
	}

	if value, ok := jsonutil.Get(root, "temperature"); ok {
		_ = jsonutil.Set(outRoot, "request.generationConfig.temperature", value)
	}
	if value, ok := jsonutil.Get(root, "top_p"); ok {
		_ = jsonutil.Set(outRoot, "request.generationConfig.topP", value)
	}
	if value, ok := jsonutil.Get(root, "top_k"); ok {
		_ = jsonutil.Set(outRoot, "request.generationConfig.topK", value)
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func buildGeminiCLIClaudeSystemInstruction(root map[string]any) (map[string]any, bool) {
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

func buildGeminiCLIClaudeContents(root map[string]any) []any {
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
				if part, ok := buildGeminiCLIClaudeContentPart(content); ok {
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

func buildGeminiCLIClaudeContentPart(content map[string]any) (map[string]any, bool) {
	contentType, _ := jsonutil.String(content, "type")
	switch contentType {
	case "text":
		text, _ := jsonutil.String(content, "text")
		return map[string]any{"text": text}, true
	case "tool_use":
		functionName, _ := jsonutil.String(content, "name")
		functionArgs, ok := geminiCLIClaudeToolInput(content)
		if !ok {
			return nil, false
		}
		return map[string]any{
			"thoughtSignature": geminiCLIClaudeThoughtSignature,
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
		functionName := toolCallID
		parts := strings.Split(toolCallID, "-")
		if len(parts) > 1 {
			functionName = strings.Join(parts[:len(parts)-1], "-")
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
			"inlineData": map[string]any{
				"mime_type": mimeType,
				"data":      data,
			},
		}, true
	default:
		return nil, false
	}
}

func geminiCLIClaudeToolInput(content map[string]any) (map[string]any, bool) {
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

func buildGeminiCLIClaudeTools(root map[string]any) []any {
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

		inputSchema, ok := jsonutil.Object(tool, "input_schema")
		if !ok {
			continue
		}

		inputSchemaJSON, errMarshal := jsonutil.MarshalAny(inputSchema)
		if errMarshal != nil {
			continue
		}
		cleanedSchema := util.CleanJSONSchemaForGemini(string(inputSchemaJSON))
		cleanedValue, errParse := jsonutil.ParseAnyBytes([]byte(cleanedSchema))
		if errParse != nil {
			continue
		}

		functionDeclaration := make(map[string]any)
		for key, value := range tool {
			switch key {
			case "strict", "input_examples", "type", "cache_control", "defer_loading", "eager_input_streaming", "input_schema":
				continue
			default:
				functionDeclaration[key] = value
			}
		}
		functionDeclaration["parametersJsonSchema"] = cleanedValue
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
