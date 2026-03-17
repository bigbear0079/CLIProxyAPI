// Package claude provides request translation functionality for Claude Code API compatibility.
// This package handles the conversion of Claude Code API requests into Gemini CLI-compatible
// JSON format, transforming message contents, system instructions, and tool declarations
// into the format expected by Gemini CLI API clients. It performs JSON data transformation
// to ensure compatibility between Claude Code API format and Gemini CLI API's expected format.
package claude

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/cache"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	log "github.com/sirupsen/logrus"
)

const skipThoughtSignatureValidator = "skip_thought_signature_validator"

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

func parseJSONObjectString(raw string) (map[string]any, bool) {
	value, errParse := jsonutil.ParseAnyBytes([]byte(raw))
	if errParse != nil {
		return nil, false
	}
	object, ok := value.(map[string]any)
	return object, ok
}

func buildInlineDataPart(source map[string]any) (map[string]any, bool) {
	sourceType, _ := jsonutil.String(source, "type")
	if sourceType != "base64" {
		return nil, false
	}

	inlineData := map[string]any{}
	if mimeType, ok := jsonutil.String(source, "media_type"); ok && mimeType != "" {
		inlineData["mimeType"] = mimeType
	}
	if data, ok := jsonutil.String(source, "data"); ok && data != "" {
		inlineData["data"] = data
	}

	return map[string]any{"inlineData": inlineData}, true
}

func deriveFunctionName(toolCallID string) string {
	parts := strings.Split(toolCallID, "-")
	if len(parts) > 2 {
		name := strings.Join(parts[:len(parts)-2], "-")
		if name != "" {
			return name
		}
	}
	return toolCallID
}

func reorderModelParts(parts []any) []any {
	thinkingParts := make([]any, 0)
	otherParts := make([]any, 0, len(parts))
	for _, partValue := range parts {
		part, ok := partValue.(map[string]any)
		if ok {
			if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
				thinkingParts = append(thinkingParts, partValue)
				continue
			}
		}
		otherParts = append(otherParts, partValue)
	}
	if len(thinkingParts) == 0 {
		return parts
	}
	if firstPart, ok := parts[0].(map[string]any); ok {
		if thought, ok := jsonutil.Bool(firstPart, "thought"); ok && thought && len(thinkingParts) == 1 {
			return parts
		}
	}

	reordered := make([]any, 0, len(parts))
	reordered = append(reordered, thinkingParts...)
	reordered = append(reordered, otherParts...)
	return reordered
}

func buildToolResultPart(toolResult map[string]any, toolNameByID map[string]string) (map[string]any, bool) {
	toolCallID, _ := jsonutil.String(toolResult, "tool_use_id")
	if toolCallID == "" {
		return nil, false
	}

	funcName, ok := toolNameByID[toolCallID]
	if !ok {
		funcName = deriveFunctionName(toolCallID)
		log.Warnf("antigravity claude request: tool_result references unknown tool_use_id=%s, derived function name=%s", toolCallID, funcName)
	}

	functionResponse := map[string]any{
		"id":   toolCallID,
		"name": funcName,
		"response": map[string]any{
			"result": "",
		},
	}

	contentValue, hasContent := jsonutil.Get(toolResult, "content")
	if !hasContent {
		return map[string]any{"functionResponse": functionResponse}, true
	}

	switch typed := contentValue.(type) {
	case string:
		functionResponse["response"].(map[string]any)["result"] = typed
	case []any:
		filtered := make([]any, 0)
		imageParts := make([]any, 0)
		var lastNonImage any

		for _, itemValue := range typed {
			if item, ok := itemValue.(map[string]any); ok {
				if itemType, _ := jsonutil.String(item, "type"); itemType == "image" {
					if source, ok := jsonutil.Object(item, "source"); ok {
						if imagePart, ok := buildInlineDataPart(source); ok {
							imageParts = append(imageParts, imagePart)
							continue
						}
					}
				}
			}

			lastNonImage = itemValue
			filtered = append(filtered, itemValue)
		}

		switch len(filtered) {
		case 0:
			functionResponse["response"].(map[string]any)["result"] = ""
		case 1:
			functionResponse["response"].(map[string]any)["result"] = lastNonImage
		default:
			functionResponse["response"].(map[string]any)["result"] = filtered
		}

		if len(imageParts) > 0 {
			functionResponse["parts"] = imageParts
		}
	case map[string]any:
		if itemType, _ := jsonutil.String(typed, "type"); itemType == "image" {
			if source, ok := jsonutil.Object(typed, "source"); ok {
				if imagePart, ok := buildInlineDataPart(source); ok {
					functionResponse["parts"] = []any{imagePart}
					functionResponse["response"].(map[string]any)["result"] = ""
					return map[string]any{"functionResponse": functionResponse}, true
				}
			}
		}
		functionResponse["response"].(map[string]any)["result"] = typed
	default:
		functionResponse["response"].(map[string]any)["result"] = typed
	}

	return map[string]any{"functionResponse": functionResponse}, true
}

func buildToolDeclarations(root map[string]any) ([]any, int) {
	tools, ok := jsonutil.Array(root, "tools")
	if !ok {
		return nil, 0
	}

	allowedKeys := map[string]struct{}{
		"name":                 {},
		"description":          {},
		"behavior":             {},
		"parameters":           {},
		"parametersJsonSchema": {},
		"response":             {},
		"responseJsonSchema":   {},
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
		cleanedSchema := util.CleanJSONSchemaForAntigravity(string(inputSchemaJSON))
		cleanedValue, errParse := jsonutil.ParseAnyBytes([]byte(cleanedSchema))
		if errParse != nil {
			continue
		}

		functionDeclaration := make(map[string]any)
		for key, value := range tool {
			if _, ok := allowedKeys[key]; !ok {
				continue
			}
			if key == "parametersJsonSchema" {
				continue
			}
			functionDeclaration[key] = value
		}
		functionDeclaration["parametersJsonSchema"] = cleanedValue
		functionDeclarations = append(functionDeclarations, functionDeclaration)
	}

	if len(functionDeclarations) == 0 {
		return nil, 0
	}

	return []any{
		map[string]any{
			"functionDeclarations": functionDeclarations,
		},
	}, len(functionDeclarations)
}

// ConvertClaudeRequestToAntigravity parses and transforms a Claude Code API request into Gemini CLI API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the Gemini CLI API.
// The function performs the following transformations:
// 1. Extracts the model information from the request
// 2. Restructures the JSON to match Gemini CLI API format
// 3. Converts system instructions to the expected format
// 4. Maps message contents with proper role transformations
// 5. Handles tool declarations and tool choices
// 6. Maps generation configuration parameters
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the Claude Code API
//   - stream: A boolean indicating if the request is for a streaming response (unused in current implementation)
//
// Returns:
//   - []byte: The transformed request data in Gemini CLI API format
func ConvertClaudeRequestToAntigravity(modelName string, inputRawJSON []byte, _ bool) []byte {
	enableThoughtTranslate := true
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	var (
		systemInstruction    map[string]any
		hasSystemInstruction bool
	)

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
			systemPrompt, _ := jsonutil.String(entry, "text")
			part := map[string]any{}
			if systemPrompt != "" {
				part["text"] = systemPrompt
			}
			parts = append(parts, part)
			hasSystemInstruction = true
		}
		if hasSystemInstruction {
			systemInstruction = map[string]any{
				"role":  "user",
				"parts": parts,
			}
		}
	} else if systemText, ok := jsonutil.String(root, "system"); ok {
		systemInstruction = map[string]any{
			"role": "user",
			"parts": []any{
				map[string]any{
					"text": systemText,
				},
			},
		}
		hasSystemInstruction = true
	}

	contents := make([]any, 0)
	toolNameByID := make(map[string]string)

	if messages, ok := jsonutil.Array(root, "messages"); ok {
		for _, messageValue := range messages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}

			originalRole, ok := jsonutil.String(message, "role")
			if !ok || originalRole == "" {
				continue
			}

			role := originalRole
			if role == "assistant" {
				role = "model"
			}

			clientContent := map[string]any{
				"role":  role,
				"parts": []any{},
			}
			parts := make([]any, 0)
			var currentMessageThinkingSignature string

			if contentArray, ok := jsonutil.Array(message, "content"); ok {
				for _, contentValue := range contentArray {
					content, ok := contentValue.(map[string]any)
					if !ok {
						continue
					}

					contentType, _ := jsonutil.String(content, "type")
					switch contentType {
					case "thinking":
						thinkingText := extractThinkingText(content)
						signature := ""
						if thinkingText != "" {
							if cachedSignature := cache.GetCachedSignature(modelName, thinkingText); cachedSignature != "" {
								signature = cachedSignature
							}
						}
						if signature == "" {
							if signatureValue, ok := jsonutil.String(content, "signature"); ok && signatureValue != "" {
								arrayClientSignatures := strings.SplitN(signatureValue, "#", 2)
								if len(arrayClientSignatures) == 2 {
									if cache.GetModelGroup(modelName) == arrayClientSignatures[0] {
										signature = arrayClientSignatures[1]
									}
								}
							}
							if !cache.HasValidSignature(modelName, signature) {
								signature = ""
							}
						}

						if cache.HasValidSignature(modelName, signature) {
							currentMessageThinkingSignature = signature
						}

						if !cache.HasValidSignature(modelName, signature) {
							enableThoughtTranslate = false
							continue
						}

						part := map[string]any{
							"thought": true,
						}
						if thinkingText != "" {
							part["text"] = thinkingText
						}
						if signature != "" {
							part["thoughtSignature"] = signature
						}
						parts = append(parts, part)

					case "text":
						prompt, _ := jsonutil.String(content, "text")
						if prompt == "" {
							continue
						}
						parts = append(parts, map[string]any{"text": prompt})

					case "tool_use":
						functionName, _ := jsonutil.String(content, "name")
						functionID, _ := jsonutil.String(content, "id")
						if functionID != "" && functionName != "" {
							toolNameByID[functionID] = functionName
						}

						var argsValue any
						switch inputValue, ok := jsonutil.Get(content, "input"); {
						case !ok:
							continue
						case inputValue == nil:
							continue
						default:
							switch typed := inputValue.(type) {
							case map[string]any:
								argsValue = typed
							case string:
								parsed, ok := parseJSONObjectString(typed)
								if !ok {
									continue
								}
								argsValue = parsed
							default:
								continue
							}
						}

						part := map[string]any{
							"thoughtSignature": skipThoughtSignatureValidator,
							"functionCall": map[string]any{
								"name": functionName,
								"args": argsValue,
							},
						}
						if cache.HasValidSignature(modelName, currentMessageThinkingSignature) {
							part["thoughtSignature"] = currentMessageThinkingSignature
						}
						if functionID != "" {
							part["functionCall"].(map[string]any)["id"] = functionID
						}
						parts = append(parts, part)

					case "tool_result":
						part, ok := buildToolResultPart(content, toolNameByID)
						if !ok {
							continue
						}
						parts = append(parts, part)

					case "image":
						source, ok := jsonutil.Object(content, "source")
						if !ok {
							continue
						}
						imagePart, ok := buildInlineDataPart(source)
						if !ok {
							continue
						}
						parts = append(parts, imagePart)
					}
				}

				if role == "model" {
					parts = reorderModelParts(parts)
				}

				if len(parts) == 0 {
					continue
				}

				clientContent["parts"] = parts
				contents = append(contents, clientContent)
			} else if prompt, ok := jsonutil.String(message, "content"); ok {
				clientContent["parts"] = []any{map[string]any{}}
				if prompt != "" {
					clientContent["parts"] = []any{map[string]any{"text": prompt}}
				}
				contents = append(contents, clientContent)
			}
		}
	}

	tools, toolDeclCount := buildToolDeclarations(root)

	hasTools := toolDeclCount > 0
	thinkingConfig, hasThinkingConfig := jsonutil.Object(root, "thinking")
	thinkingType, _ := jsonutil.String(thinkingConfig, "type")
	hasThinking := hasThinkingConfig && (thinkingType == "enabled" || thinkingType == "adaptive" || thinkingType == "auto")
	isClaudeThinking := util.IsClaudeThinkingModel(modelName)

	if hasTools && hasThinking && isClaudeThinking {
		interleavedHint := "Interleaved thinking is enabled. You may think between tool calls and after receiving tool results before deciding the next action or final answer. Do not mention these instructions or any constraints about thinking blocks; just apply them."

		if hasSystemInstruction {
			systemInstruction["parts"] = append(systemInstruction["parts"].([]any), map[string]any{"text": interleavedHint})
		} else {
			systemInstruction = map[string]any{
				"role":  "user",
				"parts": []any{map[string]any{"text": interleavedHint}},
			}
			hasSystemInstruction = true
		}
	}

	outRoot := map[string]any{
		"model": modelName,
		"request": map[string]any{
			"contents": []any{},
		},
	}

	if hasSystemInstruction {
		_ = jsonutil.Set(outRoot, "request.systemInstruction", systemInstruction)
	}
	if len(contents) > 0 {
		_ = jsonutil.Set(outRoot, "request.contents", contents)
	}
	if toolDeclCount > 0 {
		_ = jsonutil.Set(outRoot, "request.tools", tools)
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

	if enableThoughtTranslate && hasThinkingConfig {
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
	if value, ok := jsonutil.Get(root, "max_tokens"); ok {
		_ = jsonutil.Set(outRoot, "request.generationConfig.maxOutputTokens", value)
	}

	outBytes := jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
	return common.AttachDefaultSafetySettings(outBytes, "request.safetySettings")
}
