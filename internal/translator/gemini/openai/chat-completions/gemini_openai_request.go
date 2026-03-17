// Package openai provides request translation functionality for OpenAI to
// Gemini API compatibility using standard JSON trees.
package chat_completions

import (
	"encoding/json"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	log "github.com/sirupsen/logrus"
)

const geminiFunctionThoughtSignature = "skip_thought_signature_validator"

// ConvertOpenAIRequestToGemini converts an OpenAI Chat Completions request
// into a Gemini request payload.
func ConvertOpenAIRequestToGemini(modelName string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"contents":       []any{},
		"safetySettings": common.DefaultSafetySettings(),
		"model":          modelName,
	}

	if generationConfig, ok := jsonutil.Get(root, "generationConfig"); ok {
		outRoot["generationConfig"] = generationConfig
	}

	if effort, ok := jsonutil.String(root, "reasoning_effort"); ok {
		effort = strings.ToLower(strings.TrimSpace(effort))
		if effort != "" {
			if effort == "auto" {
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.thinkingBudget", -1)
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.includeThoughts", true)
			} else {
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.thinkingLevel", effort)
				_ = jsonutil.Set(outRoot, "generationConfig.thinkingConfig.includeThoughts", effort != "none")
			}
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
	if value, ok := jsonutil.Int64(root, "n"); ok && value > 1 {
		_ = jsonutil.Set(outRoot, "generationConfig.candidateCount", value)
	}

	if modalities, ok := jsonutil.Array(root, "modalities"); ok {
		responseModalities := make([]string, 0, len(modalities))
		for _, modalityValue := range modalities {
			modality, ok := modalityValue.(string)
			if !ok {
				continue
			}
			switch strings.ToLower(modality) {
			case "text":
				responseModalities = append(responseModalities, "TEXT")
			case "image":
				responseModalities = append(responseModalities, "IMAGE")
			}
		}
		if len(responseModalities) > 0 {
			_ = jsonutil.Set(outRoot, "generationConfig.responseModalities", responseModalities)
		}
	}

	if imageConfig, ok := jsonutil.Object(root, "image_config"); ok {
		if value, ok := jsonutil.Get(imageConfig, "aspect_ratio"); ok {
			_ = jsonutil.Set(outRoot, "generationConfig.imageConfig.aspectRatio", value)
		}
		if value, ok := jsonutil.Get(imageConfig, "image_size"); ok {
			_ = jsonutil.Set(outRoot, "generationConfig.imageConfig.imageSize", value)
		}
	}

	toolNameByID := map[string]string{}
	toolResponses := map[string]any{}
	if messages, ok := jsonutil.Array(root, "messages"); ok {
		for _, messageValue := range messages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}
			role, _ := jsonutil.String(message, "role")
			if role != "assistant" {
				continue
			}
			toolCalls, ok := jsonutil.Array(message, "tool_calls")
			if !ok {
				continue
			}
			for _, toolCallValue := range toolCalls {
				toolCall, ok := toolCallValue.(map[string]any)
				if !ok {
					continue
				}
				toolType, _ := jsonutil.String(toolCall, "type")
				if toolType != "function" {
					continue
				}
				function, ok := jsonutil.Object(toolCall, "function")
				if !ok {
					continue
				}
				toolCallID, _ := jsonutil.String(toolCall, "id")
				toolName, _ := jsonutil.String(function, "name")
				if toolCallID != "" && toolName != "" {
					toolNameByID[toolCallID] = toolName
				}
			}
		}

		for _, messageValue := range messages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}
			role, _ := jsonutil.String(message, "role")
			if role != "tool" {
				continue
			}
			toolCallID, _ := jsonutil.String(message, "tool_call_id")
			if toolCallID == "" {
				continue
			}
			contentValue, hasContent := jsonutil.Get(message, "content")
			if !hasContent {
				toolResponses[toolCallID] = nil
				continue
			}
			toolResponses[toolCallID] = contentValue
		}

		systemParts := make([]any, 0)
		contents := make([]any, 0, len(messages))
		messageCount := len(messages)
		for _, messageValue := range messages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}
			role, _ := jsonutil.String(message, "role")
			contentValue, _ := jsonutil.Get(message, "content")

			if (role == "system" || role == "developer") && messageCount > 1 {
				systemParts = append(systemParts, buildGeminiSystemInstructionParts(contentValue)...)
				continue
			}

			if role == "user" || ((role == "system" || role == "developer") && messageCount == 1) {
				contents = append(contents, map[string]any{
					"role":  "user",
					"parts": buildGeminiUserParts(contentValue),
				})
				continue
			}

			if role != "assistant" {
				continue
			}

			node, functionIDs := buildGeminiAssistantNode(message, contentValue)
			contents = append(contents, node)

			if toolNode, ok := buildGeminiToolResponseNode(functionIDs, toolNameByID, toolResponses); ok {
				contents = append(contents, toolNode)
			}
		}

		if len(systemParts) > 0 {
			outRoot["systemInstruction"] = map[string]any{
				"role":  "user",
				"parts": systemParts,
			}
		}
		outRoot["contents"] = contents
	}

	if tools, ok := jsonutil.Array(root, "tools"); ok && len(tools) > 0 {
		outTools := buildGeminiTools(tools)
		if len(outTools) > 0 {
			outRoot["tools"] = outTools
		}
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func buildGeminiSystemInstructionParts(contentValue any) []any {
	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case string:
		if typed != "" {
			parts = append(parts, map[string]any{"text": typed})
		}
	case map[string]any:
		if part, ok := buildGeminiTextPart(typed); ok {
			parts = append(parts, part)
		}
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if part, ok := buildGeminiTextPart(item); ok {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func buildGeminiUserParts(contentValue any) []any {
	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case string:
		if typed != "" {
			parts = append(parts, map[string]any{"text": typed})
		}
	case map[string]any:
		if part, ok := buildGeminiUserPart(typed); ok {
			parts = append(parts, part)
		}
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if part, ok := buildGeminiUserPart(item); ok {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func buildGeminiUserPart(item map[string]any) (map[string]any, bool) {
	itemType, _ := jsonutil.String(item, "type")
	switch itemType {
	case "", "text":
		return buildGeminiTextPart(item)
	case "image_url":
		imageURL := ""
		if imageURLValue, ok := jsonutil.Object(item, "image_url"); ok {
			imageURL, _ = jsonutil.String(imageURLValue, "url")
		}
		if imageURL == "" {
			imageURL, _ = jsonutil.String(item, "image_url")
		}
		return buildGeminiInlineImagePart(imageURL)
	case "file":
		fileValue, ok := jsonutil.Object(item, "file")
		if !ok {
			return nil, false
		}
		fileData, _ := jsonutil.String(fileValue, "file_data")
		if fileData == "" {
			return nil, false
		}
		filename, _ := jsonutil.String(fileValue, "filename")
		ext := strings.ToLower(geminiFileExtension(filename))
		mimeType, ok := misc.MimeTypes[ext]
		if !ok {
			log.Warnf("Unknown file name extension '%s' in user message, skip", ext)
			return nil, false
		}
		return map[string]any{
			"inlineData": map[string]any{
				"mime_type": mimeType,
				"data":      fileData,
			},
		}, true
	default:
		return nil, false
	}
}

func buildGeminiAssistantNode(message map[string]any, contentValue any) (map[string]any, []string) {
	parts := buildGeminiAssistantParts(contentValue)
	functionIDs := make([]string, 0)

	if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
		for _, toolCallValue := range toolCalls {
			toolCall, ok := toolCallValue.(map[string]any)
			if !ok {
				continue
			}
			part, functionID, ok := buildGeminiFunctionCallPart(toolCall)
			if !ok {
				continue
			}
			parts = append(parts, part)
			if functionID != "" {
				functionIDs = append(functionIDs, functionID)
			}
		}
	}

	return map[string]any{
		"role":  "model",
		"parts": parts,
	}, functionIDs
}

func buildGeminiAssistantParts(contentValue any) []any {
	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case string:
		if typed != "" {
			parts = append(parts, map[string]any{"text": typed})
		}
	case map[string]any:
		if part, ok := buildGeminiAssistantPart(typed); ok {
			parts = append(parts, part)
		}
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if part, ok := buildGeminiAssistantPart(item); ok {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func buildGeminiAssistantPart(item map[string]any) (map[string]any, bool) {
	itemType, _ := jsonutil.String(item, "type")
	switch itemType {
	case "", "text":
		return buildGeminiTextPart(item)
	case "image_url":
		imageURL := ""
		if imageURLValue, ok := jsonutil.Object(item, "image_url"); ok {
			imageURL, _ = jsonutil.String(imageURLValue, "url")
		}
		if imageURL == "" {
			imageURL, _ = jsonutil.String(item, "image_url")
		}
		return buildGeminiInlineImagePart(imageURL)
	default:
		return nil, false
	}
}

func buildGeminiTextPart(item map[string]any) (map[string]any, bool) {
	text, _ := jsonutil.String(item, "text")
	if text == "" {
		return nil, false
	}
	return map[string]any{"text": text}, true
}

func buildGeminiInlineImagePart(imageURL string) (map[string]any, bool) {
	mimeType, data, ok := geminiParseDataURL(imageURL)
	if !ok {
		return nil, false
	}
	return map[string]any{
		"inlineData": map[string]any{
			"mime_type": mimeType,
			"data":      data,
		},
		"thoughtSignature": geminiFunctionThoughtSignature,
	}, true
}

func geminiParseDataURL(imageURL string) (string, string, bool) {
	if !strings.HasPrefix(imageURL, "data:") {
		return "", "", false
	}
	pieces := strings.SplitN(imageURL[5:], ";", 2)
	if len(pieces) != 2 || !strings.HasPrefix(pieces[1], "base64,") || len(pieces[1]) <= len("base64,") {
		return "", "", false
	}
	return pieces[0], pieces[1][len("base64,"):], true
}

func buildGeminiFunctionCallPart(toolCall map[string]any) (map[string]any, string, bool) {
	toolType, _ := jsonutil.String(toolCall, "type")
	if toolType != "function" {
		return nil, "", false
	}

	function, ok := jsonutil.Object(toolCall, "function")
	if !ok {
		return nil, "", false
	}

	arguments, _ := jsonutil.String(function, "arguments")
	argsValue := any(map[string]any{})
	if rawArguments, ok := geminiRawJSON(arguments); ok {
		argsValue = rawArguments
	} else if arguments != "" {
		argsValue = map[string]any{"params": arguments}
	}

	functionID, _ := jsonutil.String(toolCall, "id")
	functionName, _ := jsonutil.String(function, "name")
	return map[string]any{
		"functionCall": map[string]any{
			"name": functionName,
			"args": argsValue,
		},
		"thoughtSignature": geminiFunctionThoughtSignature,
	}, functionID, true
}

func buildGeminiToolResponseNode(functionIDs []string, toolNameByID map[string]string, toolResponses map[string]any) (map[string]any, bool) {
	parts := make([]any, 0, len(functionIDs))
	for _, functionID := range functionIDs {
		functionName, ok := toolNameByID[functionID]
		if !ok || functionName == "" {
			continue
		}

		response := map[string]any{}
		if value, ok := toolResponses[functionID]; ok {
			switch typed := value.(type) {
			case nil:
				// Keep the response object empty to match previous null handling.
			case string:
				if rawValue, ok := geminiRawJSON(typed); ok {
					response["result"] = rawValue
				} else {
					response["result"] = typed
				}
			default:
				response["result"] = typed
			}
		} else {
			response["result"] = map[string]any{}
		}

		parts = append(parts, map[string]any{
			"functionResponse": map[string]any{
				"name":     functionName,
				"response": response,
			},
		})
	}

	if len(parts) == 0 {
		return nil, false
	}

	return map[string]any{
		"role":  "user",
		"parts": parts,
	}, true
}

func buildGeminiTools(tools []any) []any {
	functionDeclarations := make([]any, 0)
	googleSearchTools := make([]any, 0)
	codeExecutionTools := make([]any, 0)
	urlContextTools := make([]any, 0)

	for _, toolValue := range tools {
		tool, ok := toolValue.(map[string]any)
		if !ok {
			continue
		}

		toolType, _ := jsonutil.String(tool, "type")
		if toolType == "function" {
			function, ok := jsonutil.Object(tool, "function")
			if ok {
				functionDeclaration := geminiCloneObject(function)
				if parameters, ok := functionDeclaration["parameters"]; ok {
					functionDeclaration["parametersJsonSchema"] = parameters
					delete(functionDeclaration, "parameters")
				} else if _, ok := functionDeclaration["parametersJsonSchema"]; !ok {
					functionDeclaration["parametersJsonSchema"] = map[string]any{
						"type":       "object",
						"properties": map[string]any{},
					}
				}
				delete(functionDeclaration, "strict")
				functionDeclarations = append(functionDeclarations, functionDeclaration)
			}
		}

		if value, ok := jsonutil.Get(tool, "google_search"); ok {
			googleSearchTools = append(googleSearchTools, map[string]any{"googleSearch": value})
		}
		if value, ok := jsonutil.Get(tool, "code_execution"); ok {
			codeExecutionTools = append(codeExecutionTools, map[string]any{"codeExecution": value})
		}
		if value, ok := jsonutil.Get(tool, "url_context"); ok {
			urlContextTools = append(urlContextTools, map[string]any{"urlContext": value})
		}
	}

	outTools := make([]any, 0, 1+len(googleSearchTools)+len(codeExecutionTools)+len(urlContextTools))
	if len(functionDeclarations) > 0 {
		outTools = append(outTools, map[string]any{
			"functionDeclarations": functionDeclarations,
		})
	}
	outTools = append(outTools, googleSearchTools...)
	outTools = append(outTools, codeExecutionTools...)
	outTools = append(outTools, urlContextTools...)
	return outTools
}

func geminiRawJSON(value string) (json.RawMessage, bool) {
	value = strings.TrimSpace(value)
	if value == "" || !json.Valid([]byte(value)) {
		return nil, false
	}
	return json.RawMessage(value), true
}

func geminiFileExtension(filename string) string {
	dotIndex := strings.LastIndex(filename, ".")
	if dotIndex < 0 || dotIndex == len(filename)-1 {
		return ""
	}
	return filename[dotIndex+1:]
}

func geminiCloneObject(source map[string]any) map[string]any {
	cloned := make(map[string]any, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}
