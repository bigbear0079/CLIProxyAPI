// Package openai provides request translation functionality for OpenAI to
// Antigravity API compatibility using standard JSON trees.
package chat_completions

import (
	"encoding/json"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	log "github.com/sirupsen/logrus"
)

const geminiCLIFunctionThoughtSignature = "skip_thought_signature_validator"

var openAIInputAudioMimeTypes = map[string]string{
	"mp3":       "audio/mpeg",
	"wav":       "audio/wav",
	"ogg":       "audio/ogg",
	"flac":      "audio/flac",
	"aac":       "audio/aac",
	"webm":      "audio/webm",
	"pcm16":     "audio/pcm",
	"g711_ulaw": "audio/basic",
	"g711_alaw": "audio/basic",
}

// ConvertOpenAIRequestToAntigravity converts an OpenAI Chat Completions request
// into an Antigravity request payload.
func ConvertOpenAIRequestToAntigravity(modelName string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	requestRoot := map[string]any{
		"contents":       []any{},
		"safetySettings": common.DefaultSafetySettings(),
	}
	outRoot := map[string]any{
		"project": "",
		"request": requestRoot,
		"model":   modelName,
	}

	if generationConfig, ok := jsonutil.Get(root, "generationConfig"); ok {
		requestRoot["generationConfig"] = generationConfig
	}

	if effort, ok := jsonutil.String(root, "reasoning_effort"); ok {
		effort = strings.ToLower(strings.TrimSpace(effort))
		if effort != "" {
			if effort == "auto" {
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.thinkingBudget", -1)
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.includeThoughts", true)
			} else {
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.thinkingLevel", effort)
				_ = jsonutil.Set(outRoot, "request.generationConfig.thinkingConfig.includeThoughts", effort != "none")
			}
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
	if value, ok := jsonutil.Int64(root, "n"); ok && value > 1 {
		_ = jsonutil.Set(outRoot, "request.generationConfig.candidateCount", value)
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
			_ = jsonutil.Set(outRoot, "request.generationConfig.responseModalities", responseModalities)
		}
	}

	if imageConfig, ok := jsonutil.Object(root, "image_config"); ok {
		if value, ok := jsonutil.Get(imageConfig, "aspect_ratio"); ok {
			_ = jsonutil.Set(outRoot, "request.generationConfig.imageConfig.aspectRatio", value)
		}
		if value, ok := jsonutil.Get(imageConfig, "image_size"); ok {
			_ = jsonutil.Set(outRoot, "request.generationConfig.imageConfig.imageSize", value)
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
		requestContents := make([]any, 0, len(messages))
		messageCount := len(messages)
		for _, messageValue := range messages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}
			role, _ := jsonutil.String(message, "role")
			contentValue, _ := jsonutil.Get(message, "content")

			if (role == "system" || role == "developer") && messageCount > 1 {
				systemParts = append(systemParts, buildSystemInstructionParts(contentValue)...)
				continue
			}

			if role == "user" || ((role == "system" || role == "developer") && messageCount == 1) {
				requestContents = append(requestContents, map[string]any{
					"role":  "user",
					"parts": buildUserParts(contentValue),
				})
				continue
			}

			if role != "assistant" {
				continue
			}

			node, functionIDs := buildAssistantNode(message, contentValue)
			requestContents = append(requestContents, node)

			if toolNode, ok := buildToolResponseNode(functionIDs, toolNameByID, toolResponses); ok {
				requestContents = append(requestContents, toolNode)
			}
		}

		if len(systemParts) > 0 {
			requestRoot["systemInstruction"] = map[string]any{
				"role":  "user",
				"parts": systemParts,
			}
		}
		requestRoot["contents"] = requestContents
	}

	if tools, ok := jsonutil.Array(root, "tools"); ok && len(tools) > 0 {
		requestTools := buildAntigravityTools(tools)
		if len(requestTools) > 0 {
			requestRoot["tools"] = requestTools
		}
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func buildSystemInstructionParts(contentValue any) []any {
	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case string:
		if typed != "" {
			parts = append(parts, map[string]any{"text": typed})
		}
	case map[string]any:
		if part, ok := buildTextPart(typed); ok {
			parts = append(parts, part)
		}
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if part, ok := buildTextPart(item); ok {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func buildUserParts(contentValue any) []any {
	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case string:
		if typed != "" {
			parts = append(parts, map[string]any{"text": typed})
		}
	case map[string]any:
		if part, ok := buildUserPart(typed); ok {
			parts = append(parts, part)
		}
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if part, ok := buildUserPart(item); ok {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func buildUserPart(item map[string]any) (map[string]any, bool) {
	itemType, _ := jsonutil.String(item, "type")
	switch itemType {
	case "", "text":
		return buildTextPart(item)
	case "image_url":
		imageURL := ""
		if imageURLValue, ok := jsonutil.Object(item, "image_url"); ok {
			imageURL, _ = jsonutil.String(imageURLValue, "url")
		}
		if imageURL == "" {
			imageURL, _ = jsonutil.String(item, "image_url")
		}
		return buildInlineImagePart(imageURL)
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
		ext := strings.ToLower(fileExtension(filename))
		mimeType, ok := misc.MimeTypes[ext]
		if !ok {
			log.Warnf("Unknown file name extension '%s' in user message, skip", ext)
			return nil, false
		}
		return map[string]any{
			"inlineData": map[string]any{
				"mimeType": mimeType,
				"data":     fileData,
			},
		}, true
	case "input_audio":
		inputAudio, ok := jsonutil.Object(item, "input_audio")
		if !ok {
			return nil, false
		}
		audioData, _ := jsonutil.String(inputAudio, "data")
		if audioData == "" {
			return nil, false
		}
		audioFormat, _ := jsonutil.String(inputAudio, "format")
		return map[string]any{
			"inlineData": map[string]any{
				"mime_type": buildInputAudioMimeType(audioFormat),
				"data":      audioData,
			},
		}, true
	default:
		return nil, false
	}
}

func buildAssistantNode(message map[string]any, contentValue any) (map[string]any, []string) {
	parts := buildAssistantParts(contentValue)
	functionIDs := make([]string, 0)

	if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
		for _, toolCallValue := range toolCalls {
			toolCall, ok := toolCallValue.(map[string]any)
			if !ok {
				continue
			}
			part, functionID, ok := buildFunctionCallPart(toolCall)
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

func buildAssistantParts(contentValue any) []any {
	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case string:
		if typed != "" {
			parts = append(parts, map[string]any{"text": typed})
		}
	case map[string]any:
		if part, ok := buildAssistantPart(typed); ok {
			parts = append(parts, part)
		}
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if part, ok := buildAssistantPart(item); ok {
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func buildAssistantPart(item map[string]any) (map[string]any, bool) {
	itemType, _ := jsonutil.String(item, "type")
	switch itemType {
	case "", "text":
		return buildTextPart(item)
	case "image_url":
		imageURL := ""
		if imageURLValue, ok := jsonutil.Object(item, "image_url"); ok {
			imageURL, _ = jsonutil.String(imageURLValue, "url")
		}
		if imageURL == "" {
			imageURL, _ = jsonutil.String(item, "image_url")
		}
		return buildInlineImagePart(imageURL)
	default:
		return nil, false
	}
}

func buildTextPart(item map[string]any) (map[string]any, bool) {
	text, _ := jsonutil.String(item, "text")
	if text == "" {
		return nil, false
	}
	return map[string]any{"text": text}, true
}

func buildInlineImagePart(imageURL string) (map[string]any, bool) {
	mimeType, data, ok := parseDataURL(imageURL)
	if !ok {
		return nil, false
	}
	return map[string]any{
		"inlineData": map[string]any{
			"mimeType": mimeType,
			"data":     data,
		},
		"thoughtSignature": geminiCLIFunctionThoughtSignature,
	}, true
}

func parseDataURL(imageURL string) (string, string, bool) {
	if !strings.HasPrefix(imageURL, "data:") {
		return "", "", false
	}
	pieces := strings.SplitN(imageURL[5:], ";", 2)
	if len(pieces) != 2 || !strings.HasPrefix(pieces[1], "base64,") || len(pieces[1]) <= len("base64,") {
		return "", "", false
	}
	return pieces[0], pieces[1][len("base64,"):], true
}

func buildFunctionCallPart(toolCall map[string]any) (map[string]any, string, bool) {
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
	if rawArguments, ok := rawJSON(arguments); ok {
		argsValue = rawArguments
	} else if arguments != "" {
		argsValue = map[string]any{"params": arguments}
	}

	functionID, _ := jsonutil.String(toolCall, "id")
	functionName, _ := jsonutil.String(function, "name")
	return map[string]any{
		"functionCall": map[string]any{
			"id":   functionID,
			"name": functionName,
			"args": argsValue,
		},
		"thoughtSignature": geminiCLIFunctionThoughtSignature,
	}, functionID, true
}

func buildToolResponseNode(functionIDs []string, toolNameByID map[string]string, toolResponses map[string]any) (map[string]any, bool) {
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
				if rawValue, ok := rawJSON(typed); ok {
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
				"id":       functionID,
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

func buildAntigravityTools(tools []any) []any {
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
				functionDeclaration := cloneObject(function)
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

	requestTools := make([]any, 0, 1+len(googleSearchTools)+len(codeExecutionTools)+len(urlContextTools))
	if len(functionDeclarations) > 0 {
		requestTools = append(requestTools, map[string]any{
			"functionDeclarations": functionDeclarations,
		})
	}
	requestTools = append(requestTools, googleSearchTools...)
	requestTools = append(requestTools, codeExecutionTools...)
	requestTools = append(requestTools, urlContextTools...)
	return requestTools
}

func rawJSON(value string) (json.RawMessage, bool) {
	value = strings.TrimSpace(value)
	if value == "" || !json.Valid([]byte(value)) {
		return nil, false
	}
	return json.RawMessage(value), true
}

func buildInputAudioMimeType(format string) string {
	format = strings.TrimSpace(format)
	if format == "" {
		return "audio/wav"
	}
	if mimeType, ok := openAIInputAudioMimeTypes[format]; ok {
		return mimeType
	}
	return "audio/" + format
}

func fileExtension(filename string) string {
	dotIndex := strings.LastIndex(filename, ".")
	if dotIndex < 0 || dotIndex == len(filename)-1 {
		return ""
	}
	return filename[dotIndex+1:]
}

func cloneObject(source map[string]any) map[string]any {
	cloned := make(map[string]any, len(source))
	for key, value := range source {
		cloned[key] = value
	}
	return cloned
}
