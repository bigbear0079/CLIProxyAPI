package responses

import (
	"encoding/json"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
)

const geminiResponsesThoughtSignature = "skip_thought_signature_validator"

var openAIResponsesAudioMimeTypes = map[string]string{
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

func ConvertOpenAIResponsesRequestToGemini(modelName string, inputRawJSON []byte, stream bool) []byte {
	_ = modelName
	_ = stream

	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"contents":       []any{},
		"safetySettings": common.DefaultSafetySettings(),
	}

	systemParts := make([]any, 0)
	if instructions, ok := jsonutil.String(root, "instructions"); ok {
		systemParts = append(systemParts, map[string]any{"text": instructions})
	}

	contents := make([]any, 0)

	if inputItems, ok := jsonutil.Array(root, "input"); ok {
		normalizedItems := normalizeGeminiResponsesInput(inputItems)
		functionNameByCallID := collectGeminiResponsesFunctionNames(inputItems)

		for _, itemValue := range normalizedItems {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}

			itemType := geminiResponsesItemType(item)
			itemRole, _ := jsonutil.String(item, "role")

			switch itemType {
			case "message":
				if strings.EqualFold(itemRole, "system") {
					systemParts = append(systemParts, geminiResponsesSystemParts(item)...)
					continue
				}

				messageContents := geminiResponsesMessageContents(item, itemRole)
				contents = append(contents, messageContents...)
			case "function_call":
				functionCallContent, ok := buildGeminiResponsesFunctionCallContent(item)
				if ok {
					contents = append(contents, functionCallContent)
				}
			case "function_call_output":
				functionResponseContent, ok := buildGeminiResponsesFunctionResponseContent(item, functionNameByCallID)
				if ok {
					contents = append(contents, functionResponseContent)
				}
			case "reasoning":
				if thoughtContent, ok := buildGeminiResponsesReasoningContent(item); ok {
					contents = append(contents, thoughtContent)
				}
			}
		}
	} else if inputText, ok := jsonutil.String(root, "input"); ok {
		contents = append(contents, map[string]any{
			"role": "user",
			"parts": []any{
				map[string]any{"text": inputText},
			},
		})
	}

	if len(systemParts) > 0 {
		outRoot["systemInstruction"] = map[string]any{
			"parts": systemParts,
		}
	}
	if len(contents) > 0 {
		outRoot["contents"] = contents
	}

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		functionDeclarations := make([]any, 0)
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			toolType, _ := jsonutil.String(tool, "type")
			if toolType != "function" {
				continue
			}

			functionDeclaration := map[string]any{}
			if name, ok := jsonutil.String(tool, "name"); ok {
				functionDeclaration["name"] = name
			}
			if description, ok := jsonutil.String(tool, "description"); ok {
				functionDeclaration["description"] = description
			}
			if parameters, ok := jsonutil.Get(tool, "parameters"); ok {
				functionDeclaration["parametersJsonSchema"] = parameters
			} else if parameters, ok := jsonutil.Get(tool, "parametersJsonSchema"); ok {
				functionDeclaration["parametersJsonSchema"] = parameters
			}
			functionDeclarations = append(functionDeclarations, functionDeclaration)
		}
		if len(functionDeclarations) > 0 {
			outRoot["tools"] = []any{
				map[string]any{
					"functionDeclarations": functionDeclarations,
				},
			}
		}
	}

	if maxOutputTokens, ok := jsonutil.Get(root, "max_output_tokens"); ok {
		_ = jsonutil.Set(outRoot, "generationConfig.maxOutputTokens", maxOutputTokens)
	}
	if temperature, ok := jsonutil.Get(root, "temperature"); ok {
		_ = jsonutil.Set(outRoot, "generationConfig.temperature", temperature)
	}
	if topP, ok := jsonutil.Get(root, "top_p"); ok {
		_ = jsonutil.Set(outRoot, "generationConfig.topP", topP)
	}
	if stopSequences, ok := jsonutil.Array(root, "stop_sequences"); ok {
		sequences := make([]string, 0, len(stopSequences))
		for _, seqValue := range stopSequences {
			if sequence, ok := seqValue.(string); ok {
				sequences = append(sequences, sequence)
			}
		}
		if len(sequences) > 0 {
			_ = jsonutil.Set(outRoot, "generationConfig.stopSequences", sequences)
		}
	}

	if reasoningEffort, ok := jsonutil.String(root, "reasoning.effort"); ok {
		effort := strings.ToLower(strings.TrimSpace(reasoningEffort))
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

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func normalizeGeminiResponsesInput(items []any) []any {
	normalized := make([]any, 0, len(items))

	for index := 0; index < len(items); {
		item, ok := items[index].(map[string]any)
		if !ok {
			index++
			continue
		}

		itemType := geminiResponsesItemType(item)
		if itemType != "function_call" {
			normalized = append(normalized, item)
			index++
			continue
		}

		calls := make([]map[string]any, 0)
		outputs := make([]map[string]any, 0)

		for index < len(items) {
			next, ok := items[index].(map[string]any)
			if !ok {
				break
			}
			if geminiResponsesItemType(next) != "function_call" {
				break
			}
			calls = append(calls, next)
			index++
		}

		for index < len(items) {
			next, ok := items[index].(map[string]any)
			if !ok {
				break
			}
			if geminiResponsesItemType(next) != "function_call_output" {
				break
			}
			outputs = append(outputs, next)
			index++
		}

		if len(calls) == 0 {
			continue
		}

		outputByCallID := make(map[string]map[string]any, len(outputs))
		for _, output := range outputs {
			callID, _ := jsonutil.String(output, "call_id")
			if callID != "" {
				outputByCallID[callID] = output
			}
		}

		for _, call := range calls {
			normalized = append(normalized, call)
			callID, _ := jsonutil.String(call, "call_id")
			if output, ok := outputByCallID[callID]; ok {
				normalized = append(normalized, output)
				delete(outputByCallID, callID)
			}
		}

		for _, output := range outputs {
			callID, _ := jsonutil.String(output, "call_id")
			if _, ok := outputByCallID[callID]; ok {
				normalized = append(normalized, output)
			}
		}
	}

	return normalized
}

func collectGeminiResponsesFunctionNames(items []any) map[string]string {
	functionNameByCallID := make(map[string]string)
	for _, itemValue := range items {
		item, ok := itemValue.(map[string]any)
		if !ok {
			continue
		}
		if geminiResponsesItemType(item) != "function_call" {
			continue
		}
		callID, _ := jsonutil.String(item, "call_id")
		name, _ := jsonutil.String(item, "name")
		if callID != "" && name != "" {
			functionNameByCallID[callID] = name
		}
	}
	return functionNameByCallID
}

func geminiResponsesItemType(item map[string]any) string {
	itemType, _ := jsonutil.String(item, "type")
	if itemType == "" {
		if role, ok := jsonutil.String(item, "role"); ok && role != "" {
			return "message"
		}
	}
	return itemType
}

func geminiResponsesSystemParts(item map[string]any) []any {
	contentValue, ok := jsonutil.Get(item, "content")
	if !ok {
		return nil
	}

	parts := make([]any, 0)
	switch typed := contentValue.(type) {
	case []any:
		for _, contentValue := range typed {
			contentItem, ok := contentValue.(map[string]any)
			if !ok {
				continue
			}
			text, _ := jsonutil.String(contentItem, "text")
			parts = append(parts, map[string]any{"text": text})
		}
	case string:
		parts = append(parts, map[string]any{"text": typed})
	}

	return parts
}

func geminiResponsesMessageContents(item map[string]any, itemRole string) []any {
	contentValue, ok := jsonutil.Get(item, "content")
	if !ok {
		return nil
	}

	switch typed := contentValue.(type) {
	case []any:
		currentRole := ""
		currentParts := make([]any, 0)
		contents := make([]any, 0)

		flush := func() {
			if currentRole == "" || len(currentParts) == 0 {
				currentParts = nil
				return
			}
			contents = append(contents, map[string]any{
				"role":  currentRole,
				"parts": currentParts,
			})
			currentParts = nil
		}

		for _, contentValue := range typed {
			contentItem, ok := contentValue.(map[string]any)
			if !ok {
				continue
			}

			contentType, _ := jsonutil.String(contentItem, "type")
			if contentType == "" {
				contentType = "input_text"
			}
			role := geminiResponsesEffectiveRole(itemRole, contentType)

			part, ok := buildGeminiResponsesMessagePart(contentItem, contentType)
			if !ok {
				continue
			}

			if currentRole != "" && currentRole != role {
				flush()
				currentRole = ""
			}
			if currentRole == "" {
				currentRole = role
			}
			currentParts = append(currentParts, part)
		}

		flush()
		return contents
	case string:
		return []any{
			map[string]any{
				"role": geminiResponsesEffectiveRole(itemRole, ""),
				"parts": []any{
					map[string]any{"text": typed},
				},
			},
		}
	default:
		return nil
	}
}

func geminiResponsesEffectiveRole(itemRole, contentType string) string {
	role := "user"
	switch strings.ToLower(strings.TrimSpace(itemRole)) {
	case "assistant", "model":
		role = "model"
	case "", "user":
		role = "user"
	default:
		role = strings.ToLower(strings.TrimSpace(itemRole))
	}

	if contentType == "output_text" {
		role = "model"
	}
	if role == "assistant" {
		role = "model"
	}
	return role
}

func buildGeminiResponsesMessagePart(contentItem map[string]any, contentType string) (map[string]any, bool) {
	switch contentType {
	case "input_text", "output_text":
		text, ok := jsonutil.String(contentItem, "text")
		if !ok {
			return nil, false
		}
		return map[string]any{"text": text}, true
	case "input_image":
		imageURL := ""
		if value, ok := jsonutil.String(contentItem, "image_url"); ok {
			imageURL = value
		}
		if imageURL == "" {
			imageURL, _ = jsonutil.String(contentItem, "url")
		}
		mimeType, data, ok := parseGeminiResponsesDataURL(imageURL)
		if !ok {
			return nil, false
		}
		return map[string]any{
			"inline_data": map[string]any{
				"mime_type": mimeType,
				"data":      data,
			},
		}, true
	case "input_audio":
		audioData, _ := jsonutil.String(contentItem, "data")
		if audioData == "" {
			return nil, false
		}
		audioFormat, _ := jsonutil.String(contentItem, "format")
		return map[string]any{
			"inline_data": map[string]any{
				"mime_type": geminiResponsesAudioMimeType(audioFormat),
				"data":      audioData,
			},
		}, true
	default:
		return nil, false
	}
}

func parseGeminiResponsesDataURL(imageURL string) (string, string, bool) {
	if !strings.HasPrefix(imageURL, "data:") {
		return "", "", false
	}

	trimmed := strings.TrimPrefix(imageURL, "data:")
	mediaAndData := strings.SplitN(trimmed, ";base64,", 2)
	if len(mediaAndData) == 2 {
		mimeType := "application/octet-stream"
		if mediaAndData[0] != "" {
			mimeType = mediaAndData[0]
		}
		return mimeType, mediaAndData[1], mediaAndData[1] != ""
	}

	mediaAndData = strings.SplitN(trimmed, ",", 2)
	if len(mediaAndData) == 2 {
		mimeType := "application/octet-stream"
		if mediaAndData[0] != "" {
			mimeType = mediaAndData[0]
		}
		return mimeType, mediaAndData[1], mediaAndData[1] != ""
	}

	return "", "", false
}

func buildGeminiResponsesFunctionCallContent(item map[string]any) (map[string]any, bool) {
	name, _ := jsonutil.String(item, "name")
	callID, _ := jsonutil.String(item, "call_id")
	arguments, _ := jsonutil.String(item, "arguments")

	functionCall := map[string]any{
		"name": name,
		"id":   callID,
		"args": map[string]any{},
	}
	if arguments != "" && json.Valid([]byte(arguments)) {
		if parsedArguments, errParse := jsonutil.ParseAnyBytes([]byte(arguments)); errParse == nil {
			functionCall["args"] = parsedArguments
		}
	}

	return map[string]any{
		"role": "model",
		"parts": []any{
			map[string]any{
				"functionCall":     functionCall,
				"thoughtSignature": geminiResponsesThoughtSignature,
			},
		},
	}, true
}

func buildGeminiResponsesFunctionResponseContent(item map[string]any, functionNameByCallID map[string]string) (map[string]any, bool) {
	callID, _ := jsonutil.String(item, "call_id")
	if callID == "" {
		return nil, false
	}

	functionName := functionNameByCallID[callID]
	if functionName == "" {
		functionName = "unknown"
	}

	functionResponse := map[string]any{
		"name":     functionName,
		"id":       callID,
		"response": map[string]any{},
	}

	if outputValue, ok := jsonutil.Get(item, "output"); ok {
		switch typed := outputValue.(type) {
		case string:
			if typed != "" && typed != "null" {
				if json.Valid([]byte(typed)) {
					if parsedOutput, errParse := jsonutil.ParseAnyBytes([]byte(typed)); errParse == nil {
						functionResponse["response"].(map[string]any)["result"] = parsedOutput
					} else {
						functionResponse["response"].(map[string]any)["result"] = typed
					}
				} else {
					functionResponse["response"].(map[string]any)["result"] = typed
				}
			}
		case nil:
			// Keep empty response object.
		default:
			functionResponse["response"].(map[string]any)["result"] = typed
		}
	}

	return map[string]any{
		"role": "function",
		"parts": []any{
			map[string]any{
				"functionResponse": functionResponse,
			},
		},
	}, true
}

func buildGeminiResponsesReasoningContent(item map[string]any) (map[string]any, bool) {
	text := ""
	if summaryItems, ok := jsonutil.Array(item, "summary"); ok && len(summaryItems) > 0 {
		if summaryItem, ok := summaryItems[0].(map[string]any); ok {
			text, _ = jsonutil.String(summaryItem, "text")
		}
	}
	thoughtSignature, _ := jsonutil.String(item, "encrypted_content")

	return map[string]any{
		"role": "model",
		"parts": []any{
			map[string]any{
				"text":             text,
				"thoughtSignature": thoughtSignature,
				"thought":          true,
			},
		},
	}, true
}

func geminiResponsesAudioMimeType(audioFormat string) string {
	audioFormat = strings.TrimSpace(audioFormat)
	if audioFormat == "" {
		return "audio/wav"
	}
	if mimeType, ok := openAIResponsesAudioMimeTypes[audioFormat]; ok {
		return mimeType
	}
	return "audio/" + audioFormat
}
