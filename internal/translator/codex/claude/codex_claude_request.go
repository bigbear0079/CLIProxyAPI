// Package claude provides request translation functionality for Claude Code API compatibility.
// It handles parsing and transforming Claude Code API requests into the internal client format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package also performs JSON data cleaning and transformation to ensure compatibility
// between Claude Code API format and the internal client's expected format.
package claude

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
)

// ConvertClaudeRequestToCodex parses and transforms a Claude Code API request into the internal client format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the internal client.
// The function performs the following transformations:
// 1. Sets up a template with the model name and empty instructions field
// 2. Processes system messages and converts them to developer input content
// 3. Transforms message contents (text, image, tool_use, tool_result) to appropriate formats
// 4. Converts tools declarations to the expected format
// 5. Adds additional configuration parameters for the Codex API
// 6. Maps Claude thinking configuration to Codex reasoning settings
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the Claude Code API
//   - stream: A boolean indicating if the request is for a streaming response (unused in current implementation)
//
// Returns:
//   - []byte: The transformed request data in internal client format
func ConvertClaudeRequestToCodex(modelName string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)
	outRoot := map[string]any{
		"model":        modelName,
		"instructions": "",
		"input":        []any{},
	}

	shortMap := claudeCodexShortMap(root)
	inputItems := make([]any, 0)

	if systemValue, ok := jsonutil.Get(root, "system"); ok {
		if systemMessage, ok := claudeCodexSystemMessage(systemValue); ok {
			inputItems = append(inputItems, systemMessage)
		}
	}

	if messages, ok := jsonutil.Array(root, "messages"); ok {
		for _, messageValue := range messages {
			messageObject, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}

			messageRole, _ := jsonutil.String(messageObject, "role")
			contentParts := make([]any, 0)

			flushMessage := func() {
				if len(contentParts) == 0 {
					return
				}
				inputItems = append(inputItems, map[string]any{
					"type":    "message",
					"role":    messageRole,
					"content": contentParts,
				})
				contentParts = make([]any, 0)
			}

			appendTextContent := func(text string) {
				partType := "input_text"
				if messageRole == "assistant" {
					partType = "output_text"
				}
				contentParts = append(contentParts, map[string]any{
					"type": partType,
					"text": text,
				})
			}

			appendImageContent := func(dataURL string) {
				contentParts = append(contentParts, map[string]any{
					"type":      "input_image",
					"image_url": dataURL,
				})
			}

			if messageContents, ok := jsonutil.Array(messageObject, "content"); ok {
				for _, contentValue := range messageContents {
					content, ok := contentValue.(map[string]any)
					if !ok {
						continue
					}

					contentType, _ := jsonutil.String(content, "type")
					switch contentType {
					case "text":
						if text, ok := jsonutil.String(content, "text"); ok {
							appendTextContent(text)
						}
					case "image":
						if dataURL, ok := claudeCodexImageDataURL(content); ok {
							appendImageContent(dataURL)
						}
					case "tool_use":
						flushMessage()
						functionCall := map[string]any{
							"type":    "function_call",
							"call_id": claudeCodexContentString(content["id"]),
						}
						if name, ok := jsonutil.String(content, "name"); ok {
							if short, ok := shortMap[name]; ok {
								functionCall["name"] = short
							} else {
								functionCall["name"] = shortenNameIfNeeded(name)
							}
						}
						if inputValue, ok := jsonutil.Get(content, "input"); ok {
							functionCall["arguments"] = claudeCodexJSONText(inputValue)
						}
						inputItems = append(inputItems, functionCall)
					case "tool_result":
						flushMessage()
						functionOutput := map[string]any{
							"type":    "function_call_output",
							"call_id": claudeCodexContentString(content["tool_use_id"]),
						}

						if resultValue, ok := jsonutil.Get(content, "content"); ok {
							if resultArray, ok := resultValue.([]any); ok {
								if outputParts := claudeCodexToolResultOutput(resultArray); len(outputParts) > 0 {
									functionOutput["output"] = outputParts
								} else {
									functionOutput["output"] = claudeCodexContentString(resultValue)
								}
							} else {
								functionOutput["output"] = claudeCodexContentString(resultValue)
							}
						} else {
							functionOutput["output"] = ""
						}
						inputItems = append(inputItems, functionOutput)
					}
				}
				flushMessage()
				continue
			}

			if contentText, ok := jsonutil.String(messageObject, "content"); ok {
				appendTextContent(contentText)
				flushMessage()
			}
		}
	}
	outRoot["input"] = inputItems

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		codexTools := make([]any, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			if toolType, _ := jsonutil.String(tool, "type"); toolType == "web_search_20250305" {
				codexTools = append(codexTools, map[string]any{"type": "web_search"})
				continue
			}

			codexTool := map[string]any{
				"type":       "function",
				"parameters": normalizeToolParametersValue(tool["input_schema"]),
				"strict":     false,
			}
			if name, ok := jsonutil.String(tool, "name"); ok {
				if short, ok := shortMap[name]; ok {
					codexTool["name"] = short
				} else {
					codexTool["name"] = shortenNameIfNeeded(name)
				}
			}
			if description, ok := jsonutil.String(tool, "description"); ok {
				codexTool["description"] = description
			}
			codexTools = append(codexTools, codexTool)
		}
		outRoot["tools"] = codexTools
		outRoot["tool_choice"] = "auto"
	}

	parallelToolCalls := true
	if disableParallelToolUse, ok := jsonutil.Bool(root, "tool_choice.disable_parallel_tool_use"); ok {
		parallelToolCalls = !disableParallelToolUse
	}
	outRoot["parallel_tool_calls"] = parallelToolCalls

	reasoningEffort := "medium"
	if thinkingConfig, ok := jsonutil.Object(root, "thinking"); ok {
		switch thinkingType, _ := jsonutil.String(thinkingConfig, "type"); thinkingType {
		case "enabled":
			if budgetTokens, ok := jsonutil.Int64(thinkingConfig, "budget_tokens"); ok {
				if effort, ok := thinking.ConvertBudgetToLevel(int(budgetTokens)); ok && effort != "" {
					reasoningEffort = effort
				}
			}
		case "adaptive", "auto":
			if effort, ok := jsonutil.String(root, "output_config.effort"); ok {
				effort = strings.ToLower(strings.TrimSpace(effort))
				if effort != "" {
					reasoningEffort = effort
				} else {
					reasoningEffort = string(thinking.LevelXHigh)
				}
			} else {
				reasoningEffort = string(thinking.LevelXHigh)
			}
		case "disabled":
			if effort, ok := thinking.ConvertBudgetToLevel(0); ok && effort != "" {
				reasoningEffort = effort
			}
		}
	}

	outRoot["reasoning"] = map[string]any{
		"effort":  reasoningEffort,
		"summary": "auto",
	}
	outRoot["stream"] = true
	outRoot["store"] = false
	outRoot["include"] = []string{"reasoning.encrypted_content"}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

func claudeCodexShortMap(root map[string]any) map[string]string {
	names := make([]string, 0)
	if tools, ok := jsonutil.Array(root, "tools"); ok {
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			if name, ok := jsonutil.String(tool, "name"); ok && name != "" {
				names = append(names, name)
			}
		}
	}
	if len(names) == 0 {
		return map[string]string{}
	}
	return buildShortNameMap(names)
}

func claudeCodexSystemMessage(systemValue any) (map[string]any, bool) {
	contentParts := make([]any, 0)
	appendText := func(text string) {
		if text == "" || strings.HasPrefix(text, "x-anthropic-billing-header: ") {
			return
		}
		contentParts = append(contentParts, map[string]any{
			"type": "input_text",
			"text": text,
		})
	}

	switch typed := systemValue.(type) {
	case string:
		appendText(typed)
	case []any:
		for _, itemValue := range typed {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			if itemType, _ := jsonutil.String(item, "type"); itemType == "text" {
				if text, ok := jsonutil.String(item, "text"); ok {
					appendText(text)
				}
			}
		}
	}

	if len(contentParts) == 0 {
		return nil, false
	}
	return map[string]any{
		"type":    "message",
		"role":    "developer",
		"content": contentParts,
	}, true
}

func claudeCodexImageDataURL(content map[string]any) (string, bool) {
	source, ok := jsonutil.Object(content, "source")
	if !ok {
		return "", false
	}

	data, _ := jsonutil.String(source, "data")
	if data == "" {
		data, _ = jsonutil.String(source, "base64")
	}
	if data == "" {
		return "", false
	}

	mediaType, _ := jsonutil.String(source, "media_type")
	if mediaType == "" {
		mediaType, _ = jsonutil.String(source, "mime_type")
	}
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}

	return fmt.Sprintf("data:%s;base64,%s", mediaType, data), true
}

func claudeCodexToolResultOutput(content []any) []any {
	outputParts := make([]any, 0)
	for _, itemValue := range content {
		item, ok := itemValue.(map[string]any)
		if !ok {
			continue
		}

		itemType, _ := jsonutil.String(item, "type")
		switch itemType {
		case "image":
			if dataURL, ok := claudeCodexImageDataURL(item); ok {
				outputParts = append(outputParts, map[string]any{
					"type":      "input_image",
					"image_url": dataURL,
				})
			}
		case "text":
			if text, ok := jsonutil.String(item, "text"); ok {
				outputParts = append(outputParts, map[string]any{
					"type": "input_text",
					"text": text,
				})
			}
		}
	}
	return outputParts
}

func claudeCodexJSONText(value any) string {
	return string(jsonutil.MarshalOrOriginal(nil, value))
}

func claudeCodexContentString(value any) string {
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
		return string(jsonutil.MarshalOrOriginal(nil, typed))
	}
}

// shortenNameIfNeeded applies a simple shortening rule for a single name.
func shortenNameIfNeeded(name string) string {
	const limit = 64
	if len(name) <= limit {
		return name
	}
	if strings.HasPrefix(name, "mcp__") {
		idx := strings.LastIndex(name, "__")
		if idx > 0 {
			cand := "mcp__" + name[idx+2:]
			if len(cand) > limit {
				return cand[:limit]
			}
			return cand
		}
	}
	return name[:limit]
}

// buildShortNameMap ensures uniqueness of shortened names within a request.
func buildShortNameMap(names []string) map[string]string {
	const limit = 64
	used := map[string]struct{}{}
	m := map[string]string{}

	baseCandidate := func(n string) string {
		if len(n) <= limit {
			return n
		}
		if strings.HasPrefix(n, "mcp__") {
			idx := strings.LastIndex(n, "__")
			if idx > 0 {
				cand := "mcp__" + n[idx+2:]
				if len(cand) > limit {
					cand = cand[:limit]
				}
				return cand
			}
		}
		return n[:limit]
	}

	makeUnique := func(cand string) string {
		if _, ok := used[cand]; !ok {
			return cand
		}
		base := cand
		for i := 1; ; i++ {
			suffix := "_" + strconv.Itoa(i)
			allowed := limit - len(suffix)
			if allowed < 0 {
				allowed = 0
			}
			tmp := base
			if len(tmp) > allowed {
				tmp = tmp[:allowed]
			}
			tmp = tmp + suffix
			if _, ok := used[tmp]; !ok {
				return tmp
			}
		}
	}

	for _, n := range names {
		cand := baseCandidate(n)
		uniq := makeUnique(cand)
		used[uniq] = struct{}{}
		m[n] = uniq
	}
	return m
}

// buildReverseMapFromClaudeOriginalToShort builds original->short map, used to map tool_use names to short.
func buildReverseMapFromClaudeOriginalToShort(original []byte) map[string]string {
	root := jsonutil.ParseObjectBytesOrEmpty(original)
	return claudeCodexShortMap(root)
}

// normalizeToolParameters ensures object schemas contain at least an empty properties map.
func normalizeToolParameters(raw string) string {
	root, errParse := jsonutil.ParseAnyBytes([]byte(raw))
	if errParse != nil {
		return string(jsonutil.MarshalOrOriginal(nil, normalizeToolParametersValue(nil)))
	}
	return string(jsonutil.MarshalOrOriginal(nil, normalizeToolParametersValue(root)))
}

func normalizeToolParametersValue(value any) map[string]any {
	schema, ok := value.(map[string]any)
	if !ok {
		return map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}
	}

	delete(schema, "$schema")
	schemaType, _ := schema["type"].(string)
	if schemaType == "" {
		schemaType = "object"
		schema["type"] = schemaType
	}
	if strings.EqualFold(schemaType, "object") {
		if _, ok := schema["properties"]; !ok {
			schema["properties"] = map[string]any{}
		}
	}
	return schema
}
