// Package openai provides utilities to translate OpenAI Chat Completions
// request JSON into OpenAI Responses API request JSON using standard JSON trees.
// It supports tools, multimodal text/image inputs, and Structured Outputs.
// The package handles the conversion of OpenAI API requests into the format
// expected by the OpenAI Responses API, including proper mapping of messages,
// tools, and generation parameters.
package chat_completions

import (
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertOpenAIRequestToCodex converts an OpenAI Chat Completions request JSON
// into an OpenAI Responses API request JSON. The transformation follows the
// examples defined in docs/2.md exactly, including tools, multi-turn dialog,
// multimodal text/image handling, and Structured Outputs mapping.
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the OpenAI Chat Completions API
//   - stream: A boolean indicating if the request is for a streaming response
//
// Returns:
//   - []byte: The transformed request data in OpenAI Responses API format
func ConvertOpenAIRequestToCodex(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	outRoot := map[string]any{
		"instructions":        "",
		"stream":              stream,
		"parallel_tool_calls": true,
		"include":             []string{"reasoning.encrypted_content"},
		"model":               modelName,
		"store":               false,
		"input":               []any{},
		"reasoning":           map[string]any{"effort": "medium", "summary": "auto"},
	}

	if effort, ok := jsonutil.Get(root, "reasoning_effort"); ok {
		outRoot["reasoning"].(map[string]any)["effort"] = effort
	}

	originalToolNameMap := map[string]string{}
	if tools, ok := jsonutil.Array(root, "tools"); ok && len(tools) > 0 {
		names := make([]string, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			toolType, _ := jsonutil.String(tool, "type")
			if toolType != "function" {
				continue
			}
			function, ok := jsonutil.Object(tool, "function")
			if !ok {
				continue
			}
			if name, ok := jsonutil.String(function, "name"); ok && name != "" {
				names = append(names, name)
			}
		}
		if len(names) > 0 {
			originalToolNameMap = buildShortNameMap(names)
		}
	}

	inputItems := make([]any, 0)
	if messages, ok := jsonutil.Array(root, "messages"); ok {
		for _, messageValue := range messages {
			message, ok := messageValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(message, "role")
			switch role {
			case "tool":
				toolCallID, _ := jsonutil.String(message, "tool_call_id")
				content, _ := jsonutil.String(message, "content")
				inputItems = append(inputItems, map[string]any{
					"type":    "function_call_output",
					"call_id": toolCallID,
					"output":  content,
				})

			default:
				messageItem := map[string]any{
					"type":    "message",
					"role":    role,
					"content": []any{},
				}
				if role == "system" {
					messageItem["role"] = "developer"
				}

				contentParts := make([]any, 0)
				if content, ok := jsonutil.String(message, "content"); ok && content != "" {
					partType := "input_text"
					if role == "assistant" {
						partType = "output_text"
					}
					contentParts = append(contentParts, map[string]any{
						"type": partType,
						"text": content,
					})
				} else if contentArray, ok := jsonutil.Array(message, "content"); ok {
					for _, itemValue := range contentArray {
						item, ok := itemValue.(map[string]any)
						if !ok {
							continue
						}
						itemType, _ := jsonutil.String(item, "type")
						switch itemType {
						case "text":
							partType := "input_text"
							if role == "assistant" {
								partType = "output_text"
							}
							text, _ := jsonutil.String(item, "text")
							contentParts = append(contentParts, map[string]any{
								"type": partType,
								"text": text,
							})
						case "image_url":
							if role != "user" {
								continue
							}
							imageURL := ""
							if imageURLValue, ok := jsonutil.Object(item, "image_url"); ok {
								imageURL, _ = jsonutil.String(imageURLValue, "url")
							}
							if imageURL == "" {
								imageURL, _ = jsonutil.String(item, "image_url")
							}
							if imageURL == "" {
								continue
							}
							contentParts = append(contentParts, map[string]any{
								"type":      "input_image",
								"image_url": imageURL,
							})
						case "file":
							if role != "user" {
								continue
							}
							fileObject, ok := jsonutil.Object(item, "file")
							if !ok {
								continue
							}
							fileData, _ := jsonutil.String(fileObject, "file_data")
							if fileData == "" {
								continue
							}
							part := map[string]any{
								"type":      "input_file",
								"file_data": fileData,
							}
							if filename, ok := jsonutil.String(fileObject, "filename"); ok && filename != "" {
								part["filename"] = filename
							}
							contentParts = append(contentParts, part)
						}
					}
				}

				messageItem["content"] = contentParts
				if role != "assistant" || len(contentParts) > 0 {
					inputItems = append(inputItems, messageItem)
				}

				if role == "assistant" {
					if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
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
							name, _ := jsonutil.String(function, "name")
							if short, ok := originalToolNameMap[name]; ok {
								name = short
							} else {
								name = shortenNameIfNeeded(name)
							}
							arguments, _ := jsonutil.String(function, "arguments")
							callID, _ := jsonutil.String(toolCall, "id")

							inputItems = append(inputItems, map[string]any{
								"type":      "function_call",
								"call_id":   callID,
								"name":      name,
								"arguments": arguments,
							})
						}
					}
				}
			}
		}
	}
	outRoot["input"] = inputItems

	if responseFormat, ok := jsonutil.Object(root, "response_format"); ok {
		textRoot := map[string]any{}
		responseFormatType, _ := jsonutil.String(responseFormat, "type")
		switch responseFormatType {
		case "text":
			textRoot["format"] = map[string]any{"type": "text"}
		case "json_schema":
			if jsonSchema, ok := jsonutil.Object(responseFormat, "json_schema"); ok {
				format := map[string]any{"type": "json_schema"}
				if value, ok := jsonutil.Get(jsonSchema, "name"); ok {
					format["name"] = value
				}
				if value, ok := jsonutil.Get(jsonSchema, "strict"); ok {
					format["strict"] = value
				}
				if value, ok := jsonutil.Get(jsonSchema, "schema"); ok {
					format["schema"] = value
				}
				textRoot["format"] = format
			}
		}
		if textConfig, ok := jsonutil.Object(root, "text"); ok {
			if value, ok := jsonutil.Get(textConfig, "verbosity"); ok {
				textRoot["verbosity"] = value
			}
		}
		if len(textRoot) > 0 {
			outRoot["text"] = textRoot
		}
	} else if textConfig, ok := jsonutil.Object(root, "text"); ok {
		if value, ok := jsonutil.Get(textConfig, "verbosity"); ok {
			outRoot["text"] = map[string]any{"verbosity": value}
		}
	}

	if tools, ok := jsonutil.Array(root, "tools"); ok && len(tools) > 0 {
		outTools := make([]any, 0, len(tools))
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			toolType, _ := jsonutil.String(tool, "type")
			if toolType != "" && toolType != "function" {
				outTools = append(outTools, tool)
				continue
			}
			if toolType != "function" {
				continue
			}
			function, ok := jsonutil.Object(tool, "function")
			if !ok {
				continue
			}
			item := map[string]any{
				"type": "function",
			}
			if name, ok := jsonutil.String(function, "name"); ok && name != "" {
				if short, ok := originalToolNameMap[name]; ok {
					name = short
				} else {
					name = shortenNameIfNeeded(name)
				}
				item["name"] = name
			}
			if value, ok := jsonutil.Get(function, "description"); ok {
				item["description"] = value
			}
			if value, ok := jsonutil.Get(function, "parameters"); ok {
				item["parameters"] = value
			}
			if value, ok := jsonutil.Get(function, "strict"); ok {
				item["strict"] = value
			}
			outTools = append(outTools, item)
		}
		if len(outTools) > 0 {
			outRoot["tools"] = outTools
		}
	}

	if toolChoiceValue, ok := jsonutil.Get(root, "tool_choice"); ok {
		switch typed := toolChoiceValue.(type) {
		case string:
			outRoot["tool_choice"] = typed
		case map[string]any:
			toolChoiceType, _ := jsonutil.String(typed, "type")
			if toolChoiceType == "function" {
				choice := map[string]any{
					"type": "function",
				}
				if function, ok := jsonutil.Object(typed, "function"); ok {
					if name, ok := jsonutil.String(function, "name"); ok && name != "" {
						if short, ok := originalToolNameMap[name]; ok {
							name = short
						} else {
							name = shortenNameIfNeeded(name)
						}
						choice["name"] = name
					}
				}
				outRoot["tool_choice"] = choice
			} else if toolChoiceType != "" {
				outRoot["tool_choice"] = typed
			}
		}
	}

	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

// shortenNameIfNeeded applies the simple shortening rule for a single name.
// If the name length exceeds 64, it will try to preserve the "mcp__" prefix and last segment.
// Otherwise it truncates to 64 characters.
func shortenNameIfNeeded(name string) string {
	const limit = 64
	if len(name) <= limit {
		return name
	}
	if strings.HasPrefix(name, "mcp__") {
		// Keep prefix and last segment after '__'
		idx := strings.LastIndex(name, "__")
		if idx > 0 {
			candidate := "mcp__" + name[idx+2:]
			if len(candidate) > limit {
				return candidate[:limit]
			}
			return candidate
		}
	}
	return name[:limit]
}

// buildShortNameMap generates unique short names (<=64) for the given list of names.
// It preserves the "mcp__" prefix with the last segment when possible and ensures uniqueness
// by appending suffixes like "~1", "~2" if needed.
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
