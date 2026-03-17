// Package openai provides response translation functionality for Codex to
// OpenAI API compatibility using standard JSON trees.
package chat_completions

import (
	"bytes"
	"context"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

var (
	dataTag = []byte("data:")
)

// ConvertCliToOpenAIParams holds parameters for response conversion.
type ConvertCliToOpenAIParams struct {
	ResponseID                string
	CreatedAt                 int64
	Model                     string
	FunctionCallIndex         int
	HasReceivedArgumentsDelta bool
	HasToolCallAnnounced      bool
}

// ConvertCodexResponseToOpenAI translates a single chunk of a streaming
// response from the Codex API format to the OpenAI Chat Completions streaming
// format.
func ConvertCodexResponseToOpenAI(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &ConvertCliToOpenAIParams{
			ResponseID:                "",
			CreatedAt:                 0,
			Model:                     modelName,
			FunctionCallIndex:         -1,
			HasReceivedArgumentsDelta: false,
			HasToolCallAnnounced:      false,
		}
	}
	p := (*param).(*ConvertCliToOpenAIParams)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	dataType, _ := jsonutil.String(root, "type")
	if dataType == "response.created" {
		if responseID, ok := jsonutil.String(root, "response.id"); ok {
			p.ResponseID = responseID
		}
		if createdAt, ok := jsonutil.Int64(root, "response.created_at"); ok {
			p.CreatedAt = createdAt
		}
		if model, ok := jsonutil.String(root, "response.model"); ok {
			p.Model = model
		}
		return []string{}
	}

	outRoot := codexOpenAIBaseChunk(p, root, modelName)
	choice := outRoot["choices"].([]any)[0].(map[string]any)
	delta := choice["delta"].(map[string]any)
	reverseNames := buildReverseMapFromOriginalOpenAI(originalRequestRawJSON)

	if responseRoot, ok := jsonutil.Object(root, "response"); ok {
		if usage, ok := jsonutil.Object(responseRoot, "usage"); ok {
			codexOpenAIApplyUsage(outRoot, usage)
		}
	}

	switch dataType {
	case "response.reasoning_summary_text.delta":
		delta["role"] = "assistant"
		delta["reasoning_content"] = codexOpenAIString(root["delta"])

	case "response.reasoning_summary_text.done":
		delta["role"] = "assistant"
		delta["reasoning_content"] = "\n\n"

	case "response.output_text.delta":
		delta["role"] = "assistant"
		delta["content"] = codexOpenAIString(root["delta"])

	case "response.completed":
		finishReason := "stop"
		if p.FunctionCallIndex != -1 {
			finishReason = "tool_calls"
		}
		choice["finish_reason"] = finishReason
		choice["native_finish_reason"] = finishReason

	case "response.output_item.added":
		item, ok := jsonutil.Object(root, "item")
		if !ok {
			return []string{}
		}
		itemType, _ := jsonutil.String(item, "type")
		if itemType != "function_call" {
			return []string{}
		}

		p.FunctionCallIndex++
		p.HasReceivedArgumentsDelta = false
		p.HasToolCallAnnounced = true

		delta["role"] = "assistant"
		delta["tool_calls"] = []any{
			codexOpenAIToolCall(item, p.FunctionCallIndex, reverseNames, ""),
		}

	case "response.function_call_arguments.delta":
		p.HasReceivedArgumentsDelta = true
		delta["tool_calls"] = []any{
			map[string]any{
				"index": p.FunctionCallIndex,
				"function": map[string]any{
					"arguments": codexOpenAIString(root["delta"]),
				},
			},
		}

	case "response.function_call_arguments.done":
		if p.HasReceivedArgumentsDelta {
			return []string{}
		}
		arguments, _ := jsonutil.String(root, "arguments")
		delta["tool_calls"] = []any{
			map[string]any{
				"index": p.FunctionCallIndex,
				"function": map[string]any{
					"arguments": arguments,
				},
			},
		}

	case "response.output_item.done":
		item, ok := jsonutil.Object(root, "item")
		if !ok {
			return []string{}
		}
		itemType, _ := jsonutil.String(item, "type")
		if itemType != "function_call" {
			return []string{}
		}
		if p.HasToolCallAnnounced {
			p.HasToolCallAnnounced = false
			return []string{}
		}

		p.FunctionCallIndex++
		delta["role"] = "assistant"
		arguments, _ := jsonutil.String(item, "arguments")
		delta["tool_calls"] = []any{
			codexOpenAIToolCall(item, p.FunctionCallIndex, reverseNames, arguments),
		}

	default:
		return []string{}
	}

	return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}
}

// ConvertCodexResponseToOpenAINonStream converts a non-streaming Codex response
// to a non-streaming OpenAI response.
func ConvertCodexResponseToOpenAINonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	if responseType, _ := jsonutil.String(root, "type"); responseType != "response.completed" {
		return ""
	}

	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return ""
	}

	createdAt, ok := jsonutil.Int64(responseRoot, "created_at")
	if !ok {
		createdAt = time.Now().Unix()
	}

	outRoot := map[string]any{
		"id":      "",
		"object":  "chat.completion",
		"created": createdAt,
		"model":   "model",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":              "assistant",
					"content":           nil,
					"reasoning_content": nil,
					"tool_calls":        nil,
				},
				"finish_reason":        nil,
				"native_finish_reason": nil,
			},
		},
	}
	if model, ok := jsonutil.String(responseRoot, "model"); ok {
		outRoot["model"] = model
	}
	if responseID, ok := jsonutil.String(responseRoot, "id"); ok {
		outRoot["id"] = responseID
	}
	if usage, ok := jsonutil.Object(responseRoot, "usage"); ok {
		codexOpenAIApplyUsage(outRoot, usage)
	}

	message := outRoot["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	reverseNames := buildReverseMapFromOriginalOpenAI(originalRequestRawJSON)
	toolCalls := make([]any, 0)
	contentText := ""
	reasoningText := ""

	if outputItems, ok := jsonutil.Array(responseRoot, "output"); ok {
		for _, itemValue := range outputItems {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}
			itemType, _ := jsonutil.String(item, "type")
			switch itemType {
			case "reasoning":
				reasoningText = codexOpenAINonStreamReasoning(item)

			case "message":
				if content, ok := jsonutil.Array(item, "content"); ok {
					for _, contentValue := range content {
						part, ok := contentValue.(map[string]any)
						if !ok {
							continue
						}
						if partType, _ := jsonutil.String(part, "type"); partType == "output_text" {
							if text, ok := jsonutil.String(part, "text"); ok {
								contentText = text
								break
							}
						}
					}
				}

			case "function_call":
				arguments, _ := jsonutil.String(item, "arguments")
				toolCalls = append(toolCalls, codexOpenAIToolCall(item, 0, reverseNames, arguments))
			}
		}
	}

	if contentText != "" {
		message["content"] = contentText
	}
	if reasoningText != "" {
		message["reasoning_content"] = reasoningText
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	if status, ok := jsonutil.String(responseRoot, "status"); ok && status == "completed" {
		outRoot["choices"].([]any)[0].(map[string]any)["finish_reason"] = "stop"
		outRoot["choices"].([]any)[0].(map[string]any)["native_finish_reason"] = "stop"
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func codexOpenAIBaseChunk(param *ConvertCliToOpenAIParams, root map[string]any, modelName string) map[string]any {
	model := modelName
	if modelValue, ok := jsonutil.String(root, "model"); ok && modelValue != "" {
		model = modelValue
	} else if param.Model != "" {
		model = param.Model
	}

	return map[string]any{
		"id":      param.ResponseID,
		"object":  "chat.completion.chunk",
		"created": param.CreatedAt,
		"model":   model,
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"role":              nil,
					"content":           nil,
					"reasoning_content": nil,
					"tool_calls":        nil,
				},
				"finish_reason":        nil,
				"native_finish_reason": nil,
			},
		},
	}
}

func codexOpenAIApplyUsage(root map[string]any, usage map[string]any) {
	usageOut := map[string]any{}
	if outputTokens, ok := jsonutil.Int64(usage, "output_tokens"); ok {
		usageOut["completion_tokens"] = outputTokens
	}
	if totalTokens, ok := jsonutil.Int64(usage, "total_tokens"); ok {
		usageOut["total_tokens"] = totalTokens
	}
	if inputTokens, ok := jsonutil.Int64(usage, "input_tokens"); ok {
		usageOut["prompt_tokens"] = inputTokens
	}
	if cachedTokens, ok := jsonutil.Int64(usage, "input_tokens_details.cached_tokens"); ok {
		details, _ := usageOut["prompt_tokens_details"].(map[string]any)
		if details == nil {
			details = map[string]any{}
			usageOut["prompt_tokens_details"] = details
		}
		details["cached_tokens"] = cachedTokens
	}
	if reasoningTokens, ok := jsonutil.Int64(usage, "output_tokens_details.reasoning_tokens"); ok {
		details, _ := usageOut["completion_tokens_details"].(map[string]any)
		if details == nil {
			details = map[string]any{}
			usageOut["completion_tokens_details"] = details
		}
		details["reasoning_tokens"] = reasoningTokens
	}
	root["usage"] = usageOut
}

func codexOpenAIToolCall(item map[string]any, index int, reverseNames map[string]string, arguments string) map[string]any {
	name, _ := jsonutil.String(item, "name")
	if originalName, ok := reverseNames[name]; ok {
		name = originalName
	}
	callID, _ := jsonutil.String(item, "call_id")
	if callID == "" {
		callID, _ = jsonutil.String(item, "id")
	}

	return map[string]any{
		"index": index,
		"id":    callID,
		"type":  "function",
		"function": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
}

func codexOpenAINonStreamReasoning(item map[string]any) string {
	if summary, ok := jsonutil.Array(item, "summary"); ok {
		for _, partValue := range summary {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}
			if partType, _ := jsonutil.String(part, "type"); partType == "summary_text" {
				if text, ok := jsonutil.String(part, "text"); ok {
					return text
				}
			}
		}
	}
	return ""
}

func buildReverseMapFromOriginalOpenAI(original []byte) map[string]string {
	root := jsonutil.ParseObjectBytesOrEmpty(original)
	reverseMap := map[string]string{}
	names := make([]string, 0)

	if tools, ok := jsonutil.Array(root, "tools"); ok {
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
	}

	if len(names) > 0 {
		shortMap := buildShortNameMap(names)
		for originalName, shortName := range shortMap {
			reverseMap[shortName] = originalName
		}
	}
	return reverseMap
}

func codexOpenAIString(value any) string {
	if value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return ""
	}
}
