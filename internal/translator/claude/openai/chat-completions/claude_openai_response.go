// Package openai provides response translation functionality for Claude Code to
// OpenAI API compatibility using standard JSON trees.
package chat_completions

import (
	"bytes"
	"context"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

var (
	dataTag = []byte("data:")
)

// ConvertAnthropicResponseToOpenAIParams holds parameters for response
// conversion.
type ConvertAnthropicResponseToOpenAIParams struct {
	CreatedAt    int64
	ResponseID   string
	FinishReason string

	// Tool calls accumulator for streaming.
	ToolCallsAccumulator map[int]*ToolCallAccumulator
}

// ToolCallAccumulator holds the state for accumulating tool call data.
type ToolCallAccumulator struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

// ConvertClaudeResponseToOpenAI converts Claude streaming response format to
// the OpenAI Chat Completions format.
func ConvertClaudeResponseToOpenAI(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	if *param == nil {
		*param = &ConvertAnthropicResponseToOpenAIParams{
			CreatedAt:            0,
			ResponseID:           "",
			FinishReason:         "",
			ToolCallsAccumulator: make(map[int]*ToolCallAccumulator),
		}
	}
	p := (*param).(*ConvertAnthropicResponseToOpenAIParams)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	eventType, _ := jsonutil.String(root, "type")

	switch eventType {
	case "message_start":
		if message, ok := jsonutil.Object(root, "message"); ok {
			if responseID, ok := jsonutil.String(message, "id"); ok {
				p.ResponseID = responseID
			}
			p.CreatedAt = time.Now().Unix()
			if p.ToolCallsAccumulator == nil {
				p.ToolCallsAccumulator = make(map[int]*ToolCallAccumulator)
			}
		}

		outRoot := claudeOpenAIStreamingChunk(p, modelName)
		outRoot["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["role"] = "assistant"
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

	case "content_block_start":
		contentBlock, ok := jsonutil.Object(root, "content_block")
		if !ok {
			return []string{}
		}
		blockType, _ := jsonutil.String(contentBlock, "type")
		if blockType != "tool_use" {
			return []string{}
		}

		index64, _ := jsonutil.Int64(root, "index")
		index := int(index64)
		if p.ToolCallsAccumulator == nil {
			p.ToolCallsAccumulator = make(map[int]*ToolCallAccumulator)
		}
		toolID, _ := jsonutil.String(contentBlock, "id")
		toolName, _ := jsonutil.String(contentBlock, "name")
		p.ToolCallsAccumulator[index] = &ToolCallAccumulator{
			ID:   toolID,
			Name: toolName,
		}
		return []string{}

	case "content_block_delta":
		delta, ok := jsonutil.Object(root, "delta")
		if !ok {
			return []string{}
		}
		deltaType, _ := jsonutil.String(delta, "type")
		switch deltaType {
		case "text_delta":
			text, ok := jsonutil.String(delta, "text")
			if !ok {
				return []string{}
			}
			outRoot := claudeOpenAIStreamingChunk(p, modelName)
			outRoot["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["content"] = text
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

		case "thinking_delta":
			thinking, ok := jsonutil.String(delta, "thinking")
			if !ok {
				return []string{}
			}
			outRoot := claudeOpenAIStreamingChunk(p, modelName)
			outRoot["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["reasoning_content"] = thinking
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

		case "input_json_delta":
			partialJSON, ok := jsonutil.String(delta, "partial_json")
			if !ok {
				return []string{}
			}
			index64, _ := jsonutil.Int64(root, "index")
			if accumulator, ok := p.ToolCallsAccumulator[int(index64)]; ok {
				accumulator.Arguments.WriteString(partialJSON)
			}
			return []string{}
		}
		return []string{}

	case "content_block_stop":
		index64, _ := jsonutil.Int64(root, "index")
		index := int(index64)
		accumulator, ok := p.ToolCallsAccumulator[index]
		if !ok {
			return []string{}
		}

		arguments := accumulator.Arguments.String()
		if arguments == "" {
			arguments = "{}"
		}

		outRoot := claudeOpenAIStreamingChunk(p, modelName)
		delta := outRoot["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
		delta["tool_calls"] = []any{
			map[string]any{
				"index": index,
				"id":    accumulator.ID,
				"type":  "function",
				"function": map[string]any{
					"name":      accumulator.Name,
					"arguments": arguments,
				},
			},
		}

		delete(p.ToolCallsAccumulator, index)
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

	case "message_delta":
		outRoot := claudeOpenAIStreamingChunk(p, modelName)
		choice := outRoot["choices"].([]any)[0].(map[string]any)

		if delta, ok := jsonutil.Object(root, "delta"); ok {
			if stopReason, ok := jsonutil.String(delta, "stop_reason"); ok {
				p.FinishReason = mapAnthropicStopReasonToOpenAI(stopReason)
				choice["finish_reason"] = p.FinishReason
			}
		}
		if usage, ok := jsonutil.Object(root, "usage"); ok {
			outRoot["usage"] = claudeOpenAIUsage(usage)
		}
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

	case "message_stop", "ping":
		return []string{}

	case "error":
		errorData, ok := jsonutil.Object(root, "error")
		if !ok {
			return []string{}
		}
		errorRoot := map[string]any{
			"error": map[string]any{
				"message": "",
				"type":    "",
			},
		}
		if message, ok := jsonutil.String(errorData, "message"); ok {
			errorRoot["error"].(map[string]any)["message"] = message
		}
		if errorType, ok := jsonutil.String(errorData, "type"); ok {
			errorRoot["error"].(map[string]any)["type"] = errorType
		}
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, errorRoot))}

	default:
		return []string{}
	}
}

// mapAnthropicStopReasonToOpenAI maps Anthropic stop reasons to OpenAI stop
// reasons.
func mapAnthropicStopReasonToOpenAI(anthropicReason string) string {
	switch anthropicReason {
	case "end_turn":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	case "stop_sequence":
		return "stop"
	default:
		return "stop"
	}
}

// ConvertClaudeResponseToOpenAINonStream converts a non-streaming Claude Code
// response to a non-streaming OpenAI response.
func ConvertClaudeResponseToOpenAINonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	chunks := make([][]byte, 0)
	for _, line := range bytes.Split(rawJSON, []byte("\n")) {
		if !bytes.HasPrefix(line, dataTag) {
			continue
		}
		chunks = append(chunks, bytes.TrimSpace(line[5:]))
	}

	outRoot := map[string]any{
		"id":      "",
		"object":  "chat.completion",
		"created": int64(0),
		"model":   "",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     int64(0),
			"completion_tokens": int64(0),
			"total_tokens":      int64(0),
		},
	}

	var messageID string
	var model string
	var createdAt int64
	var stopReason string
	contentParts := make([]string, 0)
	reasoningParts := make([]string, 0)
	toolCallsAccumulator := make(map[int]*ToolCallAccumulator)

	for _, chunk := range chunks {
		root := jsonutil.ParseObjectBytesOrEmpty(chunk)
		eventType, _ := jsonutil.String(root, "type")

		switch eventType {
		case "message_start":
			if message, ok := jsonutil.Object(root, "message"); ok {
				messageID, _ = jsonutil.String(message, "id")
				model, _ = jsonutil.String(message, "model")
				createdAt = time.Now().Unix()
			}

		case "content_block_start":
			contentBlock, ok := jsonutil.Object(root, "content_block")
			if !ok {
				continue
			}
			blockType, _ := jsonutil.String(contentBlock, "type")
			if blockType != "tool_use" {
				continue
			}

			index64, _ := jsonutil.Int64(root, "index")
			toolID, _ := jsonutil.String(contentBlock, "id")
			toolName, _ := jsonutil.String(contentBlock, "name")
			toolCallsAccumulator[int(index64)] = &ToolCallAccumulator{
				ID:   toolID,
				Name: toolName,
			}

		case "content_block_delta":
			delta, ok := jsonutil.Object(root, "delta")
			if !ok {
				continue
			}
			deltaType, _ := jsonutil.String(delta, "type")
			switch deltaType {
			case "text_delta":
				if text, ok := jsonutil.String(delta, "text"); ok {
					contentParts = append(contentParts, text)
				}
			case "thinking_delta":
				if thinking, ok := jsonutil.String(delta, "thinking"); ok {
					reasoningParts = append(reasoningParts, thinking)
				}
			case "input_json_delta":
				partialJSON, ok := jsonutil.String(delta, "partial_json")
				if !ok {
					continue
				}
				index64, _ := jsonutil.Int64(root, "index")
				if accumulator, ok := toolCallsAccumulator[int(index64)]; ok {
					accumulator.Arguments.WriteString(partialJSON)
				}
			}

		case "content_block_stop":
			index64, _ := jsonutil.Int64(root, "index")
			if accumulator, ok := toolCallsAccumulator[int(index64)]; ok && accumulator.Arguments.Len() == 0 {
				accumulator.Arguments.WriteString("{}")
			}

		case "message_delta":
			if delta, ok := jsonutil.Object(root, "delta"); ok {
				if sr, ok := jsonutil.String(delta, "stop_reason"); ok {
					stopReason = sr
				}
			}
			if usage, ok := jsonutil.Object(root, "usage"); ok {
				outRoot["usage"] = claudeOpenAIUsage(usage)
			}
		}
	}

	outRoot["id"] = messageID
	outRoot["created"] = createdAt
	outRoot["model"] = model
	outRoot["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["content"] = strings.Join(contentParts, "")

	if len(reasoningParts) > 0 {
		outRoot["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["reasoning"] = strings.Join(reasoningParts, "")
	}

	if len(toolCallsAccumulator) > 0 {
		toolCalls := make([]any, 0, len(toolCallsAccumulator))
		maxIndex := -1
		for index := range toolCallsAccumulator {
			if index > maxIndex {
				maxIndex = index
			}
		}
		for index := 0; index <= maxIndex; index++ {
			accumulator, ok := toolCallsAccumulator[index]
			if !ok {
				continue
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   accumulator.ID,
				"type": "function",
				"function": map[string]any{
					"name":      accumulator.Name,
					"arguments": accumulator.Arguments.String(),
				},
			})
		}
		if len(toolCalls) > 0 {
			outRoot["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)["tool_calls"] = toolCalls
			outRoot["choices"].([]any)[0].(map[string]any)["finish_reason"] = "tool_calls"
		} else {
			outRoot["choices"].([]any)[0].(map[string]any)["finish_reason"] = mapAnthropicStopReasonToOpenAI(stopReason)
		}
	} else {
		outRoot["choices"].([]any)[0].(map[string]any)["finish_reason"] = mapAnthropicStopReasonToOpenAI(stopReason)
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func claudeOpenAIStreamingChunk(param *ConvertAnthropicResponseToOpenAIParams, modelName string) map[string]any {
	outRoot := map[string]any{
		"id":      "",
		"object":  "chat.completion.chunk",
		"created": int64(0),
		"model":   "",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{},
			},
		},
	}
	if modelName != "" {
		outRoot["model"] = modelName
	}
	if param.ResponseID != "" {
		outRoot["id"] = param.ResponseID
	}
	if param.CreatedAt > 0 {
		outRoot["created"] = param.CreatedAt
	}
	return outRoot
}

func claudeOpenAIUsage(usage map[string]any) map[string]any {
	inputTokens, _ := jsonutil.Int64(usage, "input_tokens")
	outputTokens, _ := jsonutil.Int64(usage, "output_tokens")
	cacheReadInputTokens, _ := jsonutil.Int64(usage, "cache_read_input_tokens")
	cacheCreationInputTokens, _ := jsonutil.Int64(usage, "cache_creation_input_tokens")

	usageRoot := map[string]any{
		"prompt_tokens":     inputTokens + cacheCreationInputTokens,
		"completion_tokens": outputTokens,
		"total_tokens":      inputTokens + outputTokens,
	}
	usageRoot["prompt_tokens_details"] = map[string]any{
		"cached_tokens": cacheReadInputTokens,
	}
	return usageRoot
}
