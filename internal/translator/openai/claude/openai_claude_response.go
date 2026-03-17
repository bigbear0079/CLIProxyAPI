// Package claude provides response translation functionality for OpenAI to
// Anthropic compatibility using standard JSON trees.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

var (
	dataTag = []byte("data:")
)

// ConvertOpenAIResponseToAnthropicParams holds parameters for response
// conversion.
type ConvertOpenAIResponseToAnthropicParams struct {
	MessageID   string
	Model       string
	CreatedAt   int64
	ToolNameMap map[string]string
	SawToolCall bool

	ContentAccumulator   strings.Builder
	ToolCallsAccumulator map[int]*ToolCallAccumulator

	TextContentBlockStarted     bool
	ThinkingContentBlockStarted bool
	FinishReason                string
	ContentBlocksStopped        bool
	MessageDeltaSent            bool
	MessageStarted              bool
	MessageStopSent             bool
	ToolCallBlockIndexes        map[int]int
	TextContentBlockIndex       int
	ThinkingContentBlockIndex   int
	NextContentBlockIndex       int
}

// ToolCallAccumulator holds the state for accumulating tool call data.
type ToolCallAccumulator struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

// ConvertOpenAIResponseToClaude converts OpenAI streaming response format to
// Anthropic API format.
func ConvertOpenAIResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = requestRawJSON

	if *param == nil {
		*param = &ConvertOpenAIResponseToAnthropicParams{
			MessageID:                   "",
			Model:                       "",
			CreatedAt:                   0,
			ToolNameMap:                 nil,
			SawToolCall:                 false,
			ContentAccumulator:          strings.Builder{},
			ToolCallsAccumulator:        nil,
			TextContentBlockStarted:     false,
			ThinkingContentBlockStarted: false,
			FinishReason:                "",
			ContentBlocksStopped:        false,
			MessageDeltaSent:            false,
			MessageStarted:              false,
			MessageStopSent:             false,
			ToolCallBlockIndexes:        make(map[int]int),
			TextContentBlockIndex:       -1,
			ThinkingContentBlockIndex:   -1,
			NextContentBlockIndex:       0,
		}
	}
	p := (*param).(*ConvertOpenAIResponseToAnthropicParams)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	if p.ToolNameMap == nil {
		p.ToolNameMap = util.ToolNameMapFromClaudeRequest(originalRequestRawJSON)
	}

	rawStr := strings.TrimSpace(string(rawJSON))
	if rawStr == "[DONE]" {
		return convertOpenAIDoneToAnthropic(p)
	}

	if !isStreamingOpenAIRequest(originalRequestRawJSON) {
		return convertOpenAINonStreamingToAnthropic(rawJSON)
	}
	return convertOpenAIStreamingChunkToAnthropic(rawJSON, p)
}

func effectiveOpenAIFinishReason(param *ConvertOpenAIResponseToAnthropicParams) string {
	if param == nil {
		return ""
	}
	if param.SawToolCall {
		return "tool_calls"
	}
	return param.FinishReason
}

func convertOpenAIStreamingChunkToAnthropic(rawJSON []byte, param *ConvertOpenAIResponseToAnthropicParams) []string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	results := make([]string, 0)

	if param.MessageID == "" {
		if messageID, ok := jsonutil.String(root, "id"); ok {
			param.MessageID = messageID
		}
	}
	if param.Model == "" {
		if model, ok := jsonutil.String(root, "model"); ok {
			param.Model = model
		}
	}
	if param.CreatedAt == 0 {
		if createdAt, ok := jsonutil.Int64(root, "created"); ok {
			param.CreatedAt = createdAt
		}
	}

	if delta, ok := jsonutil.Object(root, "choices.0.delta"); ok {
		if !param.MessageStarted {
			results = append(results, anthropicSSEEvent("message_start", map[string]any{
				"type": "message_start",
				"message": map[string]any{
					"id":            param.MessageID,
					"type":          "message",
					"role":          "assistant",
					"model":         param.Model,
					"content":       []any{},
					"stop_reason":   nil,
					"stop_sequence": nil,
					"usage": map[string]any{
						"input_tokens":  0,
						"output_tokens": 0,
					},
				},
			}))
			param.MessageStarted = true
		}

		if reasoningValue, ok := jsonutil.Get(delta, "reasoning_content"); ok {
			for _, reasoningText := range collectOpenAIReasoningTexts(reasoningValue) {
				if reasoningText == "" {
					continue
				}
				stopTextContentBlock(param, &results)
				if !param.ThinkingContentBlockStarted {
					if param.ThinkingContentBlockIndex == -1 {
						param.ThinkingContentBlockIndex = param.NextContentBlockIndex
						param.NextContentBlockIndex++
					}
					results = append(results, anthropicSSEEvent("content_block_start", map[string]any{
						"type":  "content_block_start",
						"index": param.ThinkingContentBlockIndex,
						"content_block": map[string]any{
							"type":     "thinking",
							"thinking": "",
						},
					}))
					param.ThinkingContentBlockStarted = true
				}

				results = append(results, anthropicSSEEvent("content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": param.ThinkingContentBlockIndex,
					"delta": map[string]any{
						"type":     "thinking_delta",
						"thinking": reasoningText,
					},
				}))
			}
		}

		if content, ok := jsonutil.String(delta, "content"); ok && content != "" {
			if !param.TextContentBlockStarted {
				stopThinkingContentBlock(param, &results)
				if param.TextContentBlockIndex == -1 {
					param.TextContentBlockIndex = param.NextContentBlockIndex
					param.NextContentBlockIndex++
				}
				results = append(results, anthropicSSEEvent("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": param.TextContentBlockIndex,
					"content_block": map[string]any{
						"type": "text",
						"text": "",
					},
				}))
				param.TextContentBlockStarted = true
			}

			results = append(results, anthropicSSEEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": param.TextContentBlockIndex,
				"delta": map[string]any{
					"type": "text_delta",
					"text": content,
				},
			}))
			param.ContentAccumulator.WriteString(content)
		}

		if toolCalls, ok := jsonutil.Array(delta, "tool_calls"); ok {
			if param.ToolCallsAccumulator == nil {
				param.ToolCallsAccumulator = make(map[int]*ToolCallAccumulator)
			}

			for _, toolCallValue := range toolCalls {
				toolCall, ok := toolCallValue.(map[string]any)
				if !ok {
					continue
				}

				param.SawToolCall = true
				index64, _ := jsonutil.Int64(toolCall, "index")
				index := int(index64)
				blockIndex := param.toolContentBlockIndex(index)

				if _, exists := param.ToolCallsAccumulator[index]; !exists {
					param.ToolCallsAccumulator[index] = &ToolCallAccumulator{}
				}
				accumulator := param.ToolCallsAccumulator[index]

				if toolID, ok := jsonutil.String(toolCall, "id"); ok {
					accumulator.ID = toolID
				}

				function, hasFunction := jsonutil.Object(toolCall, "function")
				if !hasFunction {
					continue
				}

				if name, ok := jsonutil.String(function, "name"); ok {
					accumulator.Name = util.MapToolName(param.ToolNameMap, name)
					stopThinkingContentBlock(param, &results)
					stopTextContentBlock(param, &results)

					results = append(results, anthropicSSEEvent("content_block_start", map[string]any{
						"type":  "content_block_start",
						"index": blockIndex,
						"content_block": map[string]any{
							"type":  "tool_use",
							"id":    util.SanitizeClaudeToolID(accumulator.ID),
							"name":  accumulator.Name,
							"input": map[string]any{},
						},
					}))
				}

				if args, ok := jsonutil.String(function, "arguments"); ok && args != "" {
					accumulator.Arguments.WriteString(args)
				}
			}
		}
	}

	if finishReason, ok := jsonutil.String(root, "choices.0.finish_reason"); ok && finishReason != "" {
		if param.SawToolCall {
			param.FinishReason = "tool_calls"
		} else {
			param.FinishReason = finishReason
		}

		if param.ThinkingContentBlockStarted {
			results = append(results, anthropicSSEEvent("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": param.ThinkingContentBlockIndex,
			}))
			param.ThinkingContentBlockStarted = false
			param.ThinkingContentBlockIndex = -1
		}

		stopTextContentBlock(param, &results)

		if !param.ContentBlocksStopped {
			for index, accumulator := range param.ToolCallsAccumulator {
				blockIndex := param.toolContentBlockIndex(index)
				if accumulator.Arguments.Len() > 0 {
					results = append(results, anthropicSSEEvent("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": blockIndex,
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": util.FixJSON(accumulator.Arguments.String()),
						},
					}))
				}
				results = append(results, anthropicSSEEvent("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": blockIndex,
				}))
				delete(param.ToolCallBlockIndexes, index)
			}
			param.ContentBlocksStopped = true
		}
	}

	if param.FinishReason != "" {
		if usage, ok := jsonutil.Object(root, "usage"); ok {
			inputTokens, outputTokens, cachedTokens := extractOpenAIUsage(usage)
			messageDelta := map[string]any{
				"type": "message_delta",
				"delta": map[string]any{
					"stop_reason":   mapOpenAIFinishReasonToAnthropic(effectiveOpenAIFinishReason(param)),
					"stop_sequence": nil,
				},
				"usage": map[string]any{
					"input_tokens":  inputTokens,
					"output_tokens": outputTokens,
				},
			}
			if cachedTokens > 0 {
				messageDelta["usage"].(map[string]any)["cache_read_input_tokens"] = cachedTokens
			}
			results = append(results, anthropicSSEEvent("message_delta", messageDelta))
			param.MessageDeltaSent = true
			emitMessageStopIfNeeded(param, &results)
		}
	}

	return results
}

func convertOpenAIDoneToAnthropic(param *ConvertOpenAIResponseToAnthropicParams) []string {
	results := make([]string, 0)

	if param.ThinkingContentBlockStarted {
		results = append(results, anthropicSSEEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": param.ThinkingContentBlockIndex,
		}))
		param.ThinkingContentBlockStarted = false
		param.ThinkingContentBlockIndex = -1
	}

	stopTextContentBlock(param, &results)

	if !param.ContentBlocksStopped {
		for index, accumulator := range param.ToolCallsAccumulator {
			blockIndex := param.toolContentBlockIndex(index)
			if accumulator.Arguments.Len() > 0 {
				results = append(results, anthropicSSEEvent("content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": blockIndex,
					"delta": map[string]any{
						"type":         "input_json_delta",
						"partial_json": util.FixJSON(accumulator.Arguments.String()),
					},
				}))
			}
			results = append(results, anthropicSSEEvent("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": blockIndex,
			}))
			delete(param.ToolCallBlockIndexes, index)
		}
		param.ContentBlocksStopped = true
	}

	if param.FinishReason != "" && !param.MessageDeltaSent {
		results = append(results, anthropicSSEEvent("message_delta", map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   mapOpenAIFinishReasonToAnthropic(effectiveOpenAIFinishReason(param)),
				"stop_sequence": nil,
			},
			"usage": map[string]any{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		}))
		param.MessageDeltaSent = true
	}

	emitMessageStopIfNeeded(param, &results)
	return results
}

func convertOpenAINonStreamingToAnthropic(rawJSON []byte) []string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)

	outRoot := map[string]any{
		"id":            "",
		"type":          "message",
		"role":          "assistant",
		"model":         "",
		"content":       []any{},
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  0,
			"output_tokens": 0,
		},
	}
	if id, ok := jsonutil.String(root, "id"); ok {
		outRoot["id"] = id
	}
	if model, ok := jsonutil.String(root, "model"); ok {
		outRoot["model"] = model
	}

	if choices, ok := jsonutil.Array(root, "choices"); ok && len(choices) > 0 {
		choice, _ := choices[0].(map[string]any)

		if message, ok := jsonutil.Object(choice, "message"); ok {
			if reasoningValue, ok := jsonutil.Get(message, "reasoning_content"); ok {
				for _, reasoningText := range collectOpenAIReasoningTexts(reasoningValue) {
					if reasoningText == "" {
						continue
					}
					outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
						"type":     "thinking",
						"thinking": reasoningText,
					})
				}
			}

			if content, ok := jsonutil.String(message, "content"); ok && content != "" {
				outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
					"type": "text",
					"text": content,
				})
			}

			if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
				for _, toolCallValue := range toolCalls {
					toolCall, ok := toolCallValue.(map[string]any)
					if !ok {
						continue
					}
					function, ok := jsonutil.Object(toolCall, "function")
					if !ok {
						continue
					}
					toolUse := map[string]any{
						"type":  "tool_use",
						"id":    util.SanitizeClaudeToolID(openAIClaudeString(toolCall["id"])),
						"name":  openAIClaudeString(function["name"]),
						"input": openAIClaudeArgumentsToObject(openAIClaudeString(function["arguments"])),
					}
					outRoot["content"] = append(outRoot["content"].([]any), toolUse)
				}
			}
		}

		if finishReason, ok := jsonutil.String(choice, "finish_reason"); ok {
			outRoot["stop_reason"] = mapOpenAIFinishReasonToAnthropic(finishReason)
		}
	}

	if usage, ok := jsonutil.Object(root, "usage"); ok {
		inputTokens, outputTokens, cachedTokens := extractOpenAIUsage(usage)
		outRoot["usage"].(map[string]any)["input_tokens"] = inputTokens
		outRoot["usage"].(map[string]any)["output_tokens"] = outputTokens
		if cachedTokens > 0 {
			outRoot["usage"].(map[string]any)["cache_read_input_tokens"] = cachedTokens
		}
	}

	return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}
}

// mapOpenAIFinishReasonToAnthropic maps OpenAI finish reasons to Anthropic
// equivalents.
func mapOpenAIFinishReasonToAnthropic(openAIReason string) string {
	switch openAIReason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "end_turn"
	case "function_call":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func (p *ConvertOpenAIResponseToAnthropicParams) toolContentBlockIndex(openAIToolIndex int) int {
	if idx, ok := p.ToolCallBlockIndexes[openAIToolIndex]; ok {
		return idx
	}
	idx := p.NextContentBlockIndex
	p.NextContentBlockIndex++
	p.ToolCallBlockIndexes[openAIToolIndex] = idx
	return idx
}

func collectOpenAIReasoningTexts(node any) []string {
	texts := make([]string, 0)
	if node == nil {
		return texts
	}

	switch typed := node.(type) {
	case string:
		if typed != "" {
			texts = append(texts, typed)
		}
	case []any:
		for _, value := range typed {
			texts = append(texts, collectOpenAIReasoningTexts(value)...)
		}
	case map[string]any:
		if text, ok := jsonutil.String(typed, "text"); ok && text != "" {
			texts = append(texts, text)
		}
	}

	return texts
}

func stopThinkingContentBlock(param *ConvertOpenAIResponseToAnthropicParams, results *[]string) {
	if !param.ThinkingContentBlockStarted {
		return
	}
	*results = append(*results, anthropicSSEEvent("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": param.ThinkingContentBlockIndex,
	}))
	param.ThinkingContentBlockStarted = false
	param.ThinkingContentBlockIndex = -1
}

func emitMessageStopIfNeeded(param *ConvertOpenAIResponseToAnthropicParams, results *[]string) {
	if param.MessageStopSent {
		return
	}
	*results = append(*results, anthropicSSEEvent("message_stop", map[string]any{
		"type": "message_stop",
	}))
	param.MessageStopSent = true
}

func stopTextContentBlock(param *ConvertOpenAIResponseToAnthropicParams, results *[]string) {
	if !param.TextContentBlockStarted {
		return
	}
	*results = append(*results, anthropicSSEEvent("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": param.TextContentBlockIndex,
	}))
	param.TextContentBlockStarted = false
	param.TextContentBlockIndex = -1
}

// ConvertOpenAIResponseToClaudeNonStream converts a non-streaming OpenAI
// response to a non-streaming Anthropic response.
func ConvertOpenAIResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = requestRawJSON

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	toolNameMap := util.ToolNameMapFromClaudeRequest(originalRequestRawJSON)

	outRoot := map[string]any{
		"id":            "",
		"type":          "message",
		"role":          "assistant",
		"model":         "",
		"content":       []any{},
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  0,
			"output_tokens": 0,
		},
	}
	if id, ok := jsonutil.String(root, "id"); ok {
		outRoot["id"] = id
	}
	if model, ok := jsonutil.String(root, "model"); ok {
		outRoot["model"] = model
	}

	hasToolCall := false
	stopReasonSet := false

	if choices, ok := jsonutil.Array(root, "choices"); ok && len(choices) > 0 {
		choice, _ := choices[0].(map[string]any)

		if finishReason, ok := jsonutil.String(choice, "finish_reason"); ok {
			outRoot["stop_reason"] = mapOpenAIFinishReasonToAnthropic(finishReason)
			stopReasonSet = true
		}

		if message, ok := jsonutil.Object(choice, "message"); ok {
			if contentValue, ok := jsonutil.Get(message, "content"); ok {
				switch typed := contentValue.(type) {
				case []any:
					textBuilder := strings.Builder{}
					thinkingBuilder := strings.Builder{}

					flushText := func() {
						if textBuilder.Len() == 0 {
							return
						}
						outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
							"type": "text",
							"text": textBuilder.String(),
						})
						textBuilder.Reset()
					}
					flushThinking := func() {
						if thinkingBuilder.Len() == 0 {
							return
						}
						outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
							"type":     "thinking",
							"thinking": thinkingBuilder.String(),
						})
						thinkingBuilder.Reset()
					}

					for _, itemValue := range typed {
						item, ok := itemValue.(map[string]any)
						if !ok {
							flushThinking()
							flushText()
							continue
						}

						itemType, _ := jsonutil.String(item, "type")
						switch itemType {
						case "text":
							flushThinking()
							if text, ok := jsonutil.String(item, "text"); ok {
								textBuilder.WriteString(text)
							}
						case "tool_calls":
							flushThinking()
							flushText()
							if toolCalls, ok := jsonutil.Array(item, "tool_calls"); ok {
								for _, tcValue := range toolCalls {
									tc, ok := tcValue.(map[string]any)
									if !ok {
										continue
									}
									function, ok := jsonutil.Object(tc, "function")
									if !ok {
										continue
									}
									hasToolCall = true
									outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
										"type":  "tool_use",
										"id":    util.SanitizeClaudeToolID(openAIClaudeString(tc["id"])),
										"name":  util.MapToolName(toolNameMap, openAIClaudeString(function["name"])),
										"input": openAIClaudeArgumentsToObject(openAIClaudeString(function["arguments"])),
									})
								}
							}
						case "reasoning":
							flushText()
							if thinking, ok := jsonutil.String(item, "text"); ok {
								thinkingBuilder.WriteString(thinking)
							}
						default:
							flushThinking()
							flushText()
						}
					}

					flushThinking()
					flushText()

				case string:
					if typed != "" {
						outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
							"type": "text",
							"text": typed,
						})
					}
				}
			}

			if reasoningValue, ok := jsonutil.Get(message, "reasoning_content"); ok {
				for _, reasoningText := range collectOpenAIReasoningTexts(reasoningValue) {
					if reasoningText == "" {
						continue
					}
					outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
						"type":     "thinking",
						"thinking": reasoningText,
					})
				}
			}

			if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
				for _, toolCallValue := range toolCalls {
					toolCall, ok := toolCallValue.(map[string]any)
					if !ok {
						continue
					}
					function, ok := jsonutil.Object(toolCall, "function")
					if !ok {
						continue
					}
					hasToolCall = true
					outRoot["content"] = append(outRoot["content"].([]any), map[string]any{
						"type":  "tool_use",
						"id":    util.SanitizeClaudeToolID(openAIClaudeString(toolCall["id"])),
						"name":  util.MapToolName(toolNameMap, openAIClaudeString(function["name"])),
						"input": openAIClaudeArgumentsToObject(openAIClaudeString(function["arguments"])),
					})
				}
			}
		}
	}

	if usage, ok := jsonutil.Object(root, "usage"); ok {
		inputTokens, outputTokens, cachedTokens := extractOpenAIUsage(usage)
		outRoot["usage"].(map[string]any)["input_tokens"] = inputTokens
		outRoot["usage"].(map[string]any)["output_tokens"] = outputTokens
		if cachedTokens > 0 {
			outRoot["usage"].(map[string]any)["cache_read_input_tokens"] = cachedTokens
		}
	}

	if !stopReasonSet {
		if hasToolCall {
			outRoot["stop_reason"] = "tool_use"
		} else {
			outRoot["stop_reason"] = "end_turn"
		}
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}

func extractOpenAIUsage(usage map[string]any) (int64, int64, int64) {
	inputTokens, _ := jsonutil.Int64(usage, "prompt_tokens")
	outputTokens, _ := jsonutil.Int64(usage, "completion_tokens")
	cachedTokens, _ := jsonutil.Int64(usage, "prompt_tokens_details.cached_tokens")
	if cachedTokens > 0 {
		if inputTokens >= cachedTokens {
			inputTokens -= cachedTokens
		} else {
			inputTokens = 0
		}
	}
	return inputTokens, outputTokens, cachedTokens
}

func isStreamingOpenAIRequest(originalRequestRawJSON []byte) bool {
	root := jsonutil.ParseObjectBytesOrEmpty(originalRequestRawJSON)
	stream, ok := jsonutil.Bool(root, "stream")
	return ok && stream
}

func anthropicSSEEvent(event string, payload map[string]any) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n", event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func openAIClaudeArgumentsToObject(arguments string) map[string]any {
	fixed := util.FixJSON(arguments)
	if fixed == "" {
		return map[string]any{}
	}
	parsed, errParse := jsonutil.ParseAnyBytes([]byte(fixed))
	if errParse != nil {
		return map[string]any{}
	}
	if object, ok := parsed.(map[string]any); ok {
		return object
	}
	return map[string]any{}
}

func openAIClaudeString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	default:
		return fmt.Sprintf("%v", typed)
	}
}
