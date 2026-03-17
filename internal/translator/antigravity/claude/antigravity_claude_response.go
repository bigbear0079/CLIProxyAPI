// Package claude provides response translation functionality for Claude Code
// API compatibility using standard JSON trees.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/cache"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

// Params holds parameters for response conversion and maintains state across
// streaming chunks.
type Params struct {
	HasFirstResponse     bool   // Indicates if the initial message_start event has been sent
	ResponseType         int    // Current response type: 0=none, 1=content, 2=thinking, 3=function
	ResponseIndex        int    // Index counter for content blocks in the streaming response
	HasFinishReason      bool   // Tracks whether a finish reason has been observed
	FinishReason         string // The finish reason string returned by the provider
	HasUsageMetadata     bool   // Tracks whether usage metadata has been observed
	PromptTokenCount     int64  // Cached prompt token count from usage metadata
	CandidatesTokenCount int64  // Cached candidate token count from usage metadata
	ThoughtsTokenCount   int64  // Cached thinking token count from usage metadata
	TotalTokenCount      int64  // Cached total token count from usage metadata
	CachedTokenCount     int64  // Cached content token count (indicates prompt caching)
	HasSentFinalEvents   bool   // Indicates if final content/message events have been sent
	HasToolUse           bool   // Indicates if tool use was observed in the stream
	HasContent           bool   // Tracks whether any content (text, thinking, or tool use) has been output
	ModelName            string // Cached model name for signature grouping

	CurrentThinkingText strings.Builder // Accumulates thinking text for signature caching
}

// toolUseIDCounter provides a process-wide unique counter for tool use
// identifiers.
var toolUseIDCounter uint64

// ConvertAntigravityResponseToClaude translates backend responses into Claude
// SSE events.
func ConvertAntigravityResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = originalRequestRawJSON

	if *param == nil {
		*param = &Params{
			HasFirstResponse: false,
			ResponseType:     0,
			ResponseIndex:    0,
		}
	}
	params := (*param).(*Params)
	if params.ModelName == "" {
		requestRoot := jsonutil.ParseObjectBytesOrEmpty(requestRawJSON)
		if modelName, ok := jsonutil.String(requestRoot, "model"); ok {
			params.ModelName = modelName
		}
	}

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		if !params.HasContent {
			return []string{}
		}

		var builder strings.Builder
		appendFinalEvents(params, &builder, true)
		builder.WriteString(antigravityClaudeEvent("message_stop", map[string]any{
			"type": "message_stop",
		}))
		return []string{builder.String()}
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, _ := jsonutil.Object(root, "response")

	var builder strings.Builder
	if !params.HasFirstResponse {
		message := map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
				"type":          "message",
				"role":          "assistant",
				"content":       []any{},
				"model":         "claude-3-5-sonnet-20241022",
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage": map[string]any{
					"input_tokens":  int64(0),
					"output_tokens": int64(0),
				},
			},
		}

		if promptTokenCount, ok := jsonutil.Int64(responseRoot, "cpaUsageMetadata.promptTokenCount"); ok {
			message["message"].(map[string]any)["usage"].(map[string]any)["input_tokens"] = promptTokenCount
		}
		if candidatesTokenCount, ok := jsonutil.Int64(responseRoot, "cpaUsageMetadata.candidatesTokenCount"); ok {
			message["message"].(map[string]any)["usage"].(map[string]any)["output_tokens"] = candidatesTokenCount
		}
		if modelVersion, ok := jsonutil.String(responseRoot, "modelVersion"); ok {
			message["message"].(map[string]any)["model"] = modelVersion
		}
		if responseID, ok := jsonutil.String(responseRoot, "responseId"); ok {
			message["message"].(map[string]any)["id"] = responseID
		}

		builder.WriteString(antigravityClaudeEvent("message_start", message))
		params.HasFirstResponse = true
	}

	if parts, ok := jsonutil.Array(responseRoot, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			partText, hasText := jsonutil.String(part, "text")
			functionCall, hasFunctionCall := jsonutil.Object(part, "functionCall")

			if hasText {
				isThought, _ := jsonutil.Bool(part, "thought")
				if isThought {
					thoughtSignature, _ := jsonutil.String(part, "thoughtSignature")
					if thoughtSignature == "" {
						thoughtSignature, _ = jsonutil.String(part, "thought_signature")
					}

					if thoughtSignature != "" {
						if params.CurrentThinkingText.Len() > 0 {
							cache.CacheSignature(params.ModelName, params.CurrentThinkingText.String(), thoughtSignature)
							params.CurrentThinkingText.Reset()
						}

						builder.WriteString(antigravityClaudeEvent("content_block_delta", map[string]any{
							"type":  "content_block_delta",
							"index": params.ResponseIndex,
							"delta": map[string]any{
								"type":      "signature_delta",
								"signature": fmt.Sprintf("%s#%s", cache.GetModelGroup(params.ModelName), thoughtSignature),
							},
						}))
						params.HasContent = true
						continue
					}

					if params.ResponseType == 2 {
						params.CurrentThinkingText.WriteString(partText)
						builder.WriteString(antigravityClaudeEvent("content_block_delta", map[string]any{
							"type":  "content_block_delta",
							"index": params.ResponseIndex,
							"delta": map[string]any{
								"type":     "thinking_delta",
								"thinking": partText,
							},
						}))
						params.HasContent = true
					} else {
						if params.ResponseType != 0 {
							builder.WriteString(antigravityClaudeEvent("content_block_stop", map[string]any{
								"type":  "content_block_stop",
								"index": params.ResponseIndex,
							}))
							params.ResponseIndex++
						}

						builder.WriteString(antigravityClaudeEvent("content_block_start", map[string]any{
							"type":  "content_block_start",
							"index": params.ResponseIndex,
							"content_block": map[string]any{
								"type":     "thinking",
								"thinking": "",
							},
						}))
						builder.WriteString(antigravityClaudeEvent("content_block_delta", map[string]any{
							"type":  "content_block_delta",
							"index": params.ResponseIndex,
							"delta": map[string]any{
								"type":     "thinking_delta",
								"thinking": partText,
							},
						}))
						params.ResponseType = 2
						params.HasContent = true
						params.CurrentThinkingText.Reset()
						params.CurrentThinkingText.WriteString(partText)
					}
					continue
				}

				if partText != "" || !jsonutil.Exists(responseRoot, "candidates.0.finishReason") {
					if params.ResponseType == 1 {
						builder.WriteString(antigravityClaudeEvent("content_block_delta", map[string]any{
							"type":  "content_block_delta",
							"index": params.ResponseIndex,
							"delta": map[string]any{
								"type": "text_delta",
								"text": partText,
							},
						}))
						params.HasContent = true
					} else {
						if params.ResponseType != 0 {
							builder.WriteString(antigravityClaudeEvent("content_block_stop", map[string]any{
								"type":  "content_block_stop",
								"index": params.ResponseIndex,
							}))
							params.ResponseIndex++
						}

						if partText != "" {
							builder.WriteString(antigravityClaudeEvent("content_block_start", map[string]any{
								"type":  "content_block_start",
								"index": params.ResponseIndex,
								"content_block": map[string]any{
									"type": "text",
									"text": "",
								},
							}))
							builder.WriteString(antigravityClaudeEvent("content_block_delta", map[string]any{
								"type":  "content_block_delta",
								"index": params.ResponseIndex,
								"delta": map[string]any{
									"type": "text_delta",
									"text": partText,
								},
							}))
							params.ResponseType = 1
							params.HasContent = true
						}
					}
				}
				continue
			}

			if hasFunctionCall {
				params.HasToolUse = true
				fcName, _ := jsonutil.String(functionCall, "name")

				if params.ResponseType == 3 {
					builder.WriteString(antigravityClaudeEvent("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": params.ResponseIndex,
					}))
					params.ResponseIndex++
					params.ResponseType = 0
				}

				if params.ResponseType != 0 {
					builder.WriteString(antigravityClaudeEvent("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": params.ResponseIndex,
					}))
					params.ResponseIndex++
				}

				builder.WriteString(antigravityClaudeEvent("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": params.ResponseIndex,
					"content_block": map[string]any{
						"type":  "tool_use",
						"id":    util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d-%d", fcName, time.Now().UnixNano(), atomic.AddUint64(&toolUseIDCounter, 1))),
						"name":  fcName,
						"input": map[string]any{},
					},
				}))

				if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
					builder.WriteString(antigravityClaudeEvent("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": params.ResponseIndex,
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": string(jsonutil.MarshalOrOriginal(nil, argsValue)),
						},
					}))
				}

				params.ResponseType = 3
				params.HasContent = true
			}
		}
	}

	if finishReason, ok := jsonutil.String(responseRoot, "candidates.0.finishReason"); ok {
		params.HasFinishReason = true
		params.FinishReason = finishReason
	}

	if usageMetadata, ok := jsonutil.Object(responseRoot, "usageMetadata"); ok {
		params.HasUsageMetadata = true
		params.CachedTokenCount, _ = jsonutil.Int64(usageMetadata, "cachedContentTokenCount")
		promptTokenCount, _ := jsonutil.Int64(usageMetadata, "promptTokenCount")
		params.PromptTokenCount = promptTokenCount - params.CachedTokenCount
		params.CandidatesTokenCount, _ = jsonutil.Int64(usageMetadata, "candidatesTokenCount")
		params.ThoughtsTokenCount, _ = jsonutil.Int64(usageMetadata, "thoughtsTokenCount")
		params.TotalTokenCount, _ = jsonutil.Int64(usageMetadata, "totalTokenCount")
		if params.CandidatesTokenCount == 0 && params.TotalTokenCount > 0 {
			params.CandidatesTokenCount = params.TotalTokenCount - params.PromptTokenCount - params.ThoughtsTokenCount
			if params.CandidatesTokenCount < 0 {
				params.CandidatesTokenCount = 0
			}
		}
	}

	if params.HasUsageMetadata && params.HasFinishReason {
		appendFinalEvents(params, &builder, false)
	}

	return []string{builder.String()}
}

func appendFinalEvents(params *Params, builder *strings.Builder, force bool) {
	if params.HasSentFinalEvents {
		return
	}
	if !params.HasUsageMetadata && !force {
		return
	}
	if !params.HasContent {
		return
	}

	if params.ResponseType != 0 {
		builder.WriteString(antigravityClaudeEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": params.ResponseIndex,
		}))
		params.ResponseType = 0
	}

	usageOutputTokens := params.CandidatesTokenCount + params.ThoughtsTokenCount
	if usageOutputTokens == 0 && params.TotalTokenCount > 0 {
		usageOutputTokens = params.TotalTokenCount - params.PromptTokenCount
		if usageOutputTokens < 0 {
			usageOutputTokens = 0
		}
	}

	messageDelta := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   resolveStopReason(params),
			"stop_sequence": nil,
		},
		"usage": map[string]any{
			"input_tokens":  params.PromptTokenCount,
			"output_tokens": usageOutputTokens,
		},
	}
	if params.CachedTokenCount > 0 {
		messageDelta["usage"].(map[string]any)["cache_read_input_tokens"] = params.CachedTokenCount
	}

	builder.WriteString(antigravityClaudeEvent("message_delta", messageDelta))
	params.HasSentFinalEvents = true
}

func resolveStopReason(params *Params) string {
	if params.HasToolUse {
		return "tool_use"
	}

	switch params.FinishReason {
	case "MAX_TOKENS":
		return "max_tokens"
	case "STOP", "FINISH_REASON_UNSPECIFIED", "UNKNOWN":
		return "end_turn"
	}

	return "end_turn"
}

// ConvertAntigravityResponseToClaudeNonStream converts a non-streaming backend
// response to a non-streaming Claude response.
func ConvertAntigravityResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = originalRequestRawJSON

	requestRoot := jsonutil.ParseObjectBytesOrEmpty(requestRawJSON)
	modelName, _ := jsonutil.String(requestRoot, "model")

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, _ := jsonutil.Object(root, "response")

	promptTokens, _ := jsonutil.Int64(responseRoot, "usageMetadata.promptTokenCount")
	candidateTokens, _ := jsonutil.Int64(responseRoot, "usageMetadata.candidatesTokenCount")
	thoughtTokens, _ := jsonutil.Int64(responseRoot, "usageMetadata.thoughtsTokenCount")
	totalTokens, _ := jsonutil.Int64(responseRoot, "usageMetadata.totalTokenCount")
	cachedTokens, _ := jsonutil.Int64(responseRoot, "usageMetadata.cachedContentTokenCount")

	outputTokens := candidateTokens + thoughtTokens
	if outputTokens == 0 && totalTokens > 0 {
		outputTokens = totalTokens - promptTokens
		if outputTokens < 0 {
			outputTokens = 0
		}
	}

	outRoot := map[string]any{
		"id":            "",
		"type":          "message",
		"role":          "assistant",
		"model":         "",
		"content":       nil,
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  promptTokens,
			"output_tokens": outputTokens,
		},
	}
	if responseID, ok := jsonutil.String(responseRoot, "responseId"); ok {
		outRoot["id"] = responseID
	}
	if modelVersion, ok := jsonutil.String(responseRoot, "modelVersion"); ok {
		outRoot["model"] = modelVersion
	}
	if cachedTokens > 0 {
		outRoot["usage"].(map[string]any)["cache_read_input_tokens"] = cachedTokens
	}

	contentBlocks := make([]any, 0)
	textBuilder := strings.Builder{}
	thinkingBuilder := strings.Builder{}
	thinkingSignature := ""
	toolIDCounter := 0
	hasToolCall := false

	flushText := func() {
		if textBuilder.Len() == 0 {
			return
		}
		contentBlocks = append(contentBlocks, map[string]any{
			"type": "text",
			"text": textBuilder.String(),
		})
		textBuilder.Reset()
	}

	flushThinking := func() {
		if thinkingBuilder.Len() == 0 && thinkingSignature == "" {
			return
		}
		block := map[string]any{
			"type":     "thinking",
			"thinking": thinkingBuilder.String(),
		}
		if thinkingSignature != "" {
			block["signature"] = fmt.Sprintf("%s#%s", cache.GetModelGroup(modelName), thinkingSignature)
		}
		contentBlocks = append(contentBlocks, block)
		thinkingBuilder.Reset()
		thinkingSignature = ""
	}

	if parts, ok := jsonutil.Array(responseRoot, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			isThought, _ := jsonutil.Bool(part, "thought")
			if isThought {
				signature, _ := jsonutil.String(part, "thoughtSignature")
				if signature == "" {
					signature, _ = jsonutil.String(part, "thought_signature")
				}
				if signature != "" {
					thinkingSignature = signature
				}
			}

			if text, ok := jsonutil.String(part, "text"); ok && text != "" {
				if isThought {
					flushText()
					thinkingBuilder.WriteString(text)
					continue
				}
				flushThinking()
				textBuilder.WriteString(text)
				continue
			}

			if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
				flushThinking()
				flushText()
				hasToolCall = true

				name, _ := jsonutil.String(functionCall, "name")
				toolIDCounter++
				toolBlock := map[string]any{
					"type":  "tool_use",
					"id":    fmt.Sprintf("tool_%d", toolIDCounter),
					"name":  name,
					"input": map[string]any{},
				}
				if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
					if argsObject, ok := argsValue.(map[string]any); ok {
						toolBlock["input"] = argsObject
					}
				}
				contentBlocks = append(contentBlocks, toolBlock)
			}
		}
	}

	flushThinking()
	flushText()
	if len(contentBlocks) > 0 {
		outRoot["content"] = contentBlocks
	}

	stopReason := "end_turn"
	if hasToolCall {
		stopReason = "tool_use"
	} else if finishReason, ok := jsonutil.String(responseRoot, "candidates.0.finishReason"); ok {
		switch finishReason {
		case "MAX_TOKENS":
			stopReason = "max_tokens"
		case "STOP", "FINISH_REASON_UNSPECIFIED", "UNKNOWN":
			stopReason = "end_turn"
		default:
			stopReason = "end_turn"
		}
	}
	outRoot["stop_reason"] = stopReason

	if promptTokens == 0 && outputTokens == 0 && !jsonutil.Exists(responseRoot, "usageMetadata") {
		delete(outRoot, "usage")
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func antigravityClaudeEvent(event string, payload map[string]any) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n\n", event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}
