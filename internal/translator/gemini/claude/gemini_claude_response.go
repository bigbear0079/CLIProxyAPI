// Package claude provides response translation functionality for Gemini to
// Claude compatibility using standard JSON trees.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

// Params holds parameters for response conversion.
type Params struct {
	IsGlAPIKey       bool
	HasFirstResponse bool
	ResponseType     int
	ResponseIndex    int
	HasContent       bool
	ToolNameMap      map[string]string
	SawToolCall      bool
}

// toolUseIDCounter provides a process-wide unique counter for tool use
// identifiers.
var toolUseIDCounter uint64

// ConvertGeminiResponseToClaude translates Gemini streaming responses to Claude
// SSE events.
func ConvertGeminiResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = requestRawJSON

	if *param == nil {
		*param = &Params{
			IsGlAPIKey:       false,
			HasFirstResponse: false,
			ResponseType:     0,
			ResponseIndex:    0,
			HasContent:       false,
			ToolNameMap:      util.ToolNameMapFromClaudeRequest(originalRequestRawJSON),
			SawToolCall:      false,
		}
	}
	p := (*param).(*Params)

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		if p.HasContent {
			return []string{geminiClaudeEvent("message_stop", map[string]any{"type": "message_stop"})}
		}
		return []string{}
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	var builder strings.Builder

	if !p.HasFirstResponse {
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
					"input_tokens":  0,
					"output_tokens": 0,
				},
			},
		}
		if modelVersion, ok := jsonutil.String(root, "modelVersion"); ok {
			message["message"].(map[string]any)["model"] = modelVersion
		}
		if responseID, ok := jsonutil.String(root, "responseId"); ok {
			message["message"].(map[string]any)["id"] = responseID
		}
		builder.WriteString(geminiClaudeEvent("message_start", message))
		p.HasFirstResponse = true
	}

	if parts, ok := jsonutil.Array(root, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			if text, ok := jsonutil.String(part, "text"); ok {
				if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
					if p.ResponseType == 2 {
						builder.WriteString(geminiClaudeEvent("content_block_delta", map[string]any{
							"type":  "content_block_delta",
							"index": p.ResponseIndex,
							"delta": map[string]any{
								"type":     "thinking_delta",
								"thinking": text,
							},
						}))
						p.HasContent = true
					} else {
						if p.ResponseType != 0 {
							builder.WriteString(geminiClaudeEvent("content_block_stop", map[string]any{
								"type":  "content_block_stop",
								"index": p.ResponseIndex,
							}))
							p.ResponseIndex++
						}
						builder.WriteString(geminiClaudeEvent("content_block_start", map[string]any{
							"type":  "content_block_start",
							"index": p.ResponseIndex,
							"content_block": map[string]any{
								"type":     "thinking",
								"thinking": "",
							},
						}))
						builder.WriteString(geminiClaudeEvent("content_block_delta", map[string]any{
							"type":  "content_block_delta",
							"index": p.ResponseIndex,
							"delta": map[string]any{
								"type":     "thinking_delta",
								"thinking": text,
							},
						}))
						p.ResponseType = 2
						p.HasContent = true
					}
					continue
				}

				if p.ResponseType == 1 {
					builder.WriteString(geminiClaudeEvent("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": p.ResponseIndex,
						"delta": map[string]any{
							"type": "text_delta",
							"text": text,
						},
					}))
					p.HasContent = true
				} else {
					if p.ResponseType != 0 {
						builder.WriteString(geminiClaudeEvent("content_block_stop", map[string]any{
							"type":  "content_block_stop",
							"index": p.ResponseIndex,
						}))
						p.ResponseIndex++
					}
					builder.WriteString(geminiClaudeEvent("content_block_start", map[string]any{
						"type":  "content_block_start",
						"index": p.ResponseIndex,
						"content_block": map[string]any{
							"type": "text",
							"text": "",
						},
					}))
					builder.WriteString(geminiClaudeEvent("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": p.ResponseIndex,
						"delta": map[string]any{
							"type": "text_delta",
							"text": text,
						},
					}))
					p.ResponseType = 1
					p.HasContent = true
				}
				continue
			}

			functionCall, ok := jsonutil.Object(part, "functionCall")
			if !ok {
				continue
			}

			p.SawToolCall = true
			upstreamToolName, _ := jsonutil.String(functionCall, "name")
			clientToolName := util.MapToolName(p.ToolNameMap, upstreamToolName)

			if p.ResponseType == 3 && upstreamToolName == "" {
				if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
					builder.WriteString(geminiClaudeEvent("content_block_delta", map[string]any{
						"type":  "content_block_delta",
						"index": p.ResponseIndex,
						"delta": map[string]any{
							"type":         "input_json_delta",
							"partial_json": string(jsonutil.MarshalOrOriginal(nil, argsValue)),
						},
					}))
				}
				continue
			}

			if p.ResponseType == 3 {
				builder.WriteString(geminiClaudeEvent("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": p.ResponseIndex,
				}))
				p.ResponseIndex++
				p.ResponseType = 0
			}
			if p.ResponseType != 0 {
				builder.WriteString(geminiClaudeEvent("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": p.ResponseIndex,
				}))
				p.ResponseIndex++
			}

			builder.WriteString(geminiClaudeEvent("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": p.ResponseIndex,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d", upstreamToolName, atomic.AddUint64(&toolUseIDCounter, 1))),
					"name":  clientToolName,
					"input": map[string]any{},
				},
			}))
			if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
				builder.WriteString(geminiClaudeEvent("content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": p.ResponseIndex,
					"delta": map[string]any{
						"type":         "input_json_delta",
						"partial_json": string(jsonutil.MarshalOrOriginal(nil, argsValue)),
					},
				}))
			}
			p.ResponseType = 3
			p.HasContent = true
		}
	}

	if usageMetadata, ok := jsonutil.Object(root, "usageMetadata"); ok {
		if _, ok := jsonutil.Get(usageMetadata, "candidatesTokenCount"); ok && bytes.Contains(rawJSON, []byte(`"finishReason"`)) {
			if p.HasContent {
				builder.WriteString(geminiClaudeEvent("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": p.ResponseIndex,
				}))

				stopReason := "end_turn"
				if p.SawToolCall {
					stopReason = "tool_use"
				} else if finishReason, ok := jsonutil.String(root, "candidates.0.finishReason"); ok && finishReason == "MAX_TOKENS" {
					stopReason = "max_tokens"
				}

				inputTokens, _ := jsonutil.Int64(usageMetadata, "promptTokenCount")
				outputTokens, _ := jsonutil.Int64(usageMetadata, "candidatesTokenCount")
				thoughtTokens, _ := jsonutil.Int64(usageMetadata, "thoughtsTokenCount")
				builder.WriteString(geminiClaudeEvent("message_delta", map[string]any{
					"type": "message_delta",
					"delta": map[string]any{
						"stop_reason":   stopReason,
						"stop_sequence": nil,
					},
					"usage": map[string]any{
						"input_tokens":  inputTokens,
						"output_tokens": outputTokens + thoughtTokens,
					},
				}))
			}
		}
	}

	return []string{builder.String()}
}

// ConvertGeminiResponseToClaudeNonStream converts a non-streaming Gemini
// response to a non-streaming Claude response.
func ConvertGeminiResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = requestRawJSON

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	toolNameMap := util.ToolNameMapFromClaudeRequest(originalRequestRawJSON)

	inputTokens, _ := jsonutil.Int64(root, "usageMetadata.promptTokenCount")
	outputTokens, _ := jsonutil.Int64(root, "usageMetadata.candidatesTokenCount")
	thoughtTokens, _ := jsonutil.Int64(root, "usageMetadata.thoughtsTokenCount")

	outRoot := map[string]any{
		"id":            "",
		"type":          "message",
		"role":          "assistant",
		"model":         "",
		"content":       []any{},
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens + thoughtTokens,
		},
	}
	if responseID, ok := jsonutil.String(root, "responseId"); ok {
		outRoot["id"] = responseID
	}
	if modelVersion, ok := jsonutil.String(root, "modelVersion"); ok {
		outRoot["model"] = modelVersion
	}

	contentBlocks := make([]any, 0)
	textBuilder := strings.Builder{}
	thinkingBuilder := strings.Builder{}
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
		if thinkingBuilder.Len() == 0 {
			return
		}
		contentBlocks = append(contentBlocks, map[string]any{
			"type":     "thinking",
			"thinking": thinkingBuilder.String(),
		})
		thinkingBuilder.Reset()
	}

	if parts, ok := jsonutil.Array(root, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			if text, ok := jsonutil.String(part, "text"); ok && text != "" {
				if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
					flushText()
					thinkingBuilder.WriteString(text)
					continue
				}
				flushThinking()
				textBuilder.WriteString(text)
				continue
			}

			functionCall, ok := jsonutil.Object(part, "functionCall")
			if !ok {
				continue
			}

			flushThinking()
			flushText()
			hasToolCall = true

			upstreamToolName, _ := jsonutil.String(functionCall, "name")
			clientToolName := util.MapToolName(toolNameMap, upstreamToolName)
			toolIDCounter++
			toolBlock := map[string]any{
				"type":  "tool_use",
				"id":    util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d", upstreamToolName, toolIDCounter)),
				"name":  clientToolName,
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

	flushThinking()
	flushText()
	outRoot["content"] = contentBlocks

	stopReason := "end_turn"
	if hasToolCall {
		stopReason = "tool_use"
	} else if finishReason, ok := jsonutil.String(root, "candidates.0.finishReason"); ok {
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

	if inputTokens == 0 && outputTokens+thoughtTokens == 0 && !jsonutil.Exists(root, "usageMetadata") {
		delete(outRoot, "usage")
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func geminiClaudeEvent(event string, payload map[string]any) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n\n", event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}
