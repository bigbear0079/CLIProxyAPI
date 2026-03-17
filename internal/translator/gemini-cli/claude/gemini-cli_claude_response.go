// Package claude provides response translation functionality for Gemini CLI to
// Claude Code API compatibility using standard JSON trees.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

// Params holds parameters for response conversion and maintains state across
// streaming chunks.
type Params struct {
	HasFirstResponse bool
	ResponseType     int
	ResponseIndex    int
	HasContent       bool
}

// toolUseIDCounter provides a process-wide unique counter for tool use identifiers.
var toolUseIDCounter uint64

// ConvertGeminiCLIResponseToClaude translates Gemini CLI streaming responses to
// Claude SSE events.
func ConvertGeminiCLIResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &Params{
			HasFirstResponse: false,
			ResponseType:     0,
			ResponseIndex:    0,
			HasContent:       false,
		}
	}
	p := (*param).(*Params)

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		if p.HasContent {
			return []string{claudeCLIEvent("message_stop", map[string]any{"type": "message_stop"})}
		}
		return []string{}
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, _ := jsonutil.Object(root, "response")

	usedTool := false
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
		if modelVersion, ok := jsonutil.String(responseRoot, "modelVersion"); ok {
			message["message"].(map[string]any)["model"] = modelVersion
		}
		if responseID, ok := jsonutil.String(responseRoot, "responseId"); ok {
			message["message"].(map[string]any)["id"] = responseID
		}
		builder.WriteString(claudeCLIEvent("message_start", message))
		p.HasFirstResponse = true
	}

	if parts, ok := jsonutil.Array(responseRoot, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			if text, ok := jsonutil.String(part, "text"); ok && text != "" {
				if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
					if p.ResponseType == 2 {
						builder.WriteString(claudeCLIEvent("content_block_delta", map[string]any{
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
							builder.WriteString(claudeCLIEvent("content_block_stop", map[string]any{
								"type":  "content_block_stop",
								"index": p.ResponseIndex,
							}))
							p.ResponseIndex++
						}
						builder.WriteString(claudeCLIEvent("content_block_start", map[string]any{
							"type":  "content_block_start",
							"index": p.ResponseIndex,
							"content_block": map[string]any{
								"type":     "thinking",
								"thinking": "",
							},
						}))
						builder.WriteString(claudeCLIEvent("content_block_delta", map[string]any{
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
					builder.WriteString(claudeCLIEvent("content_block_delta", map[string]any{
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
						builder.WriteString(claudeCLIEvent("content_block_stop", map[string]any{
							"type":  "content_block_stop",
							"index": p.ResponseIndex,
						}))
						p.ResponseIndex++
					}
					builder.WriteString(claudeCLIEvent("content_block_start", map[string]any{
						"type":  "content_block_start",
						"index": p.ResponseIndex,
						"content_block": map[string]any{
							"type": "text",
							"text": "",
						},
					}))
					builder.WriteString(claudeCLIEvent("content_block_delta", map[string]any{
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

			if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
				usedTool = true
				fcName, _ := jsonutil.String(functionCall, "name")

				if p.ResponseType == 3 {
					builder.WriteString(claudeCLIEvent("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": p.ResponseIndex,
					}))
					p.ResponseIndex++
					p.ResponseType = 0
				}
				if p.ResponseType != 0 {
					builder.WriteString(claudeCLIEvent("content_block_stop", map[string]any{
						"type":  "content_block_stop",
						"index": p.ResponseIndex,
					}))
					p.ResponseIndex++
				}

				builder.WriteString(claudeCLIEvent("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": p.ResponseIndex,
					"content_block": map[string]any{
						"type":  "tool_use",
						"id":    util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d-%d", fcName, time.Now().UnixNano(), atomic.AddUint64(&toolUseIDCounter, 1))),
						"name":  fcName,
						"input": map[string]any{},
					},
				}))
				if argsValue, ok := jsonutil.Get(functionCall, "args"); ok {
					builder.WriteString(claudeCLIEvent("content_block_delta", map[string]any{
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
	}

	if usageMetadata, ok := jsonutil.Object(responseRoot, "usageMetadata"); ok {
		if _, ok := jsonutil.Get(usageMetadata, "candidatesTokenCount"); ok && bytes.Contains(rawJSON, []byte(`"finishReason"`)) {
			if p.HasContent {
				builder.WriteString(claudeCLIEvent("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": p.ResponseIndex,
				}))

				stopReason := "end_turn"
				if usedTool {
					stopReason = "tool_use"
				} else if finishReason, ok := jsonutil.String(responseRoot, "candidates.0.finishReason"); ok && finishReason == "MAX_TOKENS" {
					stopReason = "max_tokens"
				}

				inputTokens, _ := jsonutil.Int64(usageMetadata, "promptTokenCount")
				outputTokens, _ := jsonutil.Int64(usageMetadata, "candidatesTokenCount")
				thoughtTokens, _ := jsonutil.Int64(usageMetadata, "thoughtsTokenCount")

				builder.WriteString(claudeCLIEvent("message_delta", map[string]any{
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

// ConvertGeminiCLIResponseToClaudeNonStream converts a non-streaming Gemini CLI
// response to a non-streaming Claude response.
func ConvertGeminiCLIResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, _ := jsonutil.Object(root, "response")

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
	if responseID, ok := jsonutil.String(responseRoot, "responseId"); ok {
		outRoot["id"] = responseID
	}
	if modelVersion, ok := jsonutil.String(responseRoot, "modelVersion"); ok {
		outRoot["model"] = modelVersion
	}

	inputTokens, hasInput := jsonutil.Int64(responseRoot, "usageMetadata.promptTokenCount")
	outputTokens, hasOutput := jsonutil.Int64(responseRoot, "usageMetadata.candidatesTokenCount")
	thoughtTokens, _ := jsonutil.Int64(responseRoot, "usageMetadata.thoughtsTokenCount")
	if hasInput || hasOutput {
		outRoot["usage"].(map[string]any)["input_tokens"] = inputTokens
		outRoot["usage"].(map[string]any)["output_tokens"] = outputTokens + thoughtTokens
	}

	textBuilder := strings.Builder{}
	thinkingBuilder := strings.Builder{}
	contentBlocks := make([]any, 0)
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

	if parts, ok := jsonutil.Array(responseRoot, "candidates.0.content.parts"); ok {
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
	outRoot["content"] = contentBlocks

	stopReason := "end_turn"
	if hasToolCall {
		stopReason = "tool_use"
	} else if finishReason, ok := jsonutil.String(responseRoot, "candidates.0.finishReason"); ok {
		switch finishReason {
		case "MAX_TOKENS":
			stopReason = "max_tokens"
		default:
			stopReason = "end_turn"
		}
	}
	outRoot["stop_reason"] = stopReason

	if !jsonutil.Exists(responseRoot, "usageMetadata") && inputTokens == 0 && outputTokens == 0 {
		delete(outRoot, "usage")
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func claudeCLIEvent(event string, payload map[string]any) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n\n", event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}
