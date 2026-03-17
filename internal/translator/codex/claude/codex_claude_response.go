// Package claude provides response translation functionality for Codex to
// Claude Code API compatibility using standard JSON trees.
package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
)

var (
	dataTag = []byte("data:")
)

// ConvertCodexResponseToClaudeParams holds parameters for response conversion.
type ConvertCodexResponseToClaudeParams struct {
	HasToolCall               bool
	BlockIndex                int
	HasReceivedArgumentsDelta bool
}

// ConvertCodexResponseToClaude translates Codex streaming events into Claude
// SSE events.
func ConvertCodexResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &ConvertCodexResponseToClaudeParams{
			HasToolCall:               false,
			BlockIndex:                0,
			HasReceivedArgumentsDelta: false,
		}
	}
	p := (*param).(*ConvertCodexResponseToClaudeParams)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	typeStr, _ := jsonutil.String(root, "type")
	reverseNames := buildReverseMapFromClaudeOriginalShortToOriginal(originalRequestRawJSON)
	events := make([]string, 0)

	switch typeStr {
	case "response.created":
		responseID, _ := jsonutil.String(root, "response.id")
		responseModel, _ := jsonutil.String(root, "response.model")
		message := map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            responseID,
				"type":          "message",
				"role":          "assistant",
				"model":         responseModel,
				"stop_sequence": nil,
				"usage": map[string]any{
					"input_tokens":  0,
					"output_tokens": 0,
				},
				"content":     []any{},
				"stop_reason": nil,
			},
		}
		events = append(events, claudeSSEEvent("message_start", message))

	case "response.reasoning_summary_part.added":
		events = append(events, claudeSSEEvent("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": p.BlockIndex,
			"content_block": map[string]any{
				"type":     "thinking",
				"thinking": "",
			},
		}))

	case "response.reasoning_summary_text.delta":
		events = append(events, claudeSSEEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": p.BlockIndex,
			"delta": map[string]any{
				"type":     "thinking_delta",
				"thinking": codexClaudeString(root["delta"]),
			},
		}))

	case "response.reasoning_summary_part.done":
		events = append(events, claudeSSEEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": p.BlockIndex,
		}))
		p.BlockIndex++

	case "response.content_part.added":
		events = append(events, claudeSSEEvent("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": p.BlockIndex,
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		}))

	case "response.output_text.delta":
		events = append(events, claudeSSEEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": p.BlockIndex,
			"delta": map[string]any{
				"type": "text_delta",
				"text": codexClaudeString(root["delta"]),
			},
		}))

	case "response.content_part.done":
		events = append(events, claudeSSEEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": p.BlockIndex,
		}))
		p.BlockIndex++

	case "response.completed":
		stopReason := "end_turn"
		if p.HasToolCall {
			stopReason = "tool_use"
		} else if responseStopReason, ok := jsonutil.String(root, "response.stop_reason"); ok {
			if responseStopReason == "max_tokens" || responseStopReason == "stop" {
				stopReason = responseStopReason
			}
		}

		inputTokens, outputTokens, cachedTokens := extractResponsesUsageMap(root)
		messageDelta := map[string]any{
			"type": "message_delta",
			"delta": map[string]any{
				"stop_reason":   stopReason,
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
		events = append(events, claudeSSEEvent("message_delta", messageDelta))
		events = append(events, claudeSSEEvent("message_stop", map[string]any{"type": "message_stop"}))

	case "response.output_item.added":
		item, ok := jsonutil.Object(root, "item")
		if !ok {
			break
		}
		itemType, _ := jsonutil.String(item, "type")
		if itemType == "function_call" {
			p.HasToolCall = true
			p.HasReceivedArgumentsDelta = false
			name, _ := jsonutil.String(item, "name")
			if originalName, ok := reverseNames[name]; ok {
				name = originalName
			}

			events = append(events, claudeSSEEvent("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": p.BlockIndex,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    util.SanitizeClaudeToolID(codexClaudeString(item["call_id"])),
					"name":  name,
					"input": map[string]any{},
				},
			}))
			events = append(events, claudeSSEEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": p.BlockIndex,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": "",
				},
			}))
		}

	case "response.output_item.done":
		itemType, _ := jsonutil.String(root, "item.type")
		if itemType == "function_call" {
			events = append(events, claudeSSEEvent("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": p.BlockIndex,
			}))
			p.BlockIndex++
		}

	case "response.function_call_arguments.delta":
		p.HasReceivedArgumentsDelta = true
		events = append(events, claudeSSEEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": p.BlockIndex,
			"delta": map[string]any{
				"type":         "input_json_delta",
				"partial_json": codexClaudeString(root["delta"]),
			},
		}))

	case "response.function_call_arguments.done":
		if !p.HasReceivedArgumentsDelta {
			if arguments, ok := jsonutil.String(root, "arguments"); ok && arguments != "" {
				events = append(events, claudeSSEEvent("content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": p.BlockIndex,
					"delta": map[string]any{
						"type":         "input_json_delta",
						"partial_json": arguments,
					},
				}))
			}
		}
	}

	if len(events) == 0 {
		return []string{""}
	}
	return []string{strings.Join(events, "")}
}

// ConvertCodexResponseToClaudeNonStream converts a non-streaming Codex response
// to a non-streaming Claude response.
func ConvertCodexResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, _ []byte, rawJSON []byte, _ *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	if responseType, _ := jsonutil.String(root, "type"); responseType != "response.completed" {
		return ""
	}

	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return ""
	}

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

	if responseID, ok := jsonutil.String(responseRoot, "id"); ok {
		outRoot["id"] = responseID
	}
	if modelName, ok := jsonutil.String(responseRoot, "model"); ok {
		outRoot["model"] = modelName
	}

	inputTokens, outputTokens, cachedTokens := extractResponsesUsageMap(root)
	outRoot["usage"].(map[string]any)["input_tokens"] = inputTokens
	outRoot["usage"].(map[string]any)["output_tokens"] = outputTokens
	if cachedTokens > 0 {
		outRoot["usage"].(map[string]any)["cache_read_input_tokens"] = cachedTokens
	}

	reverseNames := buildReverseMapFromClaudeOriginalShortToOriginal(originalRequestRawJSON)
	contentBlocks := make([]any, 0)
	hasToolCall := false

	if outputItems, ok := jsonutil.Array(responseRoot, "output"); ok {
		for _, itemValue := range outputItems {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}

			itemType, _ := jsonutil.String(item, "type")
			switch itemType {
			case "reasoning":
				if thinking := codexClaudeThinkingText(item); thinking != "" {
					contentBlocks = append(contentBlocks, map[string]any{
						"type":     "thinking",
						"thinking": thinking,
					})
				}

			case "message":
				if content, ok := jsonutil.Get(item, "content"); ok {
					switch typed := content.(type) {
					case []any:
						for _, partValue := range typed {
							part, ok := partValue.(map[string]any)
							if !ok {
								continue
							}
							if partType, _ := jsonutil.String(part, "type"); partType == "output_text" {
								if text, ok := jsonutil.String(part, "text"); ok && text != "" {
									contentBlocks = append(contentBlocks, map[string]any{
										"type": "text",
										"text": text,
									})
								}
							}
						}
					default:
						text := codexClaudeString(typed)
						if text != "" {
							contentBlocks = append(contentBlocks, map[string]any{
								"type": "text",
								"text": text,
							})
						}
					}
				}

			case "function_call":
				hasToolCall = true
				name, _ := jsonutil.String(item, "name")
				if originalName, ok := reverseNames[name]; ok {
					name = originalName
				}
				contentBlocks = append(contentBlocks, map[string]any{
					"type":  "tool_use",
					"id":    util.SanitizeClaudeToolID(codexClaudeString(item["call_id"])),
					"name":  name,
					"input": codexClaudeParseArguments(item),
				})
			}
		}
	}

	outRoot["content"] = contentBlocks

	if stopReason, ok := jsonutil.String(responseRoot, "stop_reason"); ok && stopReason != "" {
		outRoot["stop_reason"] = stopReason
	} else if hasToolCall {
		outRoot["stop_reason"] = "tool_use"
	} else {
		outRoot["stop_reason"] = "end_turn"
	}

	if stopSequence, ok := jsonutil.String(responseRoot, "stop_sequence"); ok && stopSequence != "" {
		outRoot["stop_sequence"] = stopSequence
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func extractResponsesUsageMap(root map[string]any) (int64, int64, int64) {
	usage, ok := jsonutil.Object(root, "response.usage")
	if !ok {
		return 0, 0, 0
	}

	inputTokens, _ := jsonutil.Int64(usage, "input_tokens")
	outputTokens, _ := jsonutil.Int64(usage, "output_tokens")
	cachedTokens, _ := jsonutil.Int64(usage, "input_tokens_details.cached_tokens")

	if cachedTokens > 0 {
		if inputTokens >= cachedTokens {
			inputTokens -= cachedTokens
		} else {
			inputTokens = 0
		}
	}

	return inputTokens, outputTokens, cachedTokens
}

func buildReverseMapFromClaudeOriginalShortToOriginal(original []byte) map[string]string {
	root := jsonutil.ParseObjectBytesOrEmpty(original)
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

	reverseMap := map[string]string{}
	if len(names) > 0 {
		shortMap := buildShortNameMap(names)
		for originalName, shortName := range shortMap {
			reverseMap[shortName] = originalName
		}
	}
	return reverseMap
}

func claudeSSEEvent(event string, payload map[string]any) string {
	return fmt.Sprintf("event: %s\ndata: %s\n\n", event, string(jsonutil.MarshalOrOriginal(nil, payload)))
}

func codexClaudeThinkingText(item map[string]any) string {
	if summary, ok := jsonutil.Get(item, "summary"); ok {
		switch typed := summary.(type) {
		case []any:
			var builder strings.Builder
			for _, partValue := range typed {
				if part, ok := partValue.(map[string]any); ok {
					if text, ok := jsonutil.String(part, "text"); ok {
						builder.WriteString(text)
						continue
					}
				}
				builder.WriteString(codexClaudeString(partValue))
			}
			if builder.Len() > 0 {
				return builder.String()
			}
		default:
			text := codexClaudeString(typed)
			if text != "" {
				return text
			}
		}
	}

	if content, ok := jsonutil.Get(item, "content"); ok {
		switch typed := content.(type) {
		case []any:
			var builder strings.Builder
			for _, partValue := range typed {
				if part, ok := partValue.(map[string]any); ok {
					if text, ok := jsonutil.String(part, "text"); ok {
						builder.WriteString(text)
						continue
					}
				}
				builder.WriteString(codexClaudeString(partValue))
			}
			return builder.String()
		default:
			return codexClaudeString(typed)
		}
	}

	return ""
}

func codexClaudeParseArguments(item map[string]any) map[string]any {
	arguments, _ := jsonutil.String(item, "arguments")
	if arguments == "" {
		return map[string]any{}
	}

	value, errParse := jsonutil.ParseAnyBytes([]byte(arguments))
	if errParse != nil {
		return map[string]any{}
	}
	object, ok := value.(map[string]any)
	if !ok {
		return map[string]any{}
	}
	return object
}

func codexClaudeString(value any) string {
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

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}
