// Package openai provides response translation functionality for Antigravity to
// OpenAI API compatibility using standard JSON trees.
package chat_completions

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/openai/chat-completions"
	log "github.com/sirupsen/logrus"
)

// convertCliResponseToOpenAIChatParams holds parameters for response conversion.
type convertCliResponseToOpenAIChatParams struct {
	UnixTimestamp        int64
	FunctionIndex        int
	SawToolCall          bool
	UpstreamFinishReason string
}

// functionCallIDCounter provides a process-wide unique counter for function call identifiers.
var functionCallIDCounter uint64

// ConvertAntigravityResponseToOpenAI translates a single chunk of a streaming
// response from Antigravity format to the OpenAI Chat Completions streaming
// format.
func ConvertAntigravityResponseToOpenAI(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &convertCliResponseToOpenAIChatParams{
			UnixTimestamp:        0,
			FunctionIndex:        0,
			SawToolCall:          false,
			UpstreamFinishReason: "",
		}
	}
	p := (*param).(*convertCliResponseToOpenAIChatParams)

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		return []string{}
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, antigravityOpenAIBaseChunk(p)))}
	}

	if finishReason, ok := jsonutil.String(responseRoot, "candidates.0.finishReason"); ok && finishReason != "" {
		p.UpstreamFinishReason = strings.ToUpper(finishReason)
	}

	outRoot := antigravityOpenAIBaseChunk(p)
	choice := outRoot["choices"].([]any)[0].(map[string]any)
	delta := choice["delta"].(map[string]any)

	if modelVersion, ok := jsonutil.String(responseRoot, "modelVersion"); ok {
		outRoot["model"] = modelVersion
	}
	if createTime, ok := jsonutil.String(responseRoot, "createTime"); ok {
		if parsedTime, errParse := time.Parse(time.RFC3339Nano, createTime); errParse == nil {
			p.UnixTimestamp = parsedTime.Unix()
		}
	}
	outRoot["created"] = p.UnixTimestamp
	if responseID, ok := jsonutil.String(responseRoot, "responseId"); ok {
		outRoot["id"] = responseID
	}

	if usageMetadata, ok := jsonutil.Object(responseRoot, "usageMetadata"); ok {
		usage := map[string]any{}
		if candidatesTokenCount, ok := jsonutil.Int64(usageMetadata, "candidatesTokenCount"); ok {
			usage["completion_tokens"] = candidatesTokenCount
		}
		if totalTokenCount, ok := jsonutil.Int64(usageMetadata, "totalTokenCount"); ok {
			usage["total_tokens"] = totalTokenCount
		}
		if promptTokenCount, ok := jsonutil.Int64(usageMetadata, "promptTokenCount"); ok {
			usage["prompt_tokens"] = promptTokenCount
		}
		if thoughtsTokenCount, ok := jsonutil.Int64(usageMetadata, "thoughtsTokenCount"); ok && thoughtsTokenCount > 0 {
			usage["completion_tokens_details"] = map[string]any{
				"reasoning_tokens": thoughtsTokenCount,
			}
		}
		if cachedTokenCount, ok := jsonutil.Int64(usageMetadata, "cachedContentTokenCount"); ok && cachedTokenCount > 0 {
			details, _ := usage["prompt_tokens_details"].(map[string]any)
			if details == nil {
				details = map[string]any{}
				usage["prompt_tokens_details"] = details
			}
			details["cached_tokens"] = cachedTokenCount
		}
		outRoot["usage"] = usage
	}

	if parts, ok := jsonutil.Array(responseRoot, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			text, hasText := jsonutil.String(part, "text")
			functionCall, hasFunctionCall := jsonutil.Object(part, "functionCall")
			thoughtSignature, hasThoughtSignature := jsonutil.String(part, "thoughtSignature")
			if !hasThoughtSignature {
				thoughtSignature, hasThoughtSignature = jsonutil.String(part, "thought_signature")
			}
			inlineData, hasInlineData := jsonutil.Object(part, "inlineData")
			if !hasInlineData {
				inlineData, hasInlineData = jsonutil.Object(part, "inline_data")
			}

			hasContentPayload := hasText || hasFunctionCall || hasInlineData
			if hasThoughtSignature && thoughtSignature != "" && !hasContentPayload {
				continue
			}

			if hasText {
				if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
					delta["reasoning_content"] = text
				} else {
					delta["content"] = text
				}
				delta["role"] = "assistant"
				continue
			}

			if hasFunctionCall {
				p.SawToolCall = true
				functionCallIndex := p.FunctionIndex
				p.FunctionIndex++

				toolCalls, _ := delta["tool_calls"].([]any)
				if toolCalls != nil {
					functionCallIndex = len(toolCalls)
				}

				toolCall := map[string]any{
					"id":    fmt.Sprintf("%s-%d-%d", antigravityOpenAIString(functionCall["name"]), time.Now().UnixNano(), atomic.AddUint64(&functionCallIDCounter, 1)),
					"index": functionCallIndex,
					"type":  "function",
					"function": map[string]any{
						"name":      antigravityOpenAIString(functionCall["name"]),
						"arguments": "",
					},
				}
				if argsValue, ok := functionCall["args"]; ok {
					toolCall["function"].(map[string]any)["arguments"] = string(jsonutil.MarshalOrOriginal(nil, argsValue))
				}

				delta["role"] = "assistant"
				if delta["tool_calls"] == nil {
					delta["tool_calls"] = []any{}
				}
				delta["tool_calls"] = append(delta["tool_calls"].([]any), toolCall)
				continue
			}

			if hasInlineData {
				data, _ := jsonutil.String(inlineData, "data")
				if data == "" {
					continue
				}
				mimeType, _ := jsonutil.String(inlineData, "mimeType")
				if mimeType == "" {
					mimeType, _ = jsonutil.String(inlineData, "mime_type")
				}
				if mimeType == "" {
					mimeType = "image/png"
				}

				imagePayload := map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": fmt.Sprintf("data:%s;base64,%s", mimeType, data),
					},
				}
				images, _ := delta["images"].([]any)
				imagePayload["index"] = len(images)
				delta["role"] = "assistant"
				delta["images"] = append(images, imagePayload)
			}
		}
	}

	usageExists := jsonutil.Exists(responseRoot, "usageMetadata")
	isFinalChunk := p.UpstreamFinishReason != "" && usageExists
	if isFinalChunk {
		finishReason := "stop"
		if p.SawToolCall {
			finishReason = "tool_calls"
		} else if p.UpstreamFinishReason == "MAX_TOKENS" {
			finishReason = "max_tokens"
		}
		choice["finish_reason"] = finishReason
		choice["native_finish_reason"] = strings.ToLower(p.UpstreamFinishReason)
	}

	return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}
}

// ConvertAntigravityResponseToOpenAINonStream converts a non-streaming
// Antigravity response to a non-streaming OpenAI response.
func ConvertAntigravityResponseToOpenAINonStream(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return ""
	}

	responseBytes, errMarshal := json.Marshal(responseRoot)
	if errMarshal != nil {
		log.Warnf("antigravity openai response: failed to marshal response body: %v", errMarshal)
		return ""
	}
	return ConvertGeminiResponseToOpenAINonStream(ctx, modelName, originalRequestRawJSON, requestRawJSON, responseBytes, param)
}

func antigravityOpenAIBaseChunk(param *convertCliResponseToOpenAIChatParams) map[string]any {
	return map[string]any{
		"id":      "",
		"object":  "chat.completion.chunk",
		"created": param.UnixTimestamp,
		"model":   "model",
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

func antigravityOpenAIString(value any) string {
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
		return ""
	}
}
