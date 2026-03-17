// Package openai provides response translation functionality for Gemini CLI to
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
	UnixTimestamp int64
	FunctionIndex int
}

// functionCallIDCounter provides a process-wide unique counter for function call identifiers.
var functionCallIDCounter uint64

// ConvertCliResponseToOpenAI translates a single chunk of a streaming response
// from the Gemini CLI API format to the OpenAI Chat Completions streaming
// format.
func ConvertCliResponseToOpenAI(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &convertCliResponseToOpenAIChatParams{
			UnixTimestamp: 0,
			FunctionIndex: 0,
		}
	}

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		return []string{}
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return []string{}
	}

	p := (*param).(*convertCliResponseToOpenAIChatParams)
	outRoot := map[string]any{
		"id":      "",
		"object":  "chat.completion.chunk",
		"created": p.UnixTimestamp,
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
	delta := outRoot["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	choice := outRoot["choices"].([]any)[0].(map[string]any)

	if modelVersion, ok := jsonutil.String(responseRoot, "modelVersion"); ok {
		outRoot["model"] = modelVersion
	}

	if createTime, ok := jsonutil.String(responseRoot, "createTime"); ok {
		if parsedTime, errParse := time.Parse(time.RFC3339Nano, createTime); errParse == nil {
			p.UnixTimestamp = parsedTime.Unix()
			outRoot["created"] = p.UnixTimestamp
		}
	}
	outRoot["created"] = p.UnixTimestamp

	if responseID, ok := jsonutil.String(responseRoot, "responseId"); ok {
		outRoot["id"] = responseID
	}

	finishReason, _ := jsonutil.String(responseRoot, "stop_reason")
	if finishReason == "" {
		finishReason, _ = jsonutil.String(responseRoot, "candidates.0.finishReason")
	}
	finishReason = strings.ToLower(finishReason)

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

	hasFunctionCall := false
	if parts, ok := jsonutil.Array(responseRoot, "candidates.0.content.parts"); ok {
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}

			text, hasText := jsonutil.String(part, "text")
			functionCall, hasFunctionCallPart := jsonutil.Object(part, "functionCall")
			inlineData, hasInlineData := jsonutil.Object(part, "inlineData")
			if !hasInlineData {
				inlineData, hasInlineData = jsonutil.Object(part, "inline_data")
			}
			thoughtSignature, hasThoughtSignature := jsonutil.String(part, "thoughtSignature")
			if !hasThoughtSignature {
				thoughtSignature, hasThoughtSignature = jsonutil.String(part, "thought_signature")
			}

			hasContentPayload := hasText || hasFunctionCallPart || hasInlineData
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

			if hasFunctionCallPart {
				hasFunctionCall = true
				functionCallIndex := p.FunctionIndex
				p.FunctionIndex++

				toolCalls, _ := delta["tool_calls"].([]any)
				if toolCalls != nil {
					functionCallIndex = len(toolCalls)
				}

				functionCallEntry := map[string]any{
					"id":    fmt.Sprintf("%s-%d-%d", cliOpenAIString(functionCall["name"]), time.Now().UnixNano(), atomic.AddUint64(&functionCallIDCounter, 1)),
					"index": functionCallIndex,
					"type":  "function",
					"function": map[string]any{
						"name":      cliOpenAIString(functionCall["name"]),
						"arguments": "",
					},
				}
				if argsValue, ok := functionCall["args"]; ok {
					functionCallEntry["function"].(map[string]any)["arguments"] = string(jsonutil.MarshalOrOriginal(nil, argsValue))
				}

				delta["role"] = "assistant"
				if delta["tool_calls"] == nil {
					delta["tool_calls"] = []any{}
				}
				delta["tool_calls"] = append(delta["tool_calls"].([]any), functionCallEntry)
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

	if hasFunctionCall {
		choice["finish_reason"] = "tool_calls"
		choice["native_finish_reason"] = "tool_calls"
	} else if finishReason != "" && p.FunctionIndex == 0 {
		if finishReason == "max_tokens" || finishReason == "stop" {
			choice["finish_reason"] = finishReason
			choice["native_finish_reason"] = finishReason
		}
	}

	return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}
}

// ConvertCliResponseToOpenAINonStream converts a non-streaming Gemini CLI
// response to a non-streaming OpenAI response.
func ConvertCliResponseToOpenAINonStream(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return ""
	}

	responseBytes, errMarshal := json.Marshal(responseRoot)
	if errMarshal != nil {
		log.Warnf("gemini-cli openai response: failed to marshal response body: %v", errMarshal)
		return ""
	}
	return ConvertGeminiResponseToOpenAINonStream(ctx, modelName, originalRequestRawJSON, requestRawJSON, responseBytes, param)
}

func cliOpenAIString(value any) string {
	if value == nil {
		return ""
	}
	switch typed := value.(type) {
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
