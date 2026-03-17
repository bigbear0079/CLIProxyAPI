// Package openai provides response translation functionality for Gemini to
// OpenAI API compatibility using standard JSON trees.
package chat_completions

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// convertGeminiResponseToOpenAIChatParams holds parameters for response
// conversion.
type convertGeminiResponseToOpenAIChatParams struct {
	UnixTimestamp int64

	// FunctionIndex tracks tool call indices per candidate index to support
	// multiple candidates.
	FunctionIndex map[int]int
}

// functionCallIDCounter provides a process-wide unique counter for function
// call identifiers.
var functionCallIDCounter uint64

// ConvertGeminiResponseToOpenAI translates a single Gemini streaming chunk to
// the OpenAI Chat Completions streaming format.
func ConvertGeminiResponseToOpenAI(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	if *param == nil {
		*param = &convertGeminiResponseToOpenAIChatParams{
			UnixTimestamp: 0,
			FunctionIndex: make(map[int]int),
		}
	}
	p := (*param).(*convertGeminiResponseToOpenAIChatParams)
	if p.FunctionIndex == nil {
		p.FunctionIndex = make(map[int]int)
	}

	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}
	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		return []string{}
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)

	if createTime, ok := jsonutil.String(root, "createTime"); ok {
		if parsedTime, errParse := time.Parse(time.RFC3339Nano, createTime); errParse == nil {
			p.UnixTimestamp = parsedTime.Unix()
		}
	}

	baseChunk := geminiOpenAIBaseChunk(p)
	if modelVersion, ok := jsonutil.String(root, "modelVersion"); ok {
		baseChunk["model"] = modelVersion
	}
	if responseID, ok := jsonutil.String(root, "responseId"); ok {
		baseChunk["id"] = responseID
	}
	if usageMetadata, ok := jsonutil.Object(root, "usageMetadata"); ok {
		baseChunk["usage"] = geminiOpenAIUsage(usageMetadata)
	}

	finishReason := ""
	if stopReason, ok := jsonutil.String(root, "stop_reason"); ok {
		finishReason = strings.ToLower(stopReason)
	}
	if finishReason == "" {
		if firstCandidateFinishReason, ok := jsonutil.String(root, "candidates.0.finishReason"); ok {
			finishReason = strings.ToLower(firstCandidateFinishReason)
		}
	}

	candidates, ok := jsonutil.Array(root, "candidates")
	if !ok {
		if jsonutil.Exists(root, "usageMetadata") {
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, baseChunk))}
		}
		return []string{}
	}

	responseStrings := make([]string, 0, len(candidates))
	for _, candidateValue := range candidates {
		candidate, ok := candidateValue.(map[string]any)
		if !ok {
			continue
		}

		outRoot := geminiOpenAIBaseChunk(p)
		if modelValue, ok := baseChunk["model"]; ok {
			outRoot["model"] = modelValue
		}
		if idValue, ok := baseChunk["id"]; ok {
			outRoot["id"] = idValue
		}
		if usageValue, ok := baseChunk["usage"]; ok {
			outRoot["usage"] = usageValue
		}

		choice := outRoot["choices"].([]any)[0].(map[string]any)
		delta := choice["delta"].(map[string]any)
		candidateIndex64, _ := jsonutil.Int64(candidate, "index")
		candidateIndex := int(candidateIndex64)
		choice["index"] = candidateIndex

		hasFunctionCall := false
		if parts, ok := jsonutil.Array(candidate, "content.parts"); ok {
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

					functionCallIndex := p.FunctionIndex[candidateIndex]
					p.FunctionIndex[candidateIndex]++

					toolCalls, _ := delta["tool_calls"].([]any)
					if toolCalls != nil {
						functionCallIndex = len(toolCalls)
					}

					fcName, _ := jsonutil.String(functionCall, "name")
					toolCall := map[string]any{
						"id":    fmt.Sprintf("%s-%d-%d", fcName, time.Now().UnixNano(), atomic.AddUint64(&functionCallIDCounter, 1)),
						"index": functionCallIndex,
						"type":  "function",
						"function": map[string]any{
							"name":      fcName,
							"arguments": "",
						},
					}
					if argsValue, ok := functionCall["args"]; ok {
						toolCall["function"].(map[string]any)["arguments"] = string(jsonutil.MarshalOrOriginal(nil, argsValue))
					}

					delta["role"] = "assistant"
					delta["tool_calls"] = append(toolCalls, toolCall)
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

					images, _ := delta["images"].([]any)
					imagePayload := map[string]any{
						"index": len(images),
						"type":  "image_url",
						"image_url": map[string]any{
							"url": fmt.Sprintf("data:%s;base64,%s", mimeType, data),
						},
					}
					delta["role"] = "assistant"
					delta["images"] = append(images, imagePayload)
				}
			}
		}

		if hasFunctionCall {
			choice["finish_reason"] = "tool_calls"
			choice["native_finish_reason"] = "tool_calls"
		} else if finishReason != "" {
			if finishReason == "max_tokens" || finishReason == "stop" {
				choice["finish_reason"] = finishReason
				choice["native_finish_reason"] = finishReason
			}
		}

		responseStrings = append(responseStrings, string(jsonutil.MarshalOrOriginal(rawJSON, outRoot)))
	}

	return responseStrings
}

// ConvertGeminiResponseToOpenAINonStream converts a non-streaming Gemini
// response to a non-streaming OpenAI response.
func ConvertGeminiResponseToOpenAINonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)

	var unixTimestamp int64
	if createTime, ok := jsonutil.String(root, "createTime"); ok {
		if parsedTime, errParse := time.Parse(time.RFC3339Nano, createTime); errParse == nil {
			unixTimestamp = parsedTime.Unix()
		}
	}

	outRoot := map[string]any{
		"id":      "",
		"object":  "chat.completion",
		"created": unixTimestamp,
		"model":   "model",
		"choices": []any{},
	}
	if modelVersion, ok := jsonutil.String(root, "modelVersion"); ok {
		outRoot["model"] = modelVersion
	}
	if responseID, ok := jsonutil.String(root, "responseId"); ok {
		outRoot["id"] = responseID
	}
	if usageMetadata, ok := jsonutil.Object(root, "usageMetadata"); ok {
		outRoot["usage"] = geminiOpenAIUsage(usageMetadata)
	}

	candidates, ok := jsonutil.Array(root, "candidates")
	if !ok {
		return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
	}

	choices := make([]any, 0, len(candidates))
	for _, candidateValue := range candidates {
		candidate, ok := candidateValue.(map[string]any)
		if !ok {
			continue
		}

		candidateIndex64, _ := jsonutil.Int64(candidate, "index")
		choice := map[string]any{
			"index": candidateIndex64,
			"message": map[string]any{
				"role":              "assistant",
				"content":           nil,
				"reasoning_content": nil,
				"tool_calls":        nil,
			},
			"finish_reason":        nil,
			"native_finish_reason": nil,
		}
		message := choice["message"].(map[string]any)

		if finishReason, ok := jsonutil.String(candidate, "finishReason"); ok {
			lowerFinishReason := strings.ToLower(finishReason)
			choice["finish_reason"] = lowerFinishReason
			choice["native_finish_reason"] = lowerFinishReason
		}

		hasFunctionCall := false
		contentText := ""
		reasoningText := ""
		toolCalls := make([]any, 0)
		images := make([]any, 0)

		if parts, ok := jsonutil.Array(candidate, "content.parts"); ok {
			for _, partValue := range parts {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}

				if text, ok := jsonutil.String(part, "text"); ok {
					if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
						reasoningText += text
					} else {
						contentText += text
					}
					message["role"] = "assistant"
					continue
				}

				if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
					hasFunctionCall = true
					fcName, _ := jsonutil.String(functionCall, "name")
					toolCall := map[string]any{
						"id":   fmt.Sprintf("%s-%d-%d", fcName, time.Now().UnixNano(), atomic.AddUint64(&functionCallIDCounter, 1)),
						"type": "function",
						"function": map[string]any{
							"name":      fcName,
							"arguments": "",
						},
					}
					if argsValue, ok := functionCall["args"]; ok {
						toolCall["function"].(map[string]any)["arguments"] = string(jsonutil.MarshalOrOriginal(nil, argsValue))
					}
					message["role"] = "assistant"
					toolCalls = append(toolCalls, toolCall)
					continue
				}

				inlineData, ok := jsonutil.Object(part, "inlineData")
				if !ok {
					inlineData, ok = jsonutil.Object(part, "inline_data")
				}
				if !ok {
					continue
				}

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

				images = append(images, map[string]any{
					"index": len(images),
					"type":  "image_url",
					"image_url": map[string]any{
						"url": fmt.Sprintf("data:%s;base64,%s", mimeType, data),
					},
				})
				message["role"] = "assistant"
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
		if len(images) > 0 {
			message["images"] = images
		}
		if hasFunctionCall {
			choice["finish_reason"] = "tool_calls"
			choice["native_finish_reason"] = "tool_calls"
		}

		choices = append(choices, choice)
	}

	outRoot["choices"] = choices
	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func geminiOpenAIBaseChunk(param *convertGeminiResponseToOpenAIChatParams) map[string]any {
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

func geminiOpenAIUsage(usageMetadata map[string]any) map[string]any {
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
		usage["prompt_tokens_details"] = map[string]any{
			"cached_tokens": cachedTokenCount,
		}
	}
	return usage
}
