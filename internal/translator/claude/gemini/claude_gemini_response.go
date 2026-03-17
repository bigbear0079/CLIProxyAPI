// Package gemini provides response translation functionality for Claude Code
// to Gemini compatibility using standard JSON trees.
package gemini

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

var (
	dataTag = []byte("data:")
)

// ConvertAnthropicResponseToGeminiParams holds parameters for response
// conversion and minimal streaming state for assembling tool use deltas.
type ConvertAnthropicResponseToGeminiParams struct {
	Model             string
	CreatedAt         int64
	ResponseID        string
	LastStorageOutput string
	IsStreaming       bool
	ToolUseNames      map[int]string
	ToolUseArgs       map[int]*strings.Builder
}

// ConvertClaudeResponseToGemini converts Claude streaming response format to
// Gemini format.
func ConvertClaudeResponseToGemini(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	if *param == nil {
		*param = &ConvertAnthropicResponseToGeminiParams{
			Model:      modelName,
			CreatedAt:  0,
			ResponseID: "",
		}
	}
	p := (*param).(*ConvertAnthropicResponseToGeminiParams)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	eventType, _ := jsonutil.String(root, "type")
	if p.CreatedAt == 0 {
		p.CreatedAt = time.Now().Unix()
	}

	switch eventType {
	case "message_start":
		if message, ok := jsonutil.Object(root, "message"); ok {
			if responseID, ok := jsonutil.String(message, "id"); ok {
				p.ResponseID = responseID
			}
			if model, ok := jsonutil.String(message, "model"); ok {
				p.Model = model
			}
		}
		return []string{}

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
		if p.ToolUseNames == nil {
			p.ToolUseNames = map[int]string{}
		}
		if name, ok := jsonutil.String(contentBlock, "name"); ok {
			p.ToolUseNames[int(index64)] = name
		}
		return []string{}

	case "content_block_delta":
		delta, ok := jsonutil.Object(root, "delta")
		if !ok {
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, claudeGeminiBaseChunk(p)))}
		}
		deltaType, _ := jsonutil.String(delta, "type")
		switch deltaType {
		case "text_delta":
			text, ok := jsonutil.String(delta, "text")
			if !ok || text == "" {
				return []string{}
			}
			outRoot := claudeGeminiBaseChunk(p)
			claudeGeminiAppendPart(outRoot, map[string]any{"text": text})
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

		case "thinking_delta":
			text, ok := jsonutil.String(delta, "thinking")
			if !ok || text == "" {
				return []string{}
			}
			outRoot := claudeGeminiBaseChunk(p)
			claudeGeminiAppendPart(outRoot, map[string]any{
				"thought": true,
				"text":    text,
			})
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

		case "input_json_delta":
			index64, _ := jsonutil.Int64(root, "index")
			if p.ToolUseArgs == nil {
				p.ToolUseArgs = map[int]*strings.Builder{}
			}
			builder, ok := p.ToolUseArgs[int(index64)]
			if !ok || builder == nil {
				builder = &strings.Builder{}
				p.ToolUseArgs[int(index64)] = builder
			}
			if partialJSON, ok := jsonutil.String(delta, "partial_json"); ok {
				builder.WriteString(partialJSON)
			}
			return []string{}
		}
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, claudeGeminiBaseChunk(p)))}

	case "content_block_stop":
		index64, _ := jsonutil.Int64(root, "index")
		index := int(index64)
		name := ""
		if p.ToolUseNames != nil {
			name = p.ToolUseNames[index]
		}

		argsValue := map[string]any{}
		argsTrim := ""
		if p.ToolUseArgs != nil {
			if builder := p.ToolUseArgs[index]; builder != nil {
				argsTrim = strings.TrimSpace(builder.String())
			}
		}
		if argsTrim != "" {
			if parsedArgs, errParse := jsonutil.ParseAnyBytes([]byte(argsTrim)); errParse == nil {
				if parsedObject, ok := parsedArgs.(map[string]any); ok {
					argsValue = parsedObject
				}
			}
		}

		if name != "" || argsTrim != "" {
			outRoot := claudeGeminiBaseChunk(p)
			claudeGeminiAppendPart(outRoot, map[string]any{
				"functionCall": map[string]any{
					"name": name,
					"args": argsValue,
				},
			})
			outRoot["candidates"].([]any)[0].(map[string]any)["finishReason"] = "STOP"
			p.LastStorageOutput = string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
			if p.ToolUseArgs != nil {
				delete(p.ToolUseArgs, index)
			}
			if p.ToolUseNames != nil {
				delete(p.ToolUseNames, index)
			}
			return []string{p.LastStorageOutput}
		}
		return []string{}

	case "message_delta":
		outRoot := claudeGeminiBaseChunk(p)
		outRoot["candidates"].([]any)[0].(map[string]any)["finishReason"] = "STOP"
		if usage, ok := jsonutil.Object(root, "usage"); ok {
			outRoot["usageMetadata"] = claudeGeminiUsageFromAnthropicUsage(usage)
		}
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}

	case "message_stop":
		return []string{}

	case "error":
		message, _ := jsonutil.String(root, "error.message")
		if message == "" {
			message = "Unknown error occurred"
		}
		return []string{string(jsonutil.MarshalOrOriginal(rawJSON, map[string]any{
			"error": map[string]any{
				"code":    400,
				"message": message,
				"status":  "INVALID_ARGUMENT",
			},
		}))}

	default:
		return []string{}
	}
}

// ConvertClaudeResponseToGeminiNonStream converts a non-streaming Claude Code
// response to a non-streaming Gemini response.
func ConvertClaudeResponseToGeminiNonStream(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	outRoot := claudeGeminiBaseChunk(&ConvertAnthropicResponseToGeminiParams{Model: modelName})
	outRoot["candidates"].([]any)[0].(map[string]any)["finishReason"] = "STOP"

	streamingEvents := make([][]byte, 0)
	scanner := bufio.NewScanner(bytes.NewReader(rawJSON))
	buffer := make([]byte, 52_428_800)
	scanner.Buffer(buffer, 52_428_800)
	for scanner.Scan() {
		line := scanner.Bytes()
		if bytes.HasPrefix(line, dataTag) {
			streamingEvents = append(streamingEvents, bytes.TrimSpace(line[5:]))
		}
	}

	newParam := &ConvertAnthropicResponseToGeminiParams{
		Model:        modelName,
		CreatedAt:    0,
		ResponseID:   "",
		IsStreaming:  false,
		ToolUseNames: nil,
		ToolUseArgs:  nil,
	}

	allParts := make([]any, 0)
	var finalUsage map[string]any
	var responseID string
	var createdAt int64

	for _, eventData := range streamingEvents {
		if len(eventData) == 0 {
			continue
		}

		root := jsonutil.ParseObjectBytesOrEmpty(eventData)
		eventType, _ := jsonutil.String(root, "type")

		switch eventType {
		case "message_start":
			if message, ok := jsonutil.Object(root, "message"); ok {
				responseID, _ = jsonutil.String(message, "id")
				newParam.ResponseID = responseID
				if model, ok := jsonutil.String(message, "model"); ok {
					newParam.Model = model
				}
				createdAt = time.Now().Unix()
				newParam.CreatedAt = createdAt
			}

		case "content_block_start":
			index64, _ := jsonutil.Int64(root, "index")
			if contentBlock, ok := jsonutil.Object(root, "content_block"); ok {
				if blockType, _ := jsonutil.String(contentBlock, "type"); blockType == "tool_use" {
					if newParam.ToolUseNames == nil {
						newParam.ToolUseNames = map[int]string{}
					}
					if name, ok := jsonutil.String(contentBlock, "name"); ok {
						newParam.ToolUseNames[int(index64)] = name
					}
				}
			}

		case "content_block_delta":
			delta, ok := jsonutil.Object(root, "delta")
			if !ok {
				continue
			}
			deltaType, _ := jsonutil.String(delta, "type")
			switch deltaType {
			case "text_delta":
				if text, ok := jsonutil.String(delta, "text"); ok && text != "" {
					allParts = append(allParts, map[string]any{"text": text})
				}
			case "thinking_delta":
				if text, ok := jsonutil.String(delta, "thinking"); ok && text != "" {
					allParts = append(allParts, map[string]any{
						"thought": true,
						"text":    text,
					})
				}
			case "input_json_delta":
				index64, _ := jsonutil.Int64(root, "index")
				if newParam.ToolUseArgs == nil {
					newParam.ToolUseArgs = map[int]*strings.Builder{}
				}
				builder, ok := newParam.ToolUseArgs[int(index64)]
				if !ok || builder == nil {
					builder = &strings.Builder{}
					newParam.ToolUseArgs[int(index64)] = builder
				}
				if partialJSON, ok := jsonutil.String(delta, "partial_json"); ok {
					builder.WriteString(partialJSON)
				}
			}

		case "content_block_stop":
			index64, _ := jsonutil.Int64(root, "index")
			index := int(index64)
			name := ""
			if newParam.ToolUseNames != nil {
				name = newParam.ToolUseNames[index]
			}

			argsValue := map[string]any{}
			argsTrim := ""
			if newParam.ToolUseArgs != nil {
				if builder := newParam.ToolUseArgs[index]; builder != nil {
					argsTrim = strings.TrimSpace(builder.String())
				}
			}
			if argsTrim != "" {
				if parsedArgs, errParse := jsonutil.ParseAnyBytes([]byte(argsTrim)); errParse == nil {
					if parsedObject, ok := parsedArgs.(map[string]any); ok {
						argsValue = parsedObject
					}
				}
			}
			if name != "" || argsTrim != "" {
				allParts = append(allParts, map[string]any{
					"functionCall": map[string]any{
						"name": name,
						"args": argsValue,
					},
				})
				if newParam.ToolUseArgs != nil {
					delete(newParam.ToolUseArgs, index)
				}
				if newParam.ToolUseNames != nil {
					delete(newParam.ToolUseNames, index)
				}
			}

		case "message_delta":
			if usage, ok := jsonutil.Object(root, "usage"); ok {
				finalUsage = claudeGeminiUsageFromAnthropicUsage(usage)
			}
		}
	}

	if responseID != "" {
		outRoot["responseId"] = responseID
	}
	if createdAt > 0 {
		outRoot["createTime"] = time.Unix(createdAt, 0).Format(time.RFC3339Nano)
	}

	consolidatedParts := consolidateParts(allParts)
	if len(consolidatedParts) > 0 {
		outRoot["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = consolidatedParts
	}
	if finalUsage != nil {
		outRoot["usageMetadata"] = finalUsage
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func GeminiTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"totalTokens":%d,"promptTokensDetails":[{"modality":"TEXT","tokenCount":%d}]}`, count, count)
}

func consolidateParts(parts []any) []any {
	if len(parts) == 0 {
		return parts
	}

	consolidated := make([]any, 0, len(parts))
	var textBuilder strings.Builder
	var thoughtBuilder strings.Builder
	hasText := false
	hasThought := false

	flushText := func() {
		if hasText && textBuilder.Len() > 0 {
			consolidated = append(consolidated, map[string]any{"text": textBuilder.String()})
			textBuilder.Reset()
			hasText = false
		}
	}
	flushThought := func() {
		if hasThought && thoughtBuilder.Len() > 0 {
			consolidated = append(consolidated, map[string]any{
				"thought": true,
				"text":    thoughtBuilder.String(),
			})
			thoughtBuilder.Reset()
			hasThought = false
		}
	}

	for _, partValue := range parts {
		part, ok := partValue.(map[string]any)
		if !ok {
			flushText()
			flushThought()
			consolidated = append(consolidated, partValue)
			continue
		}

		if thought, ok := part["thought"].(bool); ok && thought {
			flushText()
			if text, ok := part["text"].(string); ok {
				thoughtBuilder.WriteString(text)
				hasThought = true
			}
			continue
		}
		if text, ok := part["text"].(string); ok {
			flushThought()
			textBuilder.WriteString(text)
			hasText = true
			continue
		}

		flushText()
		flushThought()
		consolidated = append(consolidated, part)
	}

	flushThought()
	flushText()

	return consolidated
}

func claudeGeminiBaseChunk(param *ConvertAnthropicResponseToGeminiParams) map[string]any {
	createTime := ""
	if param.CreatedAt > 0 {
		createTime = time.Unix(param.CreatedAt, 0).Format(time.RFC3339Nano)
	}
	return map[string]any{
		"candidates": []any{
			map[string]any{
				"content": map[string]any{
					"role":  "model",
					"parts": []any{},
				},
			},
		},
		"usageMetadata": map[string]any{
			"trafficType": "PROVISIONED_THROUGHPUT",
		},
		"modelVersion": param.Model,
		"createTime":   createTime,
		"responseId":   param.ResponseID,
	}
}

func claudeGeminiAppendPart(root map[string]any, part map[string]any) {
	root["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = append(
		root["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any),
		part,
	)
}

func claudeGeminiUsageFromAnthropicUsage(usage map[string]any) map[string]any {
	inputTokens, _ := jsonutil.Int64(usage, "input_tokens")
	outputTokens, _ := jsonutil.Int64(usage, "output_tokens")
	cacheCreationTokens, _ := jsonutil.Int64(usage, "cache_creation_input_tokens")
	cacheReadTokens, _ := jsonutil.Int64(usage, "cache_read_input_tokens")
	thinkingTokens, _ := jsonutil.Int64(usage, "thinking_tokens")

	usageRoot := map[string]any{
		"promptTokenCount":     inputTokens,
		"candidatesTokenCount": outputTokens,
		"totalTokenCount":      inputTokens + outputTokens,
		"trafficType":          "PROVISIONED_THROUGHPUT",
	}
	totalCachedTokens := cacheCreationTokens + cacheReadTokens
	if totalCachedTokens > 0 {
		usageRoot["cachedContentTokenCount"] = totalCachedTokens
	}
	if thinkingTokens > 0 {
		usageRoot["thoughtsTokenCount"] = thinkingTokens
	}
	return usageRoot
}
