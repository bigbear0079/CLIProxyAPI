// Package gemini provides response translation functionality for OpenAI to
// Gemini compatibility using standard JSON trees.
package gemini

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertOpenAIResponseToGeminiParams holds parameters for response
// conversion.
type ConvertOpenAIResponseToGeminiParams struct {
	ToolCallsAccumulator map[int]*ToolCallAccumulator
	ContentAccumulator   strings.Builder
	IsFirstChunk         bool
}

// ToolCallAccumulator holds the state for accumulating tool call data.
type ToolCallAccumulator struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

// ConvertOpenAIResponseToGemini converts OpenAI Chat Completions streaming
// response format to Gemini API format.
func ConvertOpenAIResponseToGemini(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	if *param == nil {
		*param = &ConvertOpenAIResponseToGeminiParams{
			ToolCallsAccumulator: nil,
			ContentAccumulator:   strings.Builder{},
			IsFirstChunk:         false,
		}
	}
	p := (*param).(*ConvertOpenAIResponseToGeminiParams)
	if p.ToolCallsAccumulator == nil {
		p.ToolCallsAccumulator = make(map[int]*ToolCallAccumulator)
	}

	if strings.TrimSpace(string(rawJSON)) == "[DONE]" {
		return []string{}
	}
	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	model, _ := jsonutil.String(root, "model")

	choices, ok := jsonutil.Array(root, "choices")
	if !ok {
		return []string{}
	}

	if len(choices) == 0 {
		if usage, ok := jsonutil.Object(root, "usage"); ok {
			outRoot := map[string]any{
				"candidates":    []any{},
				"usageMetadata": openAIGeminiUsageMetadata(usage),
				"model":         model,
			}
			if model == "" {
				delete(outRoot, "model")
			}
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))}
		}
		return []string{}
	}

	results := make([]string, 0)
	for _, choiceValue := range choices {
		choice, ok := choiceValue.(map[string]any)
		if !ok {
			continue
		}

		delta, _ := jsonutil.Object(choice, "delta")

		if role, ok := jsonutil.String(delta, "role"); ok && p.IsFirstChunk {
			if role == "assistant" {
				outRoot := openAIGeminiBaseChunk(model)
				outRoot["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["role"] = "model"
				p.IsFirstChunk = false
				results = append(results, string(jsonutil.MarshalOrOriginal(rawJSON, outRoot)))
			}
			continue
		}

		chunkOutputs := make([]string, 0)
		if reasoningValue, ok := jsonutil.Get(delta, "reasoning_content"); ok {
			for _, reasoningText := range extractReasoningTexts(reasoningValue) {
				if reasoningText == "" {
					continue
				}
				outRoot := openAIGeminiBaseChunk(model)
				openAIGeminiAppendPart(outRoot, map[string]any{
					"thought": true,
					"text":    reasoningText,
				})
				chunkOutputs = append(chunkOutputs, string(jsonutil.MarshalOrOriginal(rawJSON, outRoot)))
			}
		}

		if content, ok := jsonutil.String(delta, "content"); ok && content != "" {
			p.ContentAccumulator.WriteString(content)
			outRoot := openAIGeminiBaseChunk(model)
			openAIGeminiAppendPart(outRoot, map[string]any{"text": content})
			chunkOutputs = append(chunkOutputs, string(jsonutil.MarshalOrOriginal(rawJSON, outRoot)))
		}
		if len(chunkOutputs) > 0 {
			results = append(results, chunkOutputs...)
			continue
		}

		if toolCalls, ok := jsonutil.Array(delta, "tool_calls"); ok {
			for _, toolCallValue := range toolCalls {
				toolCall, ok := toolCallValue.(map[string]any)
				if !ok {
					continue
				}

				toolIndex64, _ := jsonutil.Int64(toolCall, "index")
				toolIndex := int(toolIndex64)
				toolID, _ := jsonutil.String(toolCall, "id")
				toolType, _ := jsonutil.String(toolCall, "type")
				function, hasFunction := jsonutil.Object(toolCall, "function")
				if toolType != "" && toolType != "function" {
					continue
				}
				if !hasFunction {
					continue
				}

				functionName, _ := jsonutil.String(function, "name")
				functionArgs, _ := jsonutil.String(function, "arguments")
				if _, exists := p.ToolCallsAccumulator[toolIndex]; !exists {
					p.ToolCallsAccumulator[toolIndex] = &ToolCallAccumulator{
						ID:   toolID,
						Name: functionName,
					}
				}

				accumulator := p.ToolCallsAccumulator[toolIndex]
				if toolID != "" {
					accumulator.ID = toolID
				}
				if functionName != "" {
					accumulator.Name = functionName
				}
				if functionArgs != "" {
					accumulator.Arguments.WriteString(functionArgs)
				}
			}
			continue
		}

		if finishReason, ok := jsonutil.String(choice, "finish_reason"); ok {
			outRoot := openAIGeminiBaseChunk(model)
			outRoot["candidates"].([]any)[0].(map[string]any)["finishReason"] = mapOpenAIFinishReasonToGemini(finishReason)

			if len(p.ToolCallsAccumulator) > 0 {
				keys := make([]int, 0, len(p.ToolCallsAccumulator))
				for index := range p.ToolCallsAccumulator {
					keys = append(keys, index)
				}
				sort.Ints(keys)
				for _, index := range keys {
					accumulator := p.ToolCallsAccumulator[index]
					openAIGeminiAppendPart(outRoot, map[string]any{
						"functionCall": map[string]any{
							"name": accumulator.Name,
							"args": parseArgsToObject(accumulator.Arguments.String()),
						},
					})
				}
				p.ToolCallsAccumulator = make(map[int]*ToolCallAccumulator)
			}

			results = append(results, string(jsonutil.MarshalOrOriginal(rawJSON, outRoot)))
			continue
		}

		if usage, ok := jsonutil.Object(root, "usage"); ok {
			outRoot := openAIGeminiBaseChunk(model)
			outRoot["usageMetadata"] = openAIGeminiUsageMetadata(usage)
			results = append(results, string(jsonutil.MarshalOrOriginal(rawJSON, outRoot)))
		}
	}

	return results
}

// mapOpenAIFinishReasonToGemini maps OpenAI finish reasons to Gemini finish
// reasons.
func mapOpenAIFinishReasonToGemini(openAIReason string) string {
	switch openAIReason {
	case "stop":
		return "STOP"
	case "length":
		return "MAX_TOKENS"
	case "tool_calls":
		return "STOP"
	case "content_filter":
		return "SAFETY"
	default:
		return "STOP"
	}
}

func parseArgsToObject(argsStr string) map[string]any {
	trimmed := strings.TrimSpace(argsStr)
	if trimmed == "" || trimmed == "{}" {
		return map[string]any{}
	}

	if parsedArgs, errParse := jsonutil.ParseAnyBytes([]byte(trimmed)); errParse == nil {
		if parsedObject, ok := parsedArgs.(map[string]any); ok {
			return parsedObject
		}
	}

	if tolerant := tolerantParseJSONObject(trimmed); len(tolerant) > 0 {
		return tolerant
	}

	return map[string]any{}
}

// tolerantParseJSONObject attempts to parse a JSON-like object string,
// tolerating bareword values commonly seen during streamed tool calls.
func tolerantParseJSONObject(s string) map[string]any {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start == -1 || end == -1 || start >= end {
		return map[string]any{}
	}
	content := s[start+1 : end]

	runes := []rune(content)
	n := len(runes)
	i := 0
	result := make(map[string]any)

	for i < n {
		for i < n && (runes[i] == ' ' || runes[i] == '\n' || runes[i] == '\r' || runes[i] == '\t' || runes[i] == ',') {
			i++
		}
		if i >= n {
			break
		}
		if runes[i] != '"' {
			for i < n && runes[i] != ',' {
				i++
			}
			continue
		}

		keyToken, nextIdx := parseJSONStringRunes(runes, i)
		if nextIdx == -1 {
			break
		}
		keyName := jsonStringTokenToRawString(keyToken)
		i = nextIdx

		for i < n && (runes[i] == ' ' || runes[i] == '\n' || runes[i] == '\r' || runes[i] == '\t') {
			i++
		}
		if i >= n || runes[i] != ':' {
			break
		}
		i++
		for i < n && (runes[i] == ' ' || runes[i] == '\n' || runes[i] == '\r' || runes[i] == '\t') {
			i++
		}
		if i >= n {
			break
		}

		switch runes[i] {
		case '"':
			valueToken, ni := parseJSONStringRunes(runes, i)
			if ni == -1 {
				result[keyName] = ""
				i = n
			} else {
				result[keyName] = jsonStringTokenToRawString(valueToken)
				i = ni
			}
		case '{', '[':
			segment, ni := captureBracketed(runes, i)
			if ni == -1 {
				i = n
			} else {
				if parsedValue, errParse := jsonutil.ParseAnyBytes([]byte(segment)); errParse == nil {
					result[keyName] = parsedValue
				} else {
					result[keyName] = segment
				}
				i = ni
			}
		default:
			j := i
			for j < n && runes[j] != ',' {
				j++
			}
			token := strings.TrimSpace(string(runes[i:j]))
			if token == "true" {
				result[keyName] = true
			} else if token == "false" {
				result[keyName] = false
			} else if token == "null" {
				result[keyName] = nil
			} else if numVal, ok := tryParseNumber(token); ok {
				result[keyName] = numVal
			} else {
				result[keyName] = token
			}
			i = j
		}

		for i < n && (runes[i] == ' ' || runes[i] == '\n' || runes[i] == '\r' || runes[i] == '\t') {
			i++
		}
		if i < n && runes[i] == ',' {
			i++
		}
	}

	return result
}

func parseJSONStringRunes(runes []rune, start int) (string, int) {
	if start >= len(runes) || runes[start] != '"' {
		return "", -1
	}
	i := start + 1
	escaped := false
	for i < len(runes) {
		r := runes[i]
		if r == '\\' && !escaped {
			escaped = true
			i++
			continue
		}
		if r == '"' && !escaped {
			return string(runes[start : i+1]), i + 1
		}
		escaped = false
		i++
	}
	return string(runes[start:]), -1
}

func jsonStringTokenToRawString(token string) string {
	parsed, errParse := jsonutil.ParseAnyBytes([]byte(token))
	if errParse == nil {
		if value, ok := parsed.(string); ok {
			return value
		}
	}
	if len(token) >= 2 && token[0] == '"' && token[len(token)-1] == '"' {
		return token[1 : len(token)-1]
	}
	return token
}

func captureBracketed(runes []rune, i int) (string, int) {
	if i >= len(runes) {
		return "", -1
	}
	startRune := runes[i]
	var endRune rune
	if startRune == '{' {
		endRune = '}'
	} else if startRune == '[' {
		endRune = ']'
	} else {
		return "", -1
	}

	depth := 0
	j := i
	inString := false
	escaped := false
	for j < len(runes) {
		r := runes[j]
		if inString {
			if r == '\\' && !escaped {
				escaped = true
				j++
				continue
			}
			if r == '"' && !escaped {
				inString = false
			} else {
				escaped = false
			}
			j++
			continue
		}
		if r == '"' {
			inString = true
			j++
			continue
		}
		if r == startRune {
			depth++
		} else if r == endRune {
			depth--
			if depth == 0 {
				return string(runes[i : j+1]), j + 1
			}
		}
		j++
	}

	return string(runes[i:]), -1
}

func tryParseNumber(s string) (any, bool) {
	if s == "" {
		return nil, false
	}
	if i64, errParse := strconv.ParseInt(s, 10, 64); errParse == nil {
		return i64, true
	}
	if u64, errParse := strconv.ParseUint(s, 10, 64); errParse == nil {
		return u64, true
	}
	if f64, errParse := strconv.ParseFloat(s, 64); errParse == nil {
		return f64, true
	}
	return nil, false
}

// ConvertOpenAIResponseToGeminiNonStream converts a non-streaming OpenAI
// response to a non-streaming Gemini response.
func ConvertOpenAIResponseToGeminiNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	_ = originalRequestRawJSON
	_ = requestRawJSON

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	model, _ := jsonutil.String(root, "model")
	outRoot := openAIGeminiBaseChunk(model)

	if choices, ok := jsonutil.Array(root, "choices"); ok {
		for _, choiceValue := range choices {
			choice, ok := choiceValue.(map[string]any)
			if !ok {
				continue
			}

			candidate := map[string]any{
				"content": map[string]any{
					"parts": []any{},
					"role":  "model",
				},
				"index": int64(0),
			}
			if choiceIndex, ok := jsonutil.Int64(choice, "index"); ok {
				candidate["index"] = choiceIndex
			}

			message, _ := jsonutil.Object(choice, "message")
			if role, ok := jsonutil.String(message, "role"); ok && role == "assistant" {
				candidate["content"].(map[string]any)["role"] = "model"
			}

			if reasoningValue, ok := jsonutil.Get(message, "reasoning_content"); ok {
				for _, reasoningText := range extractReasoningTexts(reasoningValue) {
					if reasoningText == "" {
						continue
					}
					candidate["content"].(map[string]any)["parts"] = append(
						candidate["content"].(map[string]any)["parts"].([]any),
						map[string]any{
							"thought": true,
							"text":    reasoningText,
						},
					)
				}
			}

			if content, ok := jsonutil.String(message, "content"); ok && content != "" {
				candidate["content"].(map[string]any)["parts"] = append(
					candidate["content"].(map[string]any)["parts"].([]any),
					map[string]any{"text": content},
				)
			}

			if toolCalls, ok := jsonutil.Array(message, "tool_calls"); ok {
				for _, toolCallValue := range toolCalls {
					toolCall, ok := toolCallValue.(map[string]any)
					if !ok {
						continue
					}
					toolType, _ := jsonutil.String(toolCall, "type")
					if toolType != "function" {
						continue
					}

					function, ok := jsonutil.Object(toolCall, "function")
					if !ok {
						continue
					}
					functionName, _ := jsonutil.String(function, "name")
					functionArgs, _ := jsonutil.String(function, "arguments")
					candidate["content"].(map[string]any)["parts"] = append(
						candidate["content"].(map[string]any)["parts"].([]any),
						map[string]any{
							"functionCall": map[string]any{
								"name": functionName,
								"args": parseArgsToObject(functionArgs),
							},
						},
					)
				}
			}

			if finishReason, ok := jsonutil.String(choice, "finish_reason"); ok {
				candidate["finishReason"] = mapOpenAIFinishReasonToGemini(finishReason)
			}

			outRoot["candidates"] = []any{candidate}
		}
	}

	if usage, ok := jsonutil.Object(root, "usage"); ok {
		outRoot["usageMetadata"] = openAIGeminiUsageMetadata(usage)
	}

	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func GeminiTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"totalTokens":%d,"promptTokensDetails":[{"modality":"TEXT","tokenCount":%d}]}`, count, count)
}

func reasoningTokensFromUsage(usage map[string]any) int64 {
	if reasoningTokens, ok := jsonutil.Int64(usage, "completion_tokens_details.reasoning_tokens"); ok {
		return reasoningTokens
	}
	if reasoningTokens, ok := jsonutil.Int64(usage, "output_tokens_details.reasoning_tokens"); ok {
		return reasoningTokens
	}
	return 0
}

func extractReasoningTexts(node any) []string {
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
			texts = append(texts, extractReasoningTexts(value)...)
		}
	case map[string]any:
		if text, ok := jsonutil.String(typed, "text"); ok && text != "" {
			texts = append(texts, text)
		}
	}

	return texts
}

func openAIGeminiBaseChunk(model string) map[string]any {
	root := map[string]any{
		"candidates": []any{
			map[string]any{
				"content": map[string]any{
					"parts": []any{},
					"role":  "model",
				},
				"index": int64(0),
			},
		},
	}
	if model != "" {
		root["model"] = model
	}
	return root
}

func openAIGeminiAppendPart(root map[string]any, part map[string]any) {
	root["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = append(
		root["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"].([]any),
		part,
	)
}

func openAIGeminiUsageMetadata(usage map[string]any) map[string]any {
	usageMetadata := map[string]any{}
	if promptTokens, ok := jsonutil.Int64(usage, "prompt_tokens"); ok {
		usageMetadata["promptTokenCount"] = promptTokens
	}
	if completionTokens, ok := jsonutil.Int64(usage, "completion_tokens"); ok {
		usageMetadata["candidatesTokenCount"] = completionTokens
	}
	if totalTokens, ok := jsonutil.Int64(usage, "total_tokens"); ok {
		usageMetadata["totalTokenCount"] = totalTokens
	}
	if reasoningTokens := reasoningTokensFromUsage(usage); reasoningTokens > 0 {
		usageMetadata["thoughtsTokenCount"] = reasoningTokens
	}
	return usageMetadata
}
