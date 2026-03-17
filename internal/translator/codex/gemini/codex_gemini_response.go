// Package gemini provides response translation functionality for Codex to
// Gemini API compatibility using standard JSON trees.
package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

var (
	dataTag = []byte("data:")
)

// ConvertCodexResponseToGeminiParams holds parameters for response conversion.
type ConvertCodexResponseToGeminiParams struct {
	Model                 string
	CreatedAt             int64
	ResponseID            string
	LastStorageOutput     string
	LastStorageOutputRoot map[string]any
}

// ConvertCodexResponseToGemini converts Codex streaming response format to
// Gemini format.
func ConvertCodexResponseToGemini(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &ConvertCodexResponseToGeminiParams{
			Model:                 modelName,
			CreatedAt:             0,
			ResponseID:            "",
			LastStorageOutput:     "",
			LastStorageOutputRoot: nil,
		}
	}
	p := (*param).(*ConvertCodexResponseToGeminiParams)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	typeStr, _ := jsonutil.String(root, "type")

	responseRoot, _ := jsonutil.Object(root, "response")
	if createdAt, ok := jsonutil.Int64(responseRoot, "created_at"); ok {
		p.CreatedAt = createdAt
	}

	baseResponse := p.LastStorageOutputRoot
	if baseResponse == nil || typeStr != "response.output_item.done" {
		baseResponse = codexGeminiBaseResponse(p)
	}

	switch typeStr {
	case "response.output_item.done":
		item, ok := jsonutil.Object(root, "item")
		if !ok {
			return []string{}
		}
		itemType, _ := jsonutil.String(item, "type")
		if itemType != "function_call" {
			return []string{}
		}

		part := codexGeminiFunctionCallPart(item, buildReverseMapFromGeminiOriginal(originalRequestRawJSON))
		parts, _ := jsonutil.Array(baseResponse, "candidates.0.content.parts")
		baseResponse["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = append(parts, part)
		baseResponse["candidates"].([]any)[0].(map[string]any)["finishReason"] = "STOP"

		p.LastStorageOutputRoot = baseResponse
		p.LastStorageOutput = string(jsonutil.MarshalOrOriginal(nil, baseResponse))
		return []string{}

	case "response.created":
		if responseModel, ok := jsonutil.String(root, "response.model"); ok && responseModel != "" {
			baseResponse["modelVersion"] = responseModel
		}
		if responseID, ok := jsonutil.String(root, "response.id"); ok && responseID != "" {
			baseResponse["responseId"] = responseID
			p.ResponseID = responseID
		}
		return []string{string(jsonutil.MarshalOrOriginal(nil, baseResponse))}

	case "response.reasoning_summary_text.delta":
		baseResponse["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = []any{
			map[string]any{
				"thought": true,
				"text":    codexGeminiString(root["delta"]),
			},
		}
		return []string{string(jsonutil.MarshalOrOriginal(nil, baseResponse))}

	case "response.output_text.delta":
		baseResponse["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = []any{
			map[string]any{
				"text": codexGeminiString(root["delta"]),
			},
		}
		return []string{string(jsonutil.MarshalOrOriginal(nil, baseResponse))}

	case "response.completed":
		usageMetadata := codexGeminiUsageMetadata(responseRoot)
		if len(usageMetadata) > 1 {
			baseResponse["usageMetadata"] = usageMetadata
		}

		completedChunk := string(jsonutil.MarshalOrOriginal(nil, baseResponse))
		if p.LastStorageOutput != "" {
			return []string{p.LastStorageOutput, completedChunk}
		}
		return []string{completedChunk}

	default:
		return []string{}
	}
}

// ConvertCodexResponseToGeminiNonStream converts a non-streaming Codex response
// to a non-streaming Gemini response.
func ConvertCodexResponseToGeminiNonStream(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	if responseType, _ := jsonutil.String(root, "type"); responseType != "response.completed" {
		return ""
	}

	responseRoot, ok := jsonutil.Object(root, "response")
	if !ok {
		return ""
	}

	outRoot := map[string]any{
		"candidates": []any{
			map[string]any{
				"content": map[string]any{
					"role":  "model",
					"parts": []any{},
				},
				"finishReason": "STOP",
			},
		},
		"usageMetadata": map[string]any{
			"trafficType": "PROVISIONED_THROUGHPUT",
		},
		"modelVersion": modelName,
		"createTime":   "",
		"responseId":   "",
	}

	if responseID, ok := jsonutil.String(responseRoot, "id"); ok {
		outRoot["responseId"] = responseID
	}
	if createdAt, ok := jsonutil.Int64(responseRoot, "created_at"); ok {
		outRoot["createTime"] = time.Unix(createdAt, 0).Format(time.RFC3339Nano)
	}
	outRoot["usageMetadata"] = codexGeminiUsageMetadata(responseRoot)

	parts := make([]any, 0)
	reverseMap := buildReverseMapFromGeminiOriginal(originalRequestRawJSON)

	if output, ok := jsonutil.Array(responseRoot, "output"); ok {
		for _, itemValue := range output {
			item, ok := itemValue.(map[string]any)
			if !ok {
				continue
			}

			itemType, _ := jsonutil.String(item, "type")
			switch itemType {
			case "reasoning":
				if text := codexGeminiReasoningText(item); text != "" {
					parts = append(parts, map[string]any{
						"text":    text,
						"thought": true,
					})
				}

			case "message":
				if contentParts, ok := jsonutil.Array(item, "content"); ok {
					for _, contentValue := range contentParts {
						content, ok := contentValue.(map[string]any)
						if !ok {
							continue
						}
						if contentType, _ := jsonutil.String(content, "type"); contentType == "output_text" {
							if text, ok := jsonutil.String(content, "text"); ok && text != "" {
								parts = append(parts, map[string]any{"text": text})
							}
						}
					}
				}

			case "function_call":
				parts = append(parts, codexGeminiFunctionCallPart(item, reverseMap))
			}
		}
	}

	outRoot["candidates"].([]any)[0].(map[string]any)["content"].(map[string]any)["parts"] = parts
	return string(jsonutil.MarshalOrOriginal(rawJSON, outRoot))
}

func codexGeminiBaseResponse(param *ConvertCodexResponseToGeminiParams) map[string]any {
	response := map[string]any{
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
		"modelVersion": "gemini-2.5-pro",
		"createTime":   "2025-08-15T02:52:03.884209Z",
		"responseId":   param.ResponseID,
	}

	if param.Model != "" {
		response["modelVersion"] = param.Model
	}
	if param.CreatedAt > 0 {
		response["createTime"] = time.Unix(param.CreatedAt, 0).Format(time.RFC3339Nano)
	}
	if param.ResponseID != "" {
		response["responseId"] = param.ResponseID
	}

	return response
}

func codexGeminiFunctionCallPart(item map[string]any, reverseMap map[string]string) map[string]any {
	name, _ := jsonutil.String(item, "name")
	if originalName, ok := reverseMap[name]; ok {
		name = originalName
	}

	argsObject := map[string]any{}
	if arguments, ok := jsonutil.String(item, "arguments"); ok && arguments != "" {
		if parsedArgs, ok := codexGeminiParseJSONObject(arguments); ok {
			argsObject = parsedArgs
		}
	}

	return map[string]any{
		"functionCall": map[string]any{
			"name": name,
			"args": argsObject,
		},
	}
}

func codexGeminiReasoningText(item map[string]any) string {
	if summary, ok := jsonutil.Get(item, "summary"); ok {
		switch typed := summary.(type) {
		case []any:
			var builder bytes.Buffer
			for _, partValue := range typed {
				if part, ok := partValue.(map[string]any); ok {
					if text, ok := jsonutil.String(part, "text"); ok {
						builder.WriteString(text)
						continue
					}
				}
				builder.WriteString(codexGeminiString(partValue))
			}
			return builder.String()
		default:
			return codexGeminiString(typed)
		}
	}
	if content, ok := jsonutil.Get(item, "content"); ok {
		switch typed := content.(type) {
		case []any:
			var builder bytes.Buffer
			for _, partValue := range typed {
				if part, ok := partValue.(map[string]any); ok {
					if text, ok := jsonutil.String(part, "text"); ok {
						builder.WriteString(text)
						continue
					}
				}
				builder.WriteString(codexGeminiString(partValue))
			}
			return builder.String()
		default:
			return codexGeminiString(typed)
		}
	}
	return ""
}

func codexGeminiUsageMetadata(responseRoot map[string]any) map[string]any {
	usageMetadata := map[string]any{
		"trafficType": "PROVISIONED_THROUGHPUT",
	}

	if inputTokens, ok := jsonutil.Int64(responseRoot, "usage.input_tokens"); ok {
		usageMetadata["promptTokenCount"] = inputTokens
		if outputTokens, ok := jsonutil.Int64(responseRoot, "usage.output_tokens"); ok {
			usageMetadata["candidatesTokenCount"] = outputTokens
			usageMetadata["totalTokenCount"] = inputTokens + outputTokens
		}
	}

	return usageMetadata
}

func buildReverseMapFromGeminiOriginal(original []byte) map[string]string {
	root := jsonutil.ParseObjectBytesOrEmpty(original)
	names := make([]string, 0)
	if tools, ok := jsonutil.Array(root, "tools"); ok {
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			functionDeclarations, ok := jsonutil.Array(tool, "functionDeclarations")
			if !ok {
				functionDeclarations, ok = jsonutil.Array(tool, "function_declarations")
				if !ok {
					continue
				}
			}
			for _, declarationValue := range functionDeclarations {
				declaration, ok := declarationValue.(map[string]any)
				if !ok {
					continue
				}
				if name, ok := jsonutil.String(declaration, "name"); ok && name != "" {
					names = append(names, name)
				}
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

func codexGeminiParseJSONObject(raw string) (map[string]any, bool) {
	value, errParse := jsonutil.ParseAnyBytes([]byte(raw))
	if errParse != nil {
		return nil, false
	}
	object, ok := value.(map[string]any)
	return object, ok
}

func codexGeminiString(value any) string {
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

func GeminiTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"totalTokens":%d,"promptTokensDetails":[{"modality":"TEXT","tokenCount":%d}]}`, count, count)
}
