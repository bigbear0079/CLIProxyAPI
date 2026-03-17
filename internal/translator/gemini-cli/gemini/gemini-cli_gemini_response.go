// Package gemini provides response translation functionality for Gemini CLI to
// Gemini API compatibility using standard JSON trees.
package gemini

import (
	"bytes"
	"context"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertGeminiCliResponseToGemini converts Gemini CLI responses to Gemini
// responses.
func ConvertGeminiCliResponseToGemini(ctx context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) []string {
	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}

	alt, ok := ctx.Value("alt").(string)
	if !ok {
		return []string{}
	}

	if alt == "" {
		root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
		if response, ok := jsonutil.Get(root, "response"); ok {
			return []string{string(jsonutil.MarshalOrOriginal(rawJSON, response))}
		}
		return []string{""}
	}

	items, errParse := jsonutil.ParseArrayBytes(rawJSON)
	if errParse != nil {
		return []string{string(rawJSON)}
	}

	responses := make([]any, 0, len(items))
	for _, itemValue := range items {
		item, ok := itemValue.(map[string]any)
		if !ok {
			continue
		}
		if response, ok := jsonutil.Get(item, "response"); ok {
			responses = append(responses, response)
		}
	}
	return []string{string(jsonutil.MarshalOrOriginal(rawJSON, responses))}
}

// ConvertGeminiCliResponseToGeminiNonStream converts a non-streaming Gemini CLI
// response to a non-streaming Gemini response.
func ConvertGeminiCliResponseToGeminiNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	root := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	if response, ok := jsonutil.Get(root, "response"); ok {
		return string(jsonutil.MarshalOrOriginal(rawJSON, response))
	}
	return string(rawJSON)
}

func GeminiTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"totalTokens":%d,"promptTokensDetails":[{"modality":"TEXT","tokenCount":%d}]}`, count, count)
}
