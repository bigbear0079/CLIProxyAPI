// Package gemini_cli provides response translation functionality for Gemini API to Gemini CLI API.
// This package handles the conversion of Gemini API responses into Gemini CLI-compatible
// JSON format, transforming streaming events and non-streaming responses into the format
// expected by Gemini CLI API clients.
package geminiCLI

import (
	"bytes"
	"context"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

var dataTag = []byte("data:")

// ConvertGeminiResponseToGeminiCLI converts Gemini streaming response format to Gemini CLI single-line JSON format.
// This function processes various Gemini event types and transforms them into Gemini CLI-compatible JSON responses.
// It handles thinking content, regular text content, and function calls, outputting single-line JSON
// that matches the Gemini CLI API response format.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the Gemini API.
//   - param: A pointer to a parameter object for the conversion (unused).
//
// Returns:
//   - []string: A slice of strings, each containing a Gemini CLI-compatible JSON response.
func ConvertGeminiResponseToGeminiCLI(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) []string {
	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		return []string{}
	}
	responseValue, errParse := jsonutil.ParseAnyBytes(rawJSON)
	if errParse != nil {
		return []string{string(jsonutil.MarshalOrOriginal(nil, map[string]any{"response": string(rawJSON)}))}
	}
	return []string{string(jsonutil.MarshalOrOriginal(nil, map[string]any{"response": responseValue}))}
}

// ConvertGeminiResponseToGeminiCLINonStream converts a non-streaming Gemini response to a non-streaming Gemini CLI response.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the Gemini API.
//   - param: A pointer to a parameter object for the conversion (unused).
//
// Returns:
//   - string: A Gemini CLI-compatible JSON response.
func ConvertGeminiResponseToGeminiCLINonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	responseValue, errParse := jsonutil.ParseAnyBytes(rawJSON)
	if errParse != nil {
		return string(jsonutil.MarshalOrOriginal(nil, map[string]any{"response": string(rawJSON)}))
	}
	return string(jsonutil.MarshalOrOriginal(nil, map[string]any{"response": responseValue}))
}

func GeminiCLITokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"totalTokens":%d,"promptTokensDetails":[{"modality":"TEXT","tokenCount":%d}]}`, count, count)
}
