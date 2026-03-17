// Package gemini provides request translation functionality for Gemini to Gemini CLI API compatibility.
// It handles parsing and transforming Gemini API requests into Gemini CLI API format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between Gemini API format and Gemini CLI API's expected format.
package gemini

import (
	"bytes"
	"context"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertAntigravityResponseToGemini parses and transforms a Gemini CLI API request into Gemini API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the Gemini API.
// The function performs the following transformations:
// 1. Extracts the response data from the request
// 2. Handles alternative response formats
// 3. Processes array responses by extracting individual response objects
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model to use for the request (unused in current implementation)
//   - rawJSON: The raw JSON request data from the Gemini CLI API
//   - param: A pointer to a parameter object for the conversion (unused in current implementation)
//
// Returns:
//   - []string: The transformed request data in Gemini API format
func ConvertAntigravityResponseToGemini(ctx context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) []string {
	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
	}

	if alt, ok := ctx.Value("alt").(string); ok {
		var chunk []byte
		if alt == "" {
			if responseResult, ok := jsonutil.Get(jsonutil.ParseObjectBytesOrEmpty(rawJSON), "response"); ok {
				chunk = jsonutil.MarshalOrOriginal(rawJSON, responseResult)
				chunk = restoreUsageMetadata(chunk)
			}
		} else {
			chunk = []byte("[]")
		}
		return []string{string(chunk)}
	}
	return []string{}
}

// ConvertAntigravityResponseToGeminiNonStream converts a non-streaming Gemini CLI request to a non-streaming Gemini response.
// This function processes the complete Gemini CLI request and transforms it into a single Gemini-compatible
// JSON response. It extracts the response data from the request and returns it in the expected format.
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model being used for the response (unused in current implementation)
//   - rawJSON: The raw JSON request data from the Gemini CLI API
//   - param: A pointer to a parameter object for the conversion (unused in current implementation)
//
// Returns:
//   - string: A Gemini-compatible JSON response containing the response data
func ConvertAntigravityResponseToGeminiNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	if responseResult, ok := jsonutil.Get(jsonutil.ParseObjectBytesOrEmpty(rawJSON), "response"); ok {
		chunk := restoreUsageMetadata(jsonutil.MarshalOrOriginal(rawJSON, responseResult))
		return string(chunk)
	}
	return string(rawJSON)
}

func GeminiTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"totalTokens":%d,"promptTokensDetails":[{"modality":"TEXT","tokenCount":%d}]}`, count, count)
}

// restoreUsageMetadata renames cpaUsageMetadata back to usageMetadata.
// The executor renames usageMetadata to cpaUsageMetadata in non-terminal chunks
// to preserve usage data while hiding it from clients that don't expect it.
// When returning standard Gemini API format, we must restore the original name.
func restoreUsageMetadata(chunk []byte) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(chunk)
	if cpaUsage, ok := jsonutil.Get(root, "cpaUsageMetadata"); ok {
		root["usageMetadata"] = cpaUsage
		delete(root, "cpaUsageMetadata")
		return jsonutil.MarshalOrOriginal(chunk, root)
	}
	return chunk
}
