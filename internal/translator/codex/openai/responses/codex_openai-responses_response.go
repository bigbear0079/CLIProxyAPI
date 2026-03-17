package responses

import (
	"bytes"
	"context"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertCodexResponseToOpenAIResponses converts OpenAI Chat Completions streaming chunks
// to OpenAI Responses SSE events (response.*).

func ConvertCodexResponseToOpenAIResponses(_ context.Context, _ string, _, _, rawJSON []byte, _ *any) []string {
	if bytes.HasPrefix(rawJSON, []byte("data:")) {
		rawJSON = bytes.TrimSpace(rawJSON[5:])
		out := fmt.Sprintf("data: %s", string(rawJSON))
		return []string{out}
	}
	return []string{string(rawJSON)}
}

// ConvertCodexResponseToOpenAIResponsesNonStream builds a single Responses JSON
// from a non-streaming OpenAI Chat Completions response.
func ConvertCodexResponseToOpenAIResponsesNonStream(_ context.Context, _ string, _, _, rawJSON []byte, _ *any) string {
	rootResult, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return ""
	}
	if responseType, ok := jsonutil.String(rootResult, "type"); !ok || responseType != "response.completed" {
		return ""
	}
	responseResult, ok := jsonutil.Get(rootResult, "response")
	if !ok {
		return ""
	}
	return string(jsonutil.MarshalOrOriginal(nil, responseResult))
}
