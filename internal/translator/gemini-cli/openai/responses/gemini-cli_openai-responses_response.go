package responses

import (
	"context"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/openai/responses"
)

func ConvertGeminiCLIResponseToOpenAIResponses(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if responseResult, ok := jsonutil.Get(jsonutil.ParseObjectBytesOrEmpty(rawJSON), "response"); ok {
		rawJSON = jsonutil.MarshalOrOriginal(rawJSON, responseResult)
	}
	return ConvertGeminiResponseToOpenAIResponses(ctx, modelName, originalRequestRawJSON, requestRawJSON, rawJSON, param)
}

func ConvertGeminiCLIResponseToOpenAIResponsesNonStream(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) string {
	responseRoot := jsonutil.ParseObjectBytesOrEmpty(rawJSON)
	if responseResult, ok := jsonutil.Get(responseRoot, "response"); ok {
		rawJSON = jsonutil.MarshalOrOriginal(rawJSON, responseResult)
	}

	originalRequestRoot := jsonutil.ParseObjectBytesOrEmpty(originalRequestRawJSON)
	if _, ok := jsonutil.Get(responseRoot, "response"); ok {
		if requestResult, ok := jsonutil.Get(originalRequestRoot, "request"); ok {
			originalRequestRawJSON = jsonutil.MarshalOrOriginal(originalRequestRawJSON, requestResult)
		}
	}

	requestRoot := jsonutil.ParseObjectBytesOrEmpty(requestRawJSON)
	if _, ok := jsonutil.Get(responseRoot, "response"); ok {
		if requestResult, ok := jsonutil.Get(requestRoot, "request"); ok {
			requestRawJSON = jsonutil.MarshalOrOriginal(requestRawJSON, requestResult)
		}
	}

	return ConvertGeminiResponseToOpenAIResponsesNonStream(ctx, modelName, originalRequestRawJSON, requestRawJSON, rawJSON, param)
}
