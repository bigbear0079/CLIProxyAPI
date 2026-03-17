// Package openai provides request translation functionality for OpenAI-compatible chat completions payloads.
// It normalizes request JSON using standard encoding/json-based helpers.
package chat_completions

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// ConvertOpenAIRequestToOpenAI converts an OpenAI Chat Completions request (raw JSON)
// into an OpenAI-compatible request with the normalized model field applied.
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the OpenAI API
//   - stream: A boolean indicating if the request is for a streaming response (unused in current implementation)
//
// Returns:
//   - []byte: The transformed request data in OpenAI-compatible format
func ConvertOpenAIRequestToOpenAI(modelName string, inputRawJSON []byte, _ bool) []byte {
	root, errParse := jsonutil.ParseObjectBytes(inputRawJSON)
	if errParse != nil {
		return inputRawJSON
	}
	root["model"] = modelName
	return jsonutil.MarshalOrOriginal(inputRawJSON, root)
}
