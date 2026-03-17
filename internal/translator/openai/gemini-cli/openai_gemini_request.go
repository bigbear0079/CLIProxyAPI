// Package geminiCLI provides request translation functionality for Gemini to OpenAI API.
// It handles parsing and transforming Gemini API requests into OpenAI Chat Completions API format,
// extracting model information, generation config, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between Gemini API format and OpenAI API's expected format.
package geminiCLI

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	. "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/openai/gemini"
)

// ConvertGeminiCLIRequestToOpenAI parses and transforms a Gemini API request into OpenAI Chat Completions API format.
// It extracts the model name, generation config, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the OpenAI API.
func ConvertGeminiCLIRequestToOpenAI(modelName string, inputRawJSON []byte, stream bool) []byte {
	root, errParse := jsonutil.ParseObjectBytes(inputRawJSON)
	if errParse != nil {
		return ConvertGeminiRequestToOpenAI(modelName, inputRawJSON, stream)
	}
	requestRoot, ok := jsonutil.Object(root, "request")
	if !ok {
		return ConvertGeminiRequestToOpenAI(modelName, inputRawJSON, stream)
	}
	requestRoot["model"] = modelName
	if systemInstruction, ok := requestRoot["systemInstruction"]; ok {
		requestRoot["system_instruction"] = systemInstruction
		delete(requestRoot, "systemInstruction")
	}
	rawJSON := jsonutil.MarshalOrOriginal(inputRawJSON, requestRoot)

	return ConvertGeminiRequestToOpenAI(modelName, rawJSON, stream)
}
