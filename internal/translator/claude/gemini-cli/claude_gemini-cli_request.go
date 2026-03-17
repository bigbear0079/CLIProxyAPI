// Package geminiCLI provides request translation functionality for Gemini CLI
// to Claude Code API compatibility.
package geminiCLI

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	gemini "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/claude/gemini"
)

// ConvertGeminiCLIRequestToClaude parses and transforms a Gemini CLI API
// request into Claude Code API format.
func ConvertGeminiCLIRequestToClaude(modelName string, inputRawJSON []byte, stream bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	requestRoot, ok := jsonutil.Object(root, "request")
	if !ok {
		requestRoot = map[string]any{}
	}
	requestRoot["model"] = modelName

	if systemInstruction, ok := jsonutil.Get(requestRoot, "systemInstruction"); ok {
		requestRoot["system_instruction"] = systemInstruction
		_ = jsonutil.Delete(requestRoot, "systemInstruction")
	}

	requestRawJSON := jsonutil.MarshalOrOriginal(inputRawJSON, requestRoot)
	return gemini.ConvertGeminiRequestToClaude(modelName, requestRawJSON, stream)
}
