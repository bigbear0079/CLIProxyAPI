// Package gemini provides request translation functionality for Claude API.
// It handles parsing and transforming Claude API requests into the internal client format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package also performs JSON data cleaning and transformation to ensure compatibility
// between Claude API format and the internal client's expected format.
package geminiCLI

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
)

// PrepareClaudeRequest parses and transforms a Claude API request into internal client format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the internal client.
func ConvertGeminiCLIRequestToGemini(_ string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)
	requestRoot := map[string]any{}
	if request, ok := jsonutil.Object(root, "request"); ok {
		requestRoot = request
	}
	if modelName, ok := jsonutil.String(root, "model"); ok && modelName != "" {
		requestRoot["model"] = modelName
	}
	if systemInstruction, ok := requestRoot["systemInstruction"]; ok {
		requestRoot["system_instruction"] = systemInstruction
		delete(requestRoot, "systemInstruction")
	}

	if tools, ok := jsonutil.Array(requestRoot, "tools"); ok {
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			functionDeclarations, ok := jsonutil.Array(tool, "function_declarations")
			if !ok {
				continue
			}
			for _, declarationValue := range functionDeclarations {
				declaration, ok := declarationValue.(map[string]any)
				if !ok {
					continue
				}
				if parameters, ok := declaration["parameters"]; ok {
					declaration["parametersJsonSchema"] = parameters
					delete(declaration, "parameters")
				}
			}
		}
	}

	if contents, ok := jsonutil.Array(requestRoot, "contents"); ok {
		for _, contentValue := range contents {
			content, ok := contentValue.(map[string]any)
			if !ok {
				continue
			}
			role, _ := jsonutil.String(content, "role")
			if role != "model" {
				continue
			}
			parts, ok := jsonutil.Array(content, "parts")
			if !ok {
				continue
			}
			for _, partValue := range parts {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}
				if jsonutil.Exists(part, "functionCall") || jsonutil.Exists(part, "thoughtSignature") {
					part["thoughtSignature"] = "skip_thought_signature_validator"
				}
			}
		}
	}

	common.EnsureDefaultSafetySettings(requestRoot, "safetySettings")
	return jsonutil.MarshalOrOriginal(inputRawJSON, requestRoot)
}
