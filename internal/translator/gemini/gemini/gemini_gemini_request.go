// Package gemini provides in-provider request normalization for Gemini API.
// It ensures incoming v1beta requests meet minimal schema requirements
// expected by Google's Generative Language API.
package gemini

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	log "github.com/sirupsen/logrus"
)

// ConvertGeminiRequestToGemini normalizes Gemini v1beta requests.
//   - Adds a default role for each content if missing or invalid.
//     The first message defaults to "user", then alternates user/model when needed.
//
// It keeps the payload otherwise unchanged.
func ConvertGeminiRequestToGemini(_ string, inputRawJSON []byte, _ bool) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)

	if tools, ok := jsonutil.Array(root, "tools"); ok {
		for _, toolValue := range tools {
			tool, ok := toolValue.(map[string]any)
			if !ok {
				continue
			}
			if functionDeclarations, ok := tool["functionDeclarations"]; ok {
				tool["function_declarations"] = functionDeclarations
				delete(tool, "functionDeclarations")
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

	if contents, ok := jsonutil.Array(root, "contents"); ok {
		prevRole := ""
		for _, contentValue := range contents {
			content, ok := contentValue.(map[string]any)
			if !ok {
				continue
			}

			role, _ := jsonutil.String(content, "role")
			valid := role == "user" || role == "model"
			if role == "" || !valid {
				switch prevRole {
				case "":
					role = "user"
				case "user":
					role = "model"
				default:
					role = "user"
				}
				content["role"] = role
			}
			prevRole = role

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

	if generationConfig, ok := jsonutil.Object(root, "generationConfig"); ok {
		if responseSchema, ok := generationConfig["responseSchema"]; ok {
			generationConfig["responseJsonSchema"] = responseSchema
			delete(generationConfig, "responseSchema")
		}
	}

	backfillEmptyFunctionResponseNames(root)
	common.EnsureDefaultSafetySettings(root, "safetySettings")
	return jsonutil.MarshalOrOriginal(inputRawJSON, root)
}

// backfillEmptyFunctionResponseNames walks the contents array and for each
// model turn containing functionCall parts, records the call names in order.
// For the immediately following user/function turn containing functionResponse
// parts, any empty name is replaced with the corresponding call name.
func backfillEmptyFunctionResponseNames(root map[string]any) {
	contents, ok := jsonutil.Array(root, "contents")
	if !ok {
		return
	}

	var pendingCallNames []string

	for contentIdx, contentValue := range contents {
		content, ok := contentValue.(map[string]any)
		if !ok {
			continue
		}
		role, _ := jsonutil.String(content, "role")
		// Collect functionCall names from model turns
		if role == "model" {
			var names []string
			parts, _ := jsonutil.Array(content, "parts")
			for _, partValue := range parts {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}
				if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
					if name, ok := jsonutil.String(functionCall, "name"); ok {
						names = append(names, name)
					}
				}
			}
			if len(names) > 0 {
				pendingCallNames = names
			} else {
				pendingCallNames = nil
			}
			continue
		}

		// Backfill empty functionResponse names from pending call names
		if len(pendingCallNames) > 0 {
			ri := 0
			parts, _ := jsonutil.Array(content, "parts")
			for partIdx, partValue := range parts {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}
				functionResponse, ok := jsonutil.Object(part, "functionResponse")
				if ok {
					name, _ := jsonutil.String(functionResponse, "name")
					if strings.TrimSpace(name) == "" {
						if ri < len(pendingCallNames) {
							functionResponse["name"] = pendingCallNames[ri]
						} else {
							log.Debugf("more function responses than calls at contents[%d], skipping name backfill", contentIdx)
						}
					}
					ri++
				}
				_ = partIdx
			}
			pendingCallNames = nil
		}
	}
}
