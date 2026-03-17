// Package gemini provides request translation functionality for Gemini to
// Antigravity compatibility using standard JSON trees.
package gemini

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	log "github.com/sirupsen/logrus"
)

// ConvertGeminiRequestToAntigravity parses and transforms a Gemini request into
// the wrapped Antigravity request format.
func ConvertGeminiRequestToAntigravity(modelName string, inputRawJSON []byte, _ bool) []byte {
	requestRoot := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)
	delete(requestRoot, "model")

	outRoot := map[string]any{
		"project": "",
		"request": requestRoot,
		"model":   modelName,
	}

	if errFixCLIToolResponse := fixCLIToolResponseRoot(outRoot); errFixCLIToolResponse != nil {
		return []byte{}
	}

	if systemInstruction, ok := jsonutil.Get(requestRoot, "system_instruction"); ok {
		requestRoot["systemInstruction"] = systemInstruction
		delete(requestRoot, "system_instruction")
	}

	if contents, ok := jsonutil.Array(requestRoot, "contents"); ok {
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
		}
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

	if !strings.Contains(modelName, "claude") {
		const skipSentinel = "skip_thought_signature_validator"

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
					if thought, ok := jsonutil.Bool(part, "thought"); ok && thought {
						part["thoughtSignature"] = skipSentinel
					}
					if jsonutil.Exists(part, "functionCall") {
						existingSignature, _ := jsonutil.String(part, "thoughtSignature")
						if existingSignature == "" || len(existingSignature) < 50 {
							part["thoughtSignature"] = skipSentinel
						}
					}
				}
			}
		}
	}

	common.EnsureDefaultSafetySettings(outRoot, "request.safetySettings")
	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

// FunctionCallGroup represents a group of function calls and their responses.
type FunctionCallGroup struct {
	ResponsesNeeded int
	CallNames       []string
}

func parseFunctionResponsePart(value any, fallbackName string) map[string]any {
	if part, ok := value.(map[string]any); ok {
		if functionResponse, ok := jsonutil.Object(part, "functionResponse"); ok {
			name, _ := jsonutil.String(functionResponse, "name")
			if strings.TrimSpace(name) == "" && fallbackName != "" {
				functionResponse["name"] = fallbackName
			}
			return part
		}
	}

	log.Debugf("parse function response failed, using fallback")

	functionResponse := map[string]any{
		"name": fallbackName,
		"response": map[string]any{
			"result": antigravityGeminiContentString(value),
		},
	}

	if functionResponse["name"] == "" {
		functionResponse["name"] = "unknown"
	}

	if part, ok := value.(map[string]any); ok {
		if functionResponseValue, ok := jsonutil.Object(part, "functionResponse"); ok {
			if name, ok := jsonutil.String(functionResponseValue, "name"); ok && strings.TrimSpace(name) != "" {
				functionResponse["name"] = name
			}
			if responseValue, ok := jsonutil.Get(functionResponseValue, "response"); ok {
				functionResponse["response"] = map[string]any{
					"result": antigravityGeminiContentString(responseValue),
				}
			}
			if id, ok := jsonutil.String(functionResponseValue, "id"); ok && id != "" {
				functionResponse["id"] = id
			}
		}
	}

	return map[string]any{"functionResponse": functionResponse}
}

func fixCLIToolResponseRoot(root map[string]any) error {
	contents, ok := jsonutil.Array(root, "request.contents")
	if !ok {
		return fmt.Errorf("contents not found in input")
	}

	var pendingGroups []*FunctionCallGroup
	var collectedResponses []map[string]any
	groupedContents := make([]any, 0, len(contents))

	for _, contentValue := range contents {
		value, ok := contentValue.(map[string]any)
		if !ok {
			log.Warnf("failed to parse content")
			continue
		}
		role, _ := jsonutil.String(value, "role")
		parts, _ := jsonutil.Array(value, "parts")

		responsePartsInThisContent := make([]map[string]any, 0)
		for _, partValue := range parts {
			part, ok := partValue.(map[string]any)
			if !ok {
				continue
			}
			if jsonutil.Exists(part, "functionResponse") {
				responsePartsInThisContent = append(responsePartsInThisContent, part)
			}
		}

		if len(responsePartsInThisContent) > 0 {
			collectedResponses = append(collectedResponses, responsePartsInThisContent...)

			for len(pendingGroups) > 0 && len(collectedResponses) >= pendingGroups[0].ResponsesNeeded {
				group := pendingGroups[0]
				pendingGroups = pendingGroups[1:]

				groupResponses := collectedResponses[:group.ResponsesNeeded]
				collectedResponses = collectedResponses[group.ResponsesNeeded:]

				groupParts := make([]any, 0, len(groupResponses))
				for ri, response := range groupResponses {
					groupParts = append(groupParts, parseFunctionResponsePart(response, group.CallNames[ri]))
				}

				if len(groupParts) > 0 {
					groupedContents = append(groupedContents, map[string]any{
						"parts": groupParts,
						"role":  "function",
					})
				}
			}

			continue
		}

		if role == "model" {
			callNames := make([]string, 0)
			for _, partValue := range parts {
				part, ok := partValue.(map[string]any)
				if !ok {
					continue
				}
				if functionCall, ok := jsonutil.Object(part, "functionCall"); ok {
					if name, ok := jsonutil.String(functionCall, "name"); ok {
						callNames = append(callNames, name)
					}
				}
			}

			if len(callNames) > 0 {
				groupedContents = append(groupedContents, value)
				pendingGroups = append(pendingGroups, &FunctionCallGroup{
					ResponsesNeeded: len(callNames),
					CallNames:       callNames,
				})
				continue
			}
		}

		groupedContents = append(groupedContents, value)
	}

	for _, group := range pendingGroups {
		if len(collectedResponses) < group.ResponsesNeeded {
			continue
		}
		groupResponses := collectedResponses[:group.ResponsesNeeded]
		collectedResponses = collectedResponses[group.ResponsesNeeded:]

		groupParts := make([]any, 0, len(groupResponses))
		for ri, response := range groupResponses {
			groupParts = append(groupParts, parseFunctionResponsePart(response, group.CallNames[ri]))
		}
		if len(groupParts) > 0 {
			groupedContents = append(groupedContents, map[string]any{
				"parts": groupParts,
				"role":  "function",
			})
		}
	}

	requestRoot, ok := jsonutil.Object(root, "request")
	if !ok {
		return fmt.Errorf("request not found in input")
	}
	requestRoot["contents"] = groupedContents
	return nil
}

func fixCLIToolResponse(input string) (string, error) {
	root, errParse := jsonutil.ParseObjectBytes([]byte(input))
	if errParse != nil {
		return input, errParse
	}
	if errFix := fixCLIToolResponseRoot(root); errFix != nil {
		return input, errFix
	}
	return string(jsonutil.MarshalOrOriginal([]byte(input), root)), nil
}

func antigravityGeminiContentString(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	case json.Number:
		return typed.String()
	case bool:
		if typed {
			return "true"
		}
		return "false"
	default:
		return string(jsonutil.MarshalOrOriginal(nil, typed))
	}
}
