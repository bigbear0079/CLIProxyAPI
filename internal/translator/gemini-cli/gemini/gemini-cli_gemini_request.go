// Package gemini provides request translation functionality for Gemini CLI to Gemini API compatibility.
// It handles parsing and transforming Gemini CLI API requests into Gemini API format,
// extracting model information, system instructions, message contents, and tool declarations.
// The package performs JSON data transformation to ensure compatibility
// between Gemini CLI API format and Gemini API's expected format.
package gemini

import (
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/translator/gemini/common"
	log "github.com/sirupsen/logrus"
)

// ConvertGeminiRequestToGeminiCLI parses and transforms a Gemini CLI API request into Gemini API format.
// It extracts the model name, system instruction, message contents, and tool declarations
// from the raw JSON request and returns them in the format expected by the Gemini API.
// The function performs the following transformations:
// 1. Extracts the model information from the request
// 2. Restructures the JSON to match Gemini API format
// 3. Converts system instructions to the expected format
// 4. Fixes CLI tool response format and grouping
//
// Parameters:
//   - modelName: The name of the model to use for the request (unused in current implementation)
//   - rawJSON: The raw JSON request data from the Gemini CLI API
//   - stream: A boolean indicating if the request is for a streaming response (unused in current implementation)
//
// Returns:
//   - []byte: The transformed request data in Gemini API format
func ConvertGeminiRequestToGeminiCLI(_ string, inputRawJSON []byte, _ bool) []byte {
	requestRoot := jsonutil.ParseObjectBytesOrEmpty(inputRawJSON)
	outRoot := map[string]any{
		"project": "",
		"request": requestRoot,
		"model":   "",
	}
	if modelName, ok := jsonutil.String(requestRoot, "model"); ok {
		outRoot["model"] = modelName
		delete(requestRoot, "model")
	}

	errFixCLIToolResponse := fixCLIToolResponse(outRoot)
	if errFixCLIToolResponse != nil {
		return []byte{}
	}

	if systemInstruction, ok := jsonutil.Get(requestRoot, "system_instruction"); ok {
		requestRoot["systemInstruction"] = systemInstruction
		delete(requestRoot, "system_instruction")
	}

	// Normalize roles in request.contents: default to valid values if missing/invalid
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
				if prevRole == "" {
					role = "user"
				} else if prevRole == "user" {
					role = "model"
				} else {
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

	// Filter out contents with empty parts to avoid Gemini API error:
	// "required oneof field 'data' must have one initialized field"
	if contents, ok := jsonutil.Array(requestRoot, "contents"); ok {
		filteredContents := make([]any, 0, len(contents))
		hasFiltered := false
		for _, contentValue := range contents {
			content, ok := contentValue.(map[string]any)
			if !ok {
				hasFiltered = true
				continue
			}
			parts, ok := jsonutil.Array(content, "parts")
			if !ok || len(parts) == 0 {
				hasFiltered = true
				continue
			}
			filteredContents = append(filteredContents, content)
		}
		if hasFiltered {
			requestRoot["contents"] = filteredContents
		}
	}

	common.EnsureDefaultSafetySettings(outRoot, "request.safetySettings")
	return jsonutil.MarshalOrOriginal(inputRawJSON, outRoot)
}

// FunctionCallGroup represents a group of function calls and their responses
type FunctionCallGroup struct {
	ResponsesNeeded int
	CallNames       []string // ordered function call names for backfilling empty response names
}

// backfillFunctionResponseName ensures that a functionResponse JSON object has a non-empty name,
// falling back to fallbackName if the original is empty.
func backfillFunctionResponseName(part map[string]any, fallbackName string) {
	functionResponse, ok := jsonutil.Object(part, "functionResponse")
	if !ok {
		return
	}
	name, _ := jsonutil.String(functionResponse, "name")
	if strings.TrimSpace(name) == "" && fallbackName != "" {
		functionResponse["name"] = fallbackName
	}
}

// fixCLIToolResponse performs sophisticated tool response format conversion and grouping.
// This function transforms the CLI tool response format by intelligently grouping function calls
// with their corresponding responses, ensuring proper conversation flow and API compatibility.
// It converts from a linear format (1.json) to a grouped format (2.json) where function calls
// and their responses are properly associated and structured.
//
// Parameters:
//   - input: The input JSON string to be processed
//
// Returns:
//   - string: The processed JSON string with grouped function calls and responses
//   - error: An error if the processing fails
func fixCLIToolResponse(root map[string]any) error {
	contents, ok := jsonutil.Array(root, "request.contents")
	if !ok {
		return fmt.Errorf("contents not found in input")
	}

	// Initialize data structures for processing and grouping
	var pendingGroups []*FunctionCallGroup  // Groups awaiting completion with responses
	var collectedResponses []map[string]any // Standalone responses to be matched
	groupedContents := make([]any, 0, len(contents))

	// Process each content object in the conversation
	// This iterates through messages and groups function calls with their responses
	for _, contentValue := range contents {
		value, ok := contentValue.(map[string]any)
		if !ok {
			log.Warnf("failed to parse content")
			continue
		}
		role, _ := jsonutil.String(value, "role")
		parts, _ := jsonutil.Array(value, "parts")

		// Check if this content has function responses
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

		// If this content has function responses, collect them
		if len(responsePartsInThisContent) > 0 {
			collectedResponses = append(collectedResponses, responsePartsInThisContent...)

			// Check if pending groups can be satisfied (FIFO: oldest group first)
			for len(pendingGroups) > 0 && len(collectedResponses) >= pendingGroups[0].ResponsesNeeded {
				group := pendingGroups[0]
				pendingGroups = pendingGroups[1:]

				// Take the needed responses for this group
				groupResponses := collectedResponses[:group.ResponsesNeeded]
				collectedResponses = collectedResponses[group.ResponsesNeeded:]

				// Create merged function response content
				groupParts := make([]any, 0, len(groupResponses))
				for ri, response := range groupResponses {
					backfillFunctionResponseName(response, group.CallNames[ri])
					groupParts = append(groupParts, response)
				}

				if len(groupParts) > 0 {
					groupedContents = append(groupedContents, map[string]any{
						"parts": groupParts,
						"role":  "function",
					})
				}
			}

			continue // Skip adding this content, responses are merged
		}

		// If this is a model with function calls, create a new group
		if role == "model" {
			var callNames []string
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
				// Add the model content
				groupedContents = append(groupedContents, value)

				// Create a new group for tracking responses
				group := &FunctionCallGroup{
					ResponsesNeeded: len(callNames),
					CallNames:       callNames,
				}
				pendingGroups = append(pendingGroups, group)
			} else {
				// Regular model content without function calls
				groupedContents = append(groupedContents, value)
			}
		} else {
			// Non-model content (user, etc.)
			groupedContents = append(groupedContents, value)
		}
	}

	// Handle any remaining pending groups with remaining responses
	for _, group := range pendingGroups {
		if len(collectedResponses) >= group.ResponsesNeeded {
			groupResponses := collectedResponses[:group.ResponsesNeeded]
			collectedResponses = collectedResponses[group.ResponsesNeeded:]

			groupParts := make([]any, 0, len(groupResponses))
			for ri, response := range groupResponses {
				backfillFunctionResponseName(response, group.CallNames[ri])
				groupParts = append(groupParts, response)
			}

			if len(groupParts) > 0 {
				groupedContents = append(groupedContents, map[string]any{
					"parts": groupParts,
					"role":  "function",
				})
			}
		}
	}

	// Update the original JSON with the new contents
	requestRoot, ok := jsonutil.Object(root, "request")
	if !ok {
		return fmt.Errorf("request not found in input")
	}
	requestRoot["contents"] = groupedContents
	return nil
}
