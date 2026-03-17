package thinking

import "encoding/json"

// GetThinkingText extracts the thinking text from a content part.
// Handles various formats:
// - Simple string: { "thinking": "text" } or { "text": "text" }
// - Wrapped object: { "thinking": { "text": "text", "cache_control": {...} } }
// - Gemini-style: { "thought": true, "text": "text" }
// Returns the extracted text string.
func GetThinkingText(part any) string {
	partObj, ok := part.(map[string]any)
	if !ok {
		return ""
	}

	// Try direct text field first (Gemini-style)
	if text, okText := stringifyThinkingValue(partObj["text"]); okText {
		return text
	}

	// Try thinking field
	thinkingField, exists := partObj["thinking"]
	if !exists {
		return ""
	}

	// thinking is a string
	if text, okText := stringifyThinkingValue(thinkingField); okText {
		return text
	}

	// thinking is an object with inner text/thinking
	if thinkingObj, okThinkingObj := thinkingField.(map[string]any); okThinkingObj {
		if inner, okInner := stringifyThinkingValue(thinkingObj["text"]); okInner {
			return inner
		}
		if inner, okInner := stringifyThinkingValue(thinkingObj["thinking"]); okInner {
			return inner
		}
	}

	return ""
}

func stringifyThinkingValue(value any) (string, bool) {
	switch typed := value.(type) {
	case string:
		return typed, true
	case json.Number:
		return typed.String(), true
	default:
		return "", false
	}
}
