package common

import (
	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// DefaultSafetySettings returns the default Gemini safety configuration we attach to requests.
func DefaultSafetySettings() []map[string]string {
	return []map[string]string{
		{
			"category":  "HARM_CATEGORY_HARASSMENT",
			"threshold": "OFF",
		},
		{
			"category":  "HARM_CATEGORY_HATE_SPEECH",
			"threshold": "OFF",
		},
		{
			"category":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
			"threshold": "OFF",
		},
		{
			"category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
			"threshold": "OFF",
		},
		{
			"category":  "HARM_CATEGORY_CIVIC_INTEGRITY",
			"threshold": "BLOCK_NONE",
		},
	}
}

// EnsureDefaultSafetySettings ensures the default safety settings are present when absent.
func EnsureDefaultSafetySettings(root map[string]any, path string) {
	if jsonutil.Exists(root, path) {
		return
	}
	_ = jsonutil.Set(root, path, DefaultSafetySettings())
}

// AttachDefaultSafetySettings ensures the default safety settings are present when absent.
// The caller must provide the target JSON path (e.g. "safetySettings" or "request.safetySettings").
func AttachDefaultSafetySettings(rawJSON []byte, path string) []byte {
	root, errParse := jsonutil.ParseObjectBytes(rawJSON)
	if errParse != nil {
		return rawJSON
	}
	EnsureDefaultSafetySettings(root, path)
	return jsonutil.MarshalOrOriginal(rawJSON, root)
}
