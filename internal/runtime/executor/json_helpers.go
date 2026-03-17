package executor

import (
	"encoding/json"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

func jsonValueField(root map[string]any, path string) (any, bool) {
	if root == nil {
		return nil, false
	}
	return jsonutil.Get(root, path)
}

func jsonStringField(root map[string]any, path string) string {
	value, ok := jsonValueField(root, path)
	if !ok || value == nil {
		return ""
	}
	switch typed := value.(type) {
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
		out, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return fmt.Sprint(typed)
		}
		return string(out)
	}
}

func jsonBoolField(root map[string]any, path string) bool {
	value, ok := jsonValueField(root, path)
	if !ok {
		return false
	}
	typed, ok := value.(bool)
	return ok && typed
}

func jsonInt64Field(root map[string]any, path string) (int64, bool) {
	value, ok := jsonValueField(root, path)
	if !ok || value == nil {
		return 0, false
	}
	switch typed := value.(type) {
	case json.Number:
		intValue, errInt := typed.Int64()
		if errInt != nil {
			return 0, false
		}
		return intValue, true
	case int64:
		return typed, true
	case int:
		return int64(typed), true
	default:
		return 0, false
	}
}

func jsonStringFieldBytes(payload []byte, path string) string {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return ""
	}
	return jsonStringField(root, path)
}

func jsonValueFieldBytes(payload []byte, path string) (any, bool) {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return nil, false
	}
	return jsonValueField(root, path)
}

func jsonBoolFieldBytes(payload []byte, path string) bool {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return false
	}
	return jsonBoolField(root, path)
}

func jsonInt64FieldBytes(payload []byte, path string) (int64, bool) {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return 0, false
	}
	return jsonInt64Field(root, path)
}

func jsonArrayLengthField(root map[string]any, path string) int {
	value, ok := jsonValueField(root, path)
	if !ok {
		return 0
	}
	array, ok := value.([]any)
	if !ok {
		return 0
	}
	return len(array)
}

func setJSONFieldBytes(payload []byte, path string, value any) []byte {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return payload
	}
	if errSet := jsonutil.Set(root, path, value); errSet != nil {
		return payload
	}
	return jsonutil.MarshalOrOriginal(payload, root)
}

func setRawJSONFieldBytes(payload []byte, path string, raw []byte) []byte {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return payload
	}
	if errSet := jsonutil.SetRawBytes(root, path, raw); errSet != nil {
		return payload
	}
	return jsonutil.MarshalOrOriginal(payload, root)
}

func deleteJSONFieldBytes(payload []byte, path string) []byte {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return payload
	}
	if errDelete := jsonutil.Delete(root, path); errDelete != nil {
		return payload
	}
	return jsonutil.MarshalOrOriginal(payload, root)
}

func mutateJSONObjectBytes(payload []byte, mutate func(root map[string]any)) []byte {
	root := jsonutil.ParseObjectBytesOrEmpty(payload)
	mutate(root)
	return jsonutil.MarshalOrOriginal(payload, root)
}
