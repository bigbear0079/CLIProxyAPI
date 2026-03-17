package jsonutil

import "encoding/json"

func ParseObjectBytesOrEmpty(payload []byte) map[string]any {
	root, errParse := ParseObjectBytes(payload)
	if errParse != nil {
		return map[string]any{}
	}
	return root
}

func String(root any, path string) (string, bool) {
	value, ok := Get(root, path)
	if !ok || value == nil {
		return "", false
	}
	switch typed := value.(type) {
	case string:
		return typed, true
	case json.Number:
		return typed.String(), true
	case bool:
		if typed {
			return "true", true
		}
		return "false", true
	default:
		return "", false
	}
}

func Bool(root any, path string) (bool, bool) {
	value, ok := Get(root, path)
	if !ok {
		return false, false
	}
	boolean, ok := value.(bool)
	return boolean, ok
}

func Int64(root any, path string) (int64, bool) {
	value, ok := Get(root, path)
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

func Object(root any, path string) (map[string]any, bool) {
	value, ok := Get(root, path)
	if !ok {
		return nil, false
	}
	object, ok := value.(map[string]any)
	return object, ok
}

func Array(root any, path string) ([]any, bool) {
	value, ok := Get(root, path)
	if !ok {
		return nil, false
	}
	array, ok := value.([]any)
	return array, ok
}

func IsEmptyObject(root any, path string) bool {
	object, ok := Object(root, path)
	return ok && len(object) == 0
}
