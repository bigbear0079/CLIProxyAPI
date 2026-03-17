package jsonutil

import (
	"bytes"
	"encoding/json"
	"fmt"
)

func ParseAnyBytes(payload []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var value any
	if errDecode := decoder.Decode(&value); errDecode != nil {
		return nil, errDecode
	}
	return value, nil
}

func ParseObjectBytes(payload []byte) (map[string]any, error) {
	value, errParse := ParseAnyBytes(payload)
	if errParse != nil {
		return nil, errParse
	}
	root, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("json payload is not an object")
	}
	return root, nil
}

func ParseArrayBytes(payload []byte) ([]any, error) {
	value, errParse := ParseAnyBytes(payload)
	if errParse != nil {
		return nil, errParse
	}
	root, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("json payload is not an array")
	}
	return root, nil
}

func MarshalAny(value any) ([]byte, error) {
	return json.Marshal(value)
}

func MarshalOrOriginal(original []byte, value any) []byte {
	out, errMarshal := MarshalAny(value)
	if errMarshal != nil {
		return original
	}
	return out
}
