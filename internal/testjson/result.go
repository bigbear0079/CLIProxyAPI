package testjson

import (
	"bytes"
	"encoding/json"
	"io"
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

type Type int

const (
	Null Type = iota
	False
	Number
	String
	True
	JSON
)

func (t Type) String() string {
	switch t {
	case Null:
		return "Null"
	case False:
		return "False"
	case Number:
		return "Number"
	case String:
		return "String"
	case True:
		return "True"
	case JSON:
		return "JSON"
	default:
		return "Unknown"
	}
}

type Result struct {
	value  any
	exists bool
	Raw    string
	Type   Type
}

func ParseBytes(payload []byte) Result {
	value, errParse := parseAny(payload)
	if errParse != nil {
		return Result{}
	}
	return newResult(value, true)
}

func Parse(payload string) Result {
	return ParseBytes([]byte(payload))
}

func GetBytes(payload []byte, path string) Result {
	return ParseBytes(payload).Get(path)
}

func Get(payload string, path string) Result {
	return Parse(payload).Get(path)
}

func Valid(payload string) bool {
	_, errParse := parseAny([]byte(payload))
	return errParse == nil
}

func SetBytes(payload []byte, path string, value any) ([]byte, error) {
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return nil, errParse
	}
	if errSet := jsonutil.Set(root, path, value); errSet != nil {
		return nil, errSet
	}
	return json.Marshal(root)
}

func (r Result) Exists() bool {
	return r.exists
}

func (r Result) Get(path string) Result {
	if !r.exists {
		return Result{}
	}
	path = strings.TrimSpace(path)
	if path == "" {
		return r
	}
	if path == "#" {
		if array, ok := r.value.([]any); ok {
			return newResult(json.Number(strconv.Itoa(len(array))), true)
		}
		return Result{}
	}
	if strings.HasSuffix(path, ".#") {
		basePath := strings.TrimSuffix(path, ".#")
		base := r.Get(basePath)
		if array, ok := base.value.([]any); ok {
			return newResult(json.Number(strconv.Itoa(len(array))), true)
		}
		return Result{}
	}

	value, ok := jsonutil.Get(r.value, path)
	if !ok {
		return Result{}
	}
	return newResult(value, true)
}

func (r Result) String() string {
	if !r.exists || r.value == nil {
		return ""
	}
	switch typed := r.value.(type) {
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
		return rawString(typed)
	}
}

func (r Result) Int() int64 {
	if !r.exists || r.value == nil {
		return 0
	}
	switch typed := r.value.(type) {
	case json.Number:
		intValue, errInt := typed.Int64()
		if errInt == nil {
			return intValue
		}
		floatValue, errFloat := typed.Float64()
		if errFloat == nil {
			return int64(floatValue)
		}
	case int:
		return int64(typed)
	case int8:
		return int64(typed)
	case int16:
		return int64(typed)
	case int32:
		return int64(typed)
	case int64:
		return typed
	case uint:
		return int64(typed)
	case uint8:
		return int64(typed)
	case uint16:
		return int64(typed)
	case uint32:
		return int64(typed)
	case uint64:
		return int64(typed)
	case float32:
		return int64(typed)
	case float64:
		return int64(typed)
	case string:
		intValue, errInt := strconv.ParseInt(typed, 10, 64)
		if errInt == nil {
			return intValue
		}
	}
	return 0
}

func (r Result) Float() float64 {
	if !r.exists || r.value == nil {
		return 0
	}
	switch typed := r.value.(type) {
	case json.Number:
		floatValue, errFloat := typed.Float64()
		if errFloat == nil {
			return floatValue
		}
	case float32:
		return float64(typed)
	case float64:
		return typed
	case int:
		return float64(typed)
	case int8:
		return float64(typed)
	case int16:
		return float64(typed)
	case int32:
		return float64(typed)
	case int64:
		return float64(typed)
	case uint:
		return float64(typed)
	case uint8:
		return float64(typed)
	case uint16:
		return float64(typed)
	case uint32:
		return float64(typed)
	case uint64:
		return float64(typed)
	case string:
		floatValue, errFloat := strconv.ParseFloat(typed, 64)
		if errFloat == nil {
			return floatValue
		}
	}
	return 0
}

func (r Result) Bool() bool {
	if !r.exists || r.value == nil {
		return false
	}
	switch typed := r.value.(type) {
	case bool:
		return typed
	case string:
		boolean, errBool := strconv.ParseBool(typed)
		return errBool == nil && boolean
	default:
		return false
	}
}

func (r Result) Value() any {
	return r.value
}

func (r Result) Array() []Result {
	array, ok := r.value.([]any)
	if !ok {
		return nil
	}
	result := make([]Result, 0, len(array))
	for _, item := range array {
		result = append(result, newResult(item, true))
	}
	return result
}

func (r Result) IsArray() bool {
	_, ok := r.value.([]any)
	return ok
}

func (r Result) IsObject() bool {
	_, ok := r.value.(map[string]any)
	return ok
}

func (r Result) ForEach(fn func(key, value Result) bool) {
	if !r.exists || fn == nil {
		return
	}

	switch typed := r.value.(type) {
	case []any:
		for index, item := range typed {
			key := newResult(json.Number(strconv.Itoa(index)), true)
			if !fn(key, newResult(item, true)) {
				return
			}
		}
	case map[string]any:
		for key, item := range typed {
			if !fn(newResult(key, true), newResult(item, true)) {
				return
			}
		}
	}
}

func parseAny(payload []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()

	var value any
	if errDecode := decoder.Decode(&value); errDecode != nil {
		return nil, errDecode
	}

	var trailing any
	if errDecode := decoder.Decode(&trailing); errDecode != io.EOF {
		if errDecode == nil {
			return nil, io.ErrUnexpectedEOF
		}
		return nil, errDecode
	}

	return value, nil
}

func newResult(value any, exists bool) Result {
	return Result{
		value:  value,
		exists: exists,
		Raw:    rawString(value),
		Type:   detectType(value),
	}
}

func rawString(value any) string {
	switch typed := value.(type) {
	case nil:
		return "null"
	case string:
		encoded, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return typed
		}
		return string(encoded)
	case json.Number:
		return typed.String()
	case bool:
		if typed {
			return "true"
		}
		return "false"
	default:
		encoded, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return ""
		}
		return string(encoded)
	}
}

func detectType(value any) Type {
	switch typed := value.(type) {
	case nil:
		return Null
	case bool:
		if typed {
			return True
		}
		return False
	case string:
		return String
	case json.Number, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return Number
	case []any, map[string]any:
		return JSON
	default:
		return Null
	}
}
