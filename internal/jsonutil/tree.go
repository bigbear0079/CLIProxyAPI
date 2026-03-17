package jsonutil

import "fmt"

func Get(root any, path string) (any, bool) {
	segments, errParse := ParsePath(path)
	if errParse != nil {
		return nil, false
	}
	return getBySegments(root, segments)
}

func Exists(root any, path string) bool {
	_, ok := Get(root, path)
	return ok
}

func Set(root map[string]any, path string, value any) error {
	segments, errParse := ParsePath(path)
	if errParse != nil {
		return errParse
	}
	if len(segments) == 0 {
		return fmt.Errorf("empty path")
	}
	updated, errSet := setBySegments(root, segments, value)
	if errSet != nil {
		return errSet
	}
	if _, ok := updated.(map[string]any); !ok {
		return fmt.Errorf("updated root is not an object")
	}
	return nil
}

func SetRawBytes(root map[string]any, path string, raw []byte) error {
	value, errParse := ParseAnyBytes(raw)
	if errParse != nil {
		return errParse
	}
	return Set(root, path, value)
}

func Delete(root map[string]any, path string) error {
	segments, errParse := ParsePath(path)
	if errParse != nil {
		return errParse
	}
	if len(segments) == 0 {
		return fmt.Errorf("empty path")
	}
	updated, errDelete := deleteBySegments(root, segments)
	if errDelete != nil {
		return errDelete
	}
	if _, ok := updated.(map[string]any); !ok {
		return fmt.Errorf("updated root is not an object")
	}
	return nil
}

func getBySegments(current any, segments []Segment) (any, bool) {
	if len(segments) == 0 {
		return current, true
	}
	segment := segments[0]

	if object, ok := current.(map[string]any); ok {
		child, okChild := object[segment.Key]
		if !okChild {
			return nil, false
		}
		return getBySegments(child, segments[1:])
	}

	if segment.Append {
		return nil, false
	}

	if segment.Index == nil {
		return nil, false
	}

	array, ok := current.([]any)
	if !ok {
		return nil, false
	}
	index := *segment.Index
	if index < 0 || index >= len(array) {
		return nil, false
	}
	return getBySegments(array[index], segments[1:])
}

func setBySegments(current any, segments []Segment, value any) (any, error) {
	if len(segments) == 0 {
		return value, nil
	}
	segment := segments[0]

	if object, ok := current.(map[string]any); ok {
		child, errSet := setBySegments(object[segment.Key], segments[1:], value)
		if errSet != nil {
			return nil, errSet
		}
		object[segment.Key] = child
		return object, nil
	}

	switch typed := current.(type) {
	case nil:
		// Preserve the legacy creation behavior from sjson for missing
		// containers: numeric and append segments still materialize arrays.
		if segment.Append || segment.Index != nil {
			array := []any{}
			return setBySegments(array, segments, value)
		}
		object := map[string]any{}
		return setBySegments(object, segments, value)
	case []any:
		array := typed
		if segment.Append {
			child, errSet := setBySegments(nil, segments[1:], value)
			if errSet != nil {
				return nil, errSet
			}
			return append(array, child), nil
		}
		if segment.Index == nil {
			return nil, fmt.Errorf("path segment requires array index, got %q", segment.Key)
		}
		index := *segment.Index
		for len(array) <= index {
			array = append(array, nil)
		}
		child, errSet := setBySegments(array[index], segments[1:], value)
		if errSet != nil {
			return nil, errSet
		}
		array[index] = child
		return array, nil
	default:
		return nil, fmt.Errorf("path segment %q requires object or array, got %T", segment.Key, current)
	}
}

func deleteBySegments(current any, segments []Segment) (any, error) {
	if len(segments) == 0 {
		return current, nil
	}
	segment := segments[0]

	if object, ok := current.(map[string]any); ok {
		if len(segments) == 1 {
			delete(object, segment.Key)
			return object, nil
		}
		child, okChild := object[segment.Key]
		if !okChild {
			return object, nil
		}
		updatedChild, errDelete := deleteBySegments(child, segments[1:])
		if errDelete != nil {
			return nil, errDelete
		}
		object[segment.Key] = updatedChild
		return object, nil
	}

	if segment.Append {
		return current, fmt.Errorf("append segment is not valid for delete")
	}
	if segment.Index == nil {
		return nil, fmt.Errorf("path segment requires array index, got %q", segment.Key)
	}

	array, ok := current.([]any)
	if !ok {
		return nil, fmt.Errorf("path segment requires array, got %T", current)
	}
	index := *segment.Index
	if index < 0 || index >= len(array) {
		return array, nil
	}
	if len(segments) == 1 {
		return append(array[:index], array[index+1:]...), nil
	}
	updatedChild, errDelete := deleteBySegments(array[index], segments[1:])
	if errDelete != nil {
		return nil, errDelete
	}
	array[index] = updatedChild
	return array, nil
}
