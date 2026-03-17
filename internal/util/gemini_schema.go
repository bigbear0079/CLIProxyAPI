// Package util provides utility functions for the CLI Proxy API server.
package util

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

var jsonPathKeyReplacer = strings.NewReplacer(".", "\\.", "*", "\\*", "?", "\\?")

const placeholderReasonDescription = "Brief explanation of why you are calling this tool"

// CleanJSONSchemaForAntigravity transforms a JSON schema to be compatible with Antigravity API.
// It handles unsupported keywords, type flattening, and schema simplification while preserving
// semantic information as description hints.
func CleanJSONSchemaForAntigravity(jsonStr string) string {
	return cleanJSONSchema(jsonStr, true)
}

// CleanJSONSchemaForGemini transforms a JSON schema to be compatible with Gemini tool calling.
// It removes unsupported keywords and simplifies schemas, without adding empty-schema placeholders.
func CleanJSONSchemaForGemini(jsonStr string) string {
	return cleanJSONSchema(jsonStr, false)
}

// cleanJSONSchema performs the core cleaning operations on the JSON schema.
func cleanJSONSchema(jsonStr string, addPlaceholder bool) string {
	root, errParse := jsonutil.ParseAnyBytes([]byte(jsonStr))
	if errParse != nil {
		return jsonStr
	}

	// Phase 1: Convert and add hints
	root = convertRefsToHints(root)
	root = convertConstToEnum(root)
	root = convertEnumValuesToStrings(root)
	root = addEnumHints(root)
	root = addAdditionalPropertiesHints(root)
	root = moveConstraintsToDescription(root)

	// Phase 2: Flatten complex structures
	root = mergeAllOf(root)
	root = flattenAnyOfOneOf(root)
	root = flattenTypeArrays(root)

	// Phase 3: Cleanup
	root = removeUnsupportedKeywords(root)
	if !addPlaceholder {
		// Gemini schema cleanup: remove nullable/title and placeholder-only fields.
		root = removeKeywords(root, []string{"nullable", "title"})
		root = removePlaceholderFields(root)
	}
	root = cleanupRequiredFields(root)

	// Phase 4: Add placeholder for empty object schemas (Claude VALIDATED mode requirement)
	if addPlaceholder {
		root = addEmptySchemaPlaceholder(root)
	}

	return string(jsonutil.MarshalOrOriginal([]byte(jsonStr), root))
}

// removeKeywords removes all occurrences of specified keywords from the JSON schema.
func removeKeywords(root any, keywords []string) any {
	deletePaths := make([]string, 0)
	pathsByField := findPathsByFields(root, keywords)
	for _, key := range keywords {
		for _, p := range pathsByField[key] {
			if isPropertyDefinition(trimSuffix(p, "."+key)) {
				continue
			}
			deletePaths = append(deletePaths, p)
		}
	}
	sortByDepth(deletePaths)
	for _, p := range deletePaths {
		root = deleteAt(root, p)
	}
	return root
}

// removePlaceholderFields removes placeholder-only properties ("_" and "reason") and their required entries.
func removePlaceholderFields(root any) any {
	// Remove "_" placeholder properties.
	paths := findPaths(root, "_")
	sortByDepth(paths)
	for _, p := range paths {
		if !strings.HasSuffix(p, ".properties._") {
			continue
		}
		root = deleteAt(root, p)
		parentPath := trimSuffix(p, ".properties._")
		reqPath := joinPath(parentPath, "required")
		req := getStrings(root, reqPath)
		if len(req) == 0 {
			continue
		}

		filtered := make([]string, 0, len(req))
		for _, required := range req {
			if required != "_" {
				filtered = append(filtered, required)
			}
		}
		if len(filtered) == 0 {
			root = deleteAt(root, reqPath)
		} else {
			root = setValueAt(root, reqPath, toAnyStrings(filtered))
		}
	}

	// Remove placeholder-only "reason" objects.
	reasonPaths := findPaths(root, "reason")
	sortByDepth(reasonPaths)
	for _, p := range reasonPaths {
		if !strings.HasSuffix(p, ".properties.reason") {
			continue
		}
		parentPath := trimSuffix(p, ".properties.reason")
		props, okProps := jsonutil.Object(root, joinPath(parentPath, "properties"))
		if !okProps || len(props) != 1 {
			continue
		}
		desc, _ := jsonutil.String(root, p+".description")
		if desc != placeholderReasonDescription {
			continue
		}
		root = deleteAt(root, p)
		reqPath := joinPath(parentPath, "required")
		req := getStrings(root, reqPath)
		if len(req) == 0 {
			continue
		}

		filtered := make([]string, 0, len(req))
		for _, required := range req {
			if required != "reason" {
				filtered = append(filtered, required)
			}
		}
		if len(filtered) == 0 {
			root = deleteAt(root, reqPath)
		} else {
			root = setValueAt(root, reqPath, toAnyStrings(filtered))
		}
	}

	return root
}

// convertRefsToHints converts $ref to description hints (Lazy Hint strategy).
func convertRefsToHints(root any) any {
	paths := findPaths(root, "$ref")
	sortByDepth(paths)

	for _, p := range paths {
		refVal, _ := jsonutil.String(root, p)
		defName := refVal
		if idx := strings.LastIndex(refVal, "/"); idx >= 0 {
			defName = refVal[idx+1:]
		}

		parentPath := trimSuffix(p, ".$ref")
		hint := fmt.Sprintf("See: %s", defName)
		if existing, okExisting := jsonutil.String(root, descriptionPath(parentPath)); okExisting && existing != "" {
			hint = fmt.Sprintf("%s (%s)", existing, hint)
		}

		root = setValueAt(root, parentPath, map[string]any{
			"type":        "object",
			"description": hint,
		})
	}
	return root
}

func convertConstToEnum(root any) any {
	for _, p := range findPaths(root, "const") {
		val, ok := jsonutil.Get(root, p)
		if !ok {
			continue
		}
		enumPath := trimSuffix(p, ".const") + ".enum"
		if !jsonutil.Exists(root, enumPath) {
			root = setValueAt(root, enumPath, []any{val})
		}
	}
	return root
}

// convertEnumValuesToStrings ensures all enum values are strings and the schema type is set to string.
// Gemini API requires enum values to be of type string, not numbers or booleans.
func convertEnumValuesToStrings(root any) any {
	for _, p := range findPaths(root, "enum") {
		arr, ok := jsonutil.Array(root, p)
		if !ok {
			continue
		}

		stringVals := make([]string, 0, len(arr))
		for _, item := range arr {
			stringVals = append(stringVals, jsonValueString(item))
		}

		// Always update enum values to strings and set type to "string"
		// This ensures compatibility with Antigravity Gemini which only allows enum for STRING type
		root = setValueAt(root, p, toAnyStrings(stringVals))
		parentPath := trimSuffix(p, ".enum")
		root = setValueAt(root, joinPath(parentPath, "type"), "string")
	}
	return root
}

func addEnumHints(root any) any {
	for _, p := range findPaths(root, "enum") {
		arr, ok := jsonutil.Array(root, p)
		if !ok {
			continue
		}
		if len(arr) <= 1 || len(arr) > 10 {
			continue
		}

		vals := make([]string, 0, len(arr))
		for _, item := range arr {
			vals = append(vals, jsonValueString(item))
		}
		root = appendHint(root, trimSuffix(p, ".enum"), "Allowed: "+strings.Join(vals, ", "))
	}
	return root
}

func addAdditionalPropertiesHints(root any) any {
	for _, p := range findPaths(root, "additionalProperties") {
		value, ok := jsonutil.Get(root, p)
		if !ok {
			continue
		}
		boolean, okBool := value.(bool)
		if okBool && !boolean {
			root = appendHint(root, trimSuffix(p, ".additionalProperties"), "No extra properties allowed")
		}
	}
	return root
}

var unsupportedConstraints = []string{
	"minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum",
	"pattern", "minItems", "maxItems", "uniqueItems", "format",
	"default", "examples", // Claude rejects these in VALIDATED mode
}

func moveConstraintsToDescription(root any) any {
	pathsByField := findPathsByFields(root, unsupportedConstraints)
	for _, key := range unsupportedConstraints {
		for _, p := range pathsByField[key] {
			val, ok := jsonutil.Get(root, p)
			if !ok || isJSONObject(val) || isJSONArray(val) {
				continue
			}
			parentPath := trimSuffix(p, "."+key)
			if isPropertyDefinition(parentPath) {
				continue
			}
			root = appendHint(root, parentPath, fmt.Sprintf("%s: %s", key, jsonValueString(val)))
		}
	}
	return root
}

func mergeAllOf(root any) any {
	paths := findPaths(root, "allOf")
	sortByDepth(paths)

	for _, p := range paths {
		allOf, ok := jsonutil.Array(root, p)
		if !ok {
			continue
		}
		parentPath := trimSuffix(p, ".allOf")

		for _, item := range allOf {
			itemObj, okItem := item.(map[string]any)
			if !okItem {
				continue
			}

			if props, okProps := itemObj["properties"].(map[string]any); okProps {
				for key, value := range props {
					destPath := joinPath(parentPath, "properties."+escapeGJSONPathKey(key))
					root = setValueAt(root, destPath, value)
				}
			}

			if req, okReq := itemObj["required"].([]any); okReq {
				reqPath := joinPath(parentPath, "required")
				current := getStrings(root, reqPath)
				for _, required := range req {
					s := jsonValueString(required)
					if s != "" && !contains(current, s) {
						current = append(current, s)
					}
				}
				root = setValueAt(root, reqPath, toAnyStrings(current))
			}
		}

		root = deleteAt(root, p)
	}
	return root
}

func flattenAnyOfOneOf(root any) any {
	for _, key := range []string{"anyOf", "oneOf"} {
		paths := findPaths(root, key)
		sortByDepth(paths)

		for _, p := range paths {
			arr, ok := jsonutil.Array(root, p)
			if !ok || len(arr) == 0 {
				continue
			}

			parentPath := trimSuffix(p, "."+key)
			parentDesc, _ := jsonutil.String(root, descriptionPath(parentPath))

			bestIdx, allTypes := selectBest(arr)
			selected := arr[bestIdx]

			if parentDesc != "" {
				selected = mergeDescriptionRaw(selected, parentDesc)
			}

			if len(allTypes) > 1 {
				hint := "Accepts: " + strings.Join(allTypes, " | ")
				selected = appendHintRaw(selected, hint)
			}

			root = setValueAt(root, parentPath, selected)
		}
	}
	return root
}

func selectBest(items []any) (bestIdx int, types []string) {
	bestScore := -1
	for i, item := range items {
		t, _ := jsonutil.String(item, "type")
		score := 0

		switch {
		case t == "object" || jsonutil.Exists(item, "properties"):
			score, t = 3, orDefault(t, "object")
		case t == "array" || jsonutil.Exists(item, "items"):
			score, t = 2, orDefault(t, "array")
		case t != "" && t != "null":
			score = 1
		default:
			t = orDefault(t, "null")
		}

		if t != "" {
			types = append(types, t)
		}
		if score > bestScore {
			bestScore, bestIdx = score, i
		}
	}
	return
}

func flattenTypeArrays(root any) any {
	paths := findPaths(root, "type")
	sortByDepth(paths)

	nullableFields := make(map[string][]string)

	for _, p := range paths {
		res, ok := jsonutil.Array(root, p)
		if !ok || len(res) == 0 {
			continue
		}

		hasNull := false
		nonNullTypes := make([]string, 0, len(res))
		for _, item := range res {
			s := jsonValueString(item)
			if s == "null" {
				hasNull = true
			} else if s != "" {
				nonNullTypes = append(nonNullTypes, s)
			}
		}

		firstType := "string"
		if len(nonNullTypes) > 0 {
			firstType = nonNullTypes[0]
		}

		root = setValueAt(root, p, firstType)

		parentPath := trimSuffix(p, ".type")
		if len(nonNullTypes) > 1 {
			hint := "Accepts: " + strings.Join(nonNullTypes, " | ")
			root = appendHint(root, parentPath, hint)
		}

		if hasNull {
			parts := splitGJSONPath(p)
			if len(parts) >= 3 && parts[len(parts)-3] == "properties" {
				fieldNameEscaped := parts[len(parts)-2]
				fieldName := unescapeGJSONPathKey(fieldNameEscaped)
				objectPath := strings.Join(parts[:len(parts)-3], ".")
				nullableFields[objectPath] = append(nullableFields[objectPath], fieldName)

				propPath := joinPath(objectPath, "properties."+fieldNameEscaped)
				root = appendHint(root, propPath, "(nullable)")
			}
		}
	}

	for objectPath, fields := range nullableFields {
		reqPath := joinPath(objectPath, "required")
		req := getStrings(root, reqPath)
		if len(req) == 0 {
			continue
		}

		filtered := make([]string, 0, len(req))
		for _, required := range req {
			if !contains(fields, required) {
				filtered = append(filtered, required)
			}
		}

		if len(filtered) == 0 {
			root = deleteAt(root, reqPath)
		} else {
			root = setValueAt(root, reqPath, toAnyStrings(filtered))
		}
	}
	return root
}

func removeUnsupportedKeywords(root any) any {
	keywords := append(unsupportedConstraints,
		"$schema", "$defs", "definitions", "const", "$ref", "$id", "additionalProperties",
		"propertyNames", "patternProperties", // Gemini doesn't support these schema keywords
		"enumTitles", "prefill", "deprecated", // Schema metadata fields unsupported by Gemini
	)

	deletePaths := make([]string, 0)
	pathsByField := findPathsByFields(root, keywords)
	for _, key := range keywords {
		for _, p := range pathsByField[key] {
			if isPropertyDefinition(trimSuffix(p, "."+key)) {
				continue
			}
			deletePaths = append(deletePaths, p)
		}
	}
	sortByDepth(deletePaths)
	for _, p := range deletePaths {
		root = deleteAt(root, p)
	}

	// Remove x-* extension fields (e.g., x-google-enum-descriptions) that are not supported by Gemini API.
	root = removeExtensionFieldsValue(root)
	return root
}

// removeExtensionFields removes all x-* extension fields from the JSON schema.
// These are OpenAPI/JSON Schema extension fields that Google APIs don't recognize.
func removeExtensionFields(jsonStr string) string {
	root, errParse := jsonutil.ParseAnyBytes([]byte(jsonStr))
	if errParse != nil {
		return jsonStr
	}
	root = removeExtensionFieldsValue(root)
	return string(jsonutil.MarshalOrOriginal([]byte(jsonStr), root))
}

func removeExtensionFieldsValue(root any) any {
	var paths []string
	walkForExtensions(root, "", &paths)
	sortByDepth(paths)
	for _, p := range paths {
		root = deleteAt(root, p)
	}
	return root
}

func walkForExtensions(value any, path string, paths *[]string) {
	switch typed := value.(type) {
	case []any:
		for index := len(typed) - 1; index >= 0; index-- {
			childPath := joinPath(path, strconv.Itoa(index))
			walkForExtensions(typed[index], childPath, paths)
		}
	case map[string]any:
		for keyStr, val := range typed {
			safeKey := escapeGJSONPathKey(keyStr)
			childPath := joinPath(path, safeKey)

			// If it's an extension field, delete it and skip its children.
			if strings.HasPrefix(keyStr, "x-") && !isPropertyDefinition(path) {
				*paths = append(*paths, childPath)
				continue
			}

			walkForExtensions(val, childPath, paths)
		}
	}
}

func cleanupRequiredFields(root any) any {
	for _, p := range findPaths(root, "required") {
		parentPath := trimSuffix(p, ".required")
		propsPath := joinPath(parentPath, "properties")

		req := getStrings(root, p)
		props, okProps := jsonutil.Object(root, propsPath)
		if len(req) == 0 || !okProps {
			continue
		}

		valid := make([]string, 0, len(req))
		for _, key := range req {
			if _, exists := props[key]; exists {
				valid = append(valid, key)
			}
		}

		if len(valid) != len(req) {
			if len(valid) == 0 {
				root = deleteAt(root, p)
			} else {
				root = setValueAt(root, p, toAnyStrings(valid))
			}
		}
	}
	return root
}

// addEmptySchemaPlaceholder adds a placeholder "reason" property to empty object schemas.
// Claude VALIDATED mode requires at least one required property in tool schemas.
func addEmptySchemaPlaceholder(root any) any {
	// Find all "type" fields
	paths := findPaths(root, "type")

	// Process from deepest to shallowest (to handle nested objects properly)
	sortByDepth(paths)

	for _, p := range paths {
		typeVal, _ := jsonutil.String(root, p)
		if typeVal != "object" {
			continue
		}

		// Get the parent path (the object containing "type")
		parentPath := trimSuffix(p, ".type")

		// Check if properties exists and is empty or missing
		propsPath := joinPath(parentPath, "properties")
		propsVal, existsProps := jsonutil.Get(root, propsPath)
		propsObj, okProps := propsVal.(map[string]any)
		reqPath := joinPath(parentPath, "required")
		reqVal, okReq := jsonutil.Array(root, reqPath)
		hasRequiredProperties := okReq && len(reqVal) > 0

		needsPlaceholder := false
		if !existsProps {
			// No properties field at all
			needsPlaceholder = true
		} else if okProps && len(propsObj) == 0 {
			// Empty properties object
			needsPlaceholder = true
		}

		if needsPlaceholder {
			// Add placeholder "reason" property
			reasonPath := joinPath(propsPath, "reason")
			root = setValueAt(root, reasonPath+".type", "string")
			root = setValueAt(root, reasonPath+".description", placeholderReasonDescription)

			// Add to required array
			root = setValueAt(root, reqPath, toAnyStrings([]string{"reason"}))
			continue
		}

		// If schema has properties but none are required, add a minimal placeholder.
		if okProps && !hasRequiredProperties {
			// DO NOT add placeholder if it's a top-level schema (parentPath is empty)
			// or if we've already added a placeholder reason above.
			if parentPath == "" {
				continue
			}
			placeholderPath := joinPath(propsPath, "_")
			if !jsonutil.Exists(root, placeholderPath) {
				root = setValueAt(root, placeholderPath+".type", "boolean")
			}
			root = setValueAt(root, reqPath, toAnyStrings([]string{"_"}))
		}
	}

	return root
}

// --- Helpers ---

func findPaths(root any, field string) []string {
	var paths []string
	Walk(root, "", field, &paths)
	return paths
}

func findPathsByFields(root any, fields []string) map[string][]string {
	set := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		set[field] = struct{}{}
	}
	paths := make(map[string][]string, len(set))
	walkForFields(root, "", set, paths)
	return paths
}

func walkForFields(value any, path string, fields map[string]struct{}, paths map[string][]string) {
	switch typed := value.(type) {
	case map[string]any:
		for keyStr, child := range typed {
			safeKey := escapeGJSONPathKey(keyStr)

			var childPath string
			if path == "" {
				childPath = safeKey
			} else {
				childPath = path + "." + safeKey
			}

			if _, ok := fields[keyStr]; ok {
				paths[keyStr] = append(paths[keyStr], childPath)
			}

			walkForFields(child, childPath, fields, paths)
		}
	case []any:
		for index, child := range typed {
			childPath := strconv.Itoa(index)
			if path != "" {
				childPath = path + "." + childPath
			}
			walkForFields(child, childPath, fields, paths)
		}
	}
}

func sortByDepth(paths []string) {
	sort.Slice(paths, func(i, j int) bool { return len(paths[i]) > len(paths[j]) })
}

func trimSuffix(path, suffix string) string {
	if path == strings.TrimPrefix(suffix, ".") {
		return ""
	}
	return strings.TrimSuffix(path, suffix)
}

func joinPath(base, suffix string) string {
	if base == "" {
		return suffix
	}
	return base + "." + suffix
}

func setValueAt(root any, path string, value any) any {
	if path == "" {
		return value
	}
	rootObj, ok := root.(map[string]any)
	if !ok {
		return root
	}
	if errSet := jsonutil.Set(rootObj, path, value); errSet != nil {
		return root
	}
	return rootObj
}

func deleteAt(root any, path string) any {
	if path == "" {
		return root
	}
	rootObj, ok := root.(map[string]any)
	if !ok {
		return root
	}
	if errDelete := jsonutil.Delete(rootObj, path); errDelete != nil {
		return root
	}
	return rootObj
}

func isPropertyDefinition(path string) bool {
	return path == "properties" || strings.HasSuffix(path, ".properties")
}

func descriptionPath(parentPath string) string {
	if parentPath == "" || parentPath == "@this" {
		return "description"
	}
	return parentPath + ".description"
}

func appendHint(root any, parentPath, hint string) any {
	descPath := parentPath + ".description"
	if parentPath == "" || parentPath == "@this" {
		descPath = "description"
	}
	if existing, okExisting := jsonutil.String(root, descPath); okExisting && existing != "" {
		hint = fmt.Sprintf("%s (%s)", existing, hint)
	}
	return setValueAt(root, descPath, hint)
}

func appendHintRaw(value any, hint string) any {
	obj, ok := value.(map[string]any)
	if !ok {
		return value
	}
	if existing, okExisting := obj["description"].(string); okExisting && existing != "" {
		hint = fmt.Sprintf("%s (%s)", existing, hint)
	}
	obj["description"] = hint
	return obj
}

func getStrings(root any, path string) []string {
	arr, ok := jsonutil.Array(root, path)
	if !ok {
		return nil
	}

	result := make([]string, 0, len(arr))
	for _, item := range arr {
		result = append(result, jsonValueString(item))
	}
	return result
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func toAnyStrings(values []string) []any {
	result := make([]any, 0, len(values))
	for _, value := range values {
		result = append(result, value)
	}
	return result
}

func orDefault(val, def string) string {
	if val == "" {
		return def
	}
	return val
}

func escapeGJSONPathKey(key string) string {
	if strings.IndexAny(key, ".*?") == -1 {
		return key
	}
	return jsonPathKeyReplacer.Replace(key)
}

func unescapeGJSONPathKey(key string) string {
	if !strings.Contains(key, "\\") {
		return key
	}
	var b strings.Builder
	b.Grow(len(key))
	for i := 0; i < len(key); i++ {
		if key[i] == '\\' && i+1 < len(key) {
			i++
			b.WriteByte(key[i])
			continue
		}
		b.WriteByte(key[i])
	}
	return b.String()
}

func splitGJSONPath(path string) []string {
	if path == "" {
		return nil
	}

	parts := make([]string, 0, strings.Count(path, ".")+1)
	var b strings.Builder
	b.Grow(len(path))

	for i := 0; i < len(path); i++ {
		c := path[i]
		if c == '\\' && i+1 < len(path) {
			b.WriteByte('\\')
			i++
			b.WriteByte(path[i])
			continue
		}
		if c == '.' {
			parts = append(parts, b.String())
			b.Reset()
			continue
		}
		b.WriteByte(c)
	}
	parts = append(parts, b.String())
	return parts
}

func mergeDescriptionRaw(value any, parentDesc string) any {
	obj, ok := value.(map[string]any)
	if !ok {
		return value
	}
	childDesc, _ := obj["description"].(string)
	switch {
	case childDesc == "":
		obj["description"] = parentDesc
		return obj
	case childDesc == parentDesc:
		return obj
	default:
		obj["description"] = fmt.Sprintf("%s (%s)", parentDesc, childDesc)
		return obj
	}
}

func isJSONObject(value any) bool {
	_, ok := value.(map[string]any)
	return ok
}

func isJSONArray(value any) bool {
	_, ok := value.([]any)
	return ok
}

func jsonValueString(value any) string {
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
	case float64:
		return strconv.FormatFloat(typed, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(typed), 'f', -1, 32)
	case int:
		return strconv.Itoa(typed)
	case int8:
		return strconv.FormatInt(int64(typed), 10)
	case int16:
		return strconv.FormatInt(int64(typed), 10)
	case int32:
		return strconv.FormatInt(int64(typed), 10)
	case int64:
		return strconv.FormatInt(typed, 10)
	case uint:
		return strconv.FormatUint(uint64(typed), 10)
	case uint8:
		return strconv.FormatUint(uint64(typed), 10)
	case uint16:
		return strconv.FormatUint(uint64(typed), 10)
	case uint32:
		return strconv.FormatUint(uint64(typed), 10)
	case uint64:
		return strconv.FormatUint(typed, 10)
	case nil:
		return ""
	default:
		out, errMarshal := json.Marshal(typed)
		if errMarshal != nil {
			return fmt.Sprint(typed)
		}
		return string(out)
	}
}
