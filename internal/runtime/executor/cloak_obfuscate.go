package executor

import (
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/jsonutil"
)

// zeroWidthSpace is the Unicode zero-width space character used for obfuscation.
const zeroWidthSpace = "\u200B"

// SensitiveWordMatcher holds the compiled regex for matching sensitive words.
type SensitiveWordMatcher struct {
	regex *regexp.Regexp
}

// buildSensitiveWordMatcher compiles a regex from the word list.
// Words are sorted by length (longest first) for proper matching.
func buildSensitiveWordMatcher(words []string) *SensitiveWordMatcher {
	if len(words) == 0 {
		return nil
	}

	// Filter and normalize words
	var validWords []string
	for _, w := range words {
		w = strings.TrimSpace(w)
		if utf8.RuneCountInString(w) >= 2 && !strings.Contains(w, zeroWidthSpace) {
			validWords = append(validWords, w)
		}
	}

	if len(validWords) == 0 {
		return nil
	}

	// Sort by length (longest first) for proper matching
	sort.Slice(validWords, func(i, j int) bool {
		return len(validWords[i]) > len(validWords[j])
	})

	// Escape and join
	escaped := make([]string, len(validWords))
	for i, w := range validWords {
		escaped[i] = regexp.QuoteMeta(w)
	}

	pattern := "(?i)" + strings.Join(escaped, "|")
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil
	}

	return &SensitiveWordMatcher{regex: re}
}

// obfuscateWord inserts a zero-width space after the first grapheme.
func obfuscateWord(word string) string {
	if strings.Contains(word, zeroWidthSpace) {
		return word
	}

	// Get first rune
	r, size := utf8.DecodeRuneInString(word)
	if r == utf8.RuneError || size >= len(word) {
		return word
	}

	return string(r) + zeroWidthSpace + word[size:]
}

// obfuscateText replaces all sensitive words in the text.
func (m *SensitiveWordMatcher) obfuscateText(text string) string {
	if m == nil || m.regex == nil {
		return text
	}
	return m.regex.ReplaceAllStringFunc(text, obfuscateWord)
}

// obfuscateSensitiveWords processes the payload and obfuscates sensitive words
// in system blocks and message content.
func obfuscateSensitiveWords(payload []byte, matcher *SensitiveWordMatcher) []byte {
	if matcher == nil || matcher.regex == nil {
		return payload
	}
	root, errParse := jsonutil.ParseObjectBytes(payload)
	if errParse != nil {
		return payload
	}

	modifiedSystem := obfuscateSystemBlocks(root, matcher)
	modifiedMessages := obfuscateMessages(root, matcher)
	if !modifiedSystem && !modifiedMessages {
		return payload
	}
	return jsonutil.MarshalOrOriginal(payload, root)
}

// obfuscateSystemBlocks obfuscates sensitive words in system blocks.
func obfuscateSystemBlocks(root map[string]any, matcher *SensitiveWordMatcher) bool {
	systemValue, ok := root["system"]
	if !ok {
		return false
	}

	if systemArray, ok := systemValue.([]any); ok {
		modified := false
		for _, value := range systemArray {
			block, ok := value.(map[string]any)
			if !ok {
				continue
			}
			if blockType, _ := block["type"].(string); blockType == "text" {
				text, _ := block["text"].(string)
				obfuscated := matcher.obfuscateText(text)
				if obfuscated != text {
					block["text"] = obfuscated
					modified = true
				}
			}
		}
		return modified
	}
	if text, ok := systemValue.(string); ok {
		obfuscated := matcher.obfuscateText(text)
		if obfuscated != text {
			root["system"] = obfuscated
			return true
		}
	}

	return false
}

// obfuscateMessages obfuscates sensitive words in message content.
func obfuscateMessages(root map[string]any, matcher *SensitiveWordMatcher) bool {
	messagesValue, ok := root["messages"]
	if !ok {
		return false
	}
	messages, ok := messagesValue.([]any)
	if !ok {
		return false
	}

	modified := false
	for _, msgValue := range messages {
		msg, ok := msgValue.(map[string]any)
		if !ok {
			continue
		}

		content, ok := msg["content"]
		if !ok {
			continue
		}
		if text, ok := content.(string); ok {
			// Simple string content
			obfuscated := matcher.obfuscateText(text)
			if obfuscated != text {
				msg["content"] = obfuscated
				modified = true
			}
		} else if contentArray, ok := content.([]any); ok {
			// Array of content blocks
			for _, blockValue := range contentArray {
				block, ok := blockValue.(map[string]any)
				if !ok {
					continue
				}
				if blockType, _ := block["type"].(string); blockType == "text" {
					text, _ := block["text"].(string)
					obfuscated := matcher.obfuscateText(text)
					if obfuscated != text {
						block["text"] = obfuscated
						modified = true
					}
				}
			}
		}
	}

	return modified
}
