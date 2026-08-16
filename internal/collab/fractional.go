// Package collab provides real-time multi-user collaborative editing for
// Parallax timelines using a WebSocket hub, fractional indexing, and
// last-write-wins (LWW) field registers.
package collab

import "fmt"

// Fractional indexing over the alphabet 'a'-'z'.
// Keys are compared lexicographically. Any two distinct keys always have room
// for at least one key between them (at worst by appending characters).
//
// Initial keys are spread across the middle of the alphabet so that both
// prepend and append operations leave headroom before needing to go deeper.

const (
	fracAlpha  = "abcdefghijklmnopqrstuvwxyz"
	fracBase   = 26
	fracMinInt = 1  // 'a'
	fracMaxInt = 26 // 'z'
	fracMidInt = 13 // 'n'
	fracAbove  = fracMaxInt + 1 // sentinel for "past end of upper bound"
	fracBelow  = 0              // sentinel for "past end of lower bound"
)

// charRank returns the 1-based rank (1='a'…26='z') of position i in s.
// If i is past the end of s:
//   - isHigh=true  → returns fracAbove (27): effectively infinite
//   - isHigh=false → returns fracBelow (0): effectively minus-infinity
func charRank(s string, i int, isHigh bool) int {
	if i >= len(s) {
		if isHigh {
			return fracAbove
		}
		return fracBelow
	}
	return int(s[i]-'a') + 1
}

// rankToChar converts a 1-based rank back to a character.
func rankToChar(r int) byte {
	if r < fracMinInt {
		r = fracMinInt
	}
	if r > fracMaxInt {
		r = fracMaxInt
	}
	return 'a' + byte(r-1)
}

// KeyBetween returns a key k such that low < k < high (lexicographically).
// Either bound may be "" to indicate an open (unbounded) side.
// It panics if low >= high when both are non-empty.
func KeyBetween(low, high string) string {
	if low != "" && high != "" && low >= high {
		panic(fmt.Sprintf("collab: KeyBetween: low %q >= high %q", low, high))
	}

	for depth := 0; depth <= len(low)+len(high)+16; depth++ {
		lo := charRank(low, depth, false)
		hi := charRank(high, depth, true)

		if hi > lo+1 {
			// Found a gap of ≥ 2: place the midpoint at this depth.
			mid := (lo + hi) / 2
			// Prefix is the low key up to (not including) this depth.
			prefix := low
			if depth < len(low) {
				prefix = low[:depth]
			}
			return prefix + string(rankToChar(mid))
		}
		// hi ≤ lo+1: equal or adjacent at this position — go one level deeper.
	}

	// Fallback (should be unreachable with valid inputs).
	return low + "n"
}

// InitialKeys generates n evenly-spaced keys for a new sequence.
// They are spread across the middle of the alphabet to leave headroom.
func InitialKeys(n int) []string {
	if n <= 0 {
		return nil
	}
	keys := make([]string, n)
	// Spread across [fracMidInt-8 .. fracMidInt+8] capped to valid range,
	// using 2-character keys for more room.
	step := float64(fracBase-2) / float64(n+1)
	for i := range keys {
		idx := int(step*float64(i+1)) + 1
		if idx < 1 {
			idx = 1
		}
		if idx > fracBase {
			idx = fracBase
		}
		keys[i] = string(rankToChar(idx))
	}
	// Deduplicate (can happen when n > 26)
	keys = ensureUnique(keys)
	return keys
}

// RegenerateRanks assigns fresh evenly-spaced keys to a slice of IDs.
// Call this when fractional keys have become too deep or duplicated.
func RegenerateRanks(ids []string) map[string]string {
	out := make(map[string]string, len(ids))
	keys := distributeKeys(len(ids))
	for i, id := range ids {
		out[id] = keys[i]
	}
	return out
}

// distributeKeys generates n evenly-spaced 2-character keys.
func distributeKeys(n int) []string {
	if n == 0 {
		return nil
	}
	// Use 2-char keys for the initial layout: first char fixed at 'n',
	// second char spread across 'a'-'z'.
	keys := make([]string, n)
	step := float64(fracBase) / float64(n+1)
	for i := range keys {
		second := int(step * float64(i+1))
		if second < 1 {
			second = 1
		}
		if second > fracBase {
			second = fracBase
		}
		keys[i] = "n" + string(rankToChar(second))
	}
	return keys
}

func ensureUnique(keys []string) []string {
	seen := make(map[string]bool, len(keys))
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		for seen[k] {
			k = k + "n"
		}
		seen[k] = true
		out = append(out, k)
	}
	return out
}
