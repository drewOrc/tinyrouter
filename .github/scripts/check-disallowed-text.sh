#!/usr/bin/env bash
# Reads text on stdin and fails if any line matches a pattern in
# .github/disallowed-trailers.txt (extended regex, case-insensitive).
# The one definition shared by the commit-hygiene job (ci.yml) and the
# pr-text-hygiene job (pr-text.yml), so the two cannot drift apart.
# Usage: <text> | check-disallowed-text.sh <what is being checked>
# Exit: 0 clean, 1 a line matched, 2 the pattern file has no patterns.
set -euo pipefail

label="${1:-text}"
patterns_file="${PATTERNS_FILE:-.github/disallowed-trailers.txt}"
patterns="$(grep -vE '^[[:space:]]*(#|$)' "$patterns_file" | paste -sd'|' - || true)"
if [ -z "$patterns" ]; then
  echo "::error::$patterns_file has no patterns"
  exit 2
fi
bad="$(grep -iE "$patterns" || true)"
if [ -n "$bad" ]; then
  echo "::error::$label must not carry tool-attribution lines:"
  printf '%s\n' "$bad"
  exit 1
fi
echo "clean: $label"
