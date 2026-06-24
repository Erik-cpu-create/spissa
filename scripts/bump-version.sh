#!/usr/bin/env bash
# Spissa automated version bump + changelog generator.
#
# Reads ./VERSION (four-part A.B.C.D), inspects Conventional-Commit subjects since
# the last `v*` tag, decides the bump level, then rewrites:
#   - VERSION              (A.B.C.D)
#   - Cargo.toml           ([workspace.package].version -> A.B.C)
#   - Cargo.lock           (local crate versions -> A.B.C)
#   - CHANGELOG.md         (prepends a new dated section under the BUMP:INSERT marker)
#
# When $GITHUB_OUTPUT is set, exports: version, semver, tag, level, changelog.
# Pure bash + awk; no network, no cargo. See CONTRIBUTING.md for the rules.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION_FILE="VERSION"
CHANGELOG="CHANGELOG.md"
MARKER="<!-- BUMP:INSERT -->"
LOCAL_CRATES="rtc-codec spissa-cli spissa-container spissa-import spissa-runtime"

# ---- current version -------------------------------------------------------
[[ -f "$VERSION_FILE" ]] || echo "0.0.0.0" > "$VERSION_FILE"
CUR="$(tr -d '[:space:]' < "$VERSION_FILE")"
IFS='.' read -r A B C D <<< "$CUR"
A=${A:-0}; B=${B:-0}; C=${C:-0}; D=${D:-0}

# ---- commits since the last release baseline -------------------------------
# Prefer the latest `v*` tag. Before any tag exists, start from the commit that
# last touched VERSION (this setup / previous release commit) so the first run
# never dumps the entire history.
LAST_TAG="$(git tag --list 'v*' --sort=-v:refname | head -n1 || true)"
if [[ -n "$LAST_TAG" ]]; then
  BASE="$LAST_TAG"
else
  BASE="$(git log -1 --format=%H -- "$VERSION_FILE" 2>/dev/null || true)"
fi
if [[ -n "$BASE" ]]; then RANGE="${BASE}..HEAD"; else RANGE="HEAD"; fi

mapfile -t SUBJECTS < <(git log "$RANGE" --no-merges --pretty=format:'%s' 2>/dev/null \
  | grep -vE '^chore\(release\):' || true)

# ---- decide bump level -----------------------------------------------------
# Policy: every push advances the PATCH segment (the 3rd part), e.g.
#   0.0.1.0 -> 0.0.2.0. An explicit marker in the HEAD commit message overrides
# this for the rare minor/major/revision bump.
HEAD_MSG="$(git log -1 --pretty=format:'%B' 2>/dev/null || true)"
level="patch"
if   [[ "$HEAD_MSG" == *"[major]"* ]] || [[ "$HEAD_MSG" == *"BREAKING CHANGE"* ]]; then level="major"
elif [[ "$HEAD_MSG" == *"[minor]"* ]];    then level="minor"
elif [[ "$HEAD_MSG" == *"[revision]"* ]]; then level="revision"
fi

case "$level" in
  major)    A=$((A+1)); B=0; C=0; D=0 ;;
  minor)    B=$((B+1)); C=0; D=0 ;;
  patch)    C=$((C+1)); D=0 ;;
  revision) D=$((D+1)) ;;
esac

NEW="$A.$B.$C.$D"
SEMVER="$A.$B.$C"
TAG="v$NEW"
DATE="$(date -u +%Y-%m-%d)"

# ---- write VERSION ---------------------------------------------------------
echo "$NEW" > "$VERSION_FILE"

# ---- update Cargo.toml workspace version (first top-level `version = "..."`) -
sed -i -E "0,/^version = \"[^\"]*\"/s//version = \"$SEMVER\"/" Cargo.toml

# ---- update local crate versions in Cargo.lock -----------------------------
if [[ -f Cargo.lock ]]; then
  awk -v v="$SEMVER" -v list="$LOCAL_CRATES" '
    BEGIN { n=split(list, a, " "); for (i=1;i<=n;i++) names[a[i]]=1 }
    /^name = "/ { nm=$0; sub(/^name = "/,"",nm); sub(/"$/,"",nm); hit=(nm in names) }
    hit && /^version = "/ { sub(/"[^"]*"/, "\"" v "\""); hit=0 }
    { print }
  ' Cargo.lock > Cargo.lock.tmp && mv Cargo.lock.tmp Cargo.lock
fi

# ---- build the changelog section -------------------------------------------
section="$(mktemp)"
emit() { # <title> <regex>
  local title="$1" rx="$2" body
  body="$(printf '%s\n' "${SUBJECTS[@]:-}" | grep -E "$rx" 2>/dev/null \
            | sed -E 's/^[a-z]+(\([^)]*\))?!?:[[:space:]]*//' || true)"
  if [[ -n "$body" ]]; then
    { echo ""; echo "### $title"; } >> "$section"
    while IFS= read -r l; do [[ -n "$l" ]] && echo "- $l" >> "$section"; done <<< "$body"
  fi
}

echo "## [$NEW] - $DATE" > "$section"
emit "Features"      '^feat(\(|!|:)'
emit "Fixes"         '^fix(\(|!|:)'
emit "Performance"   '^perf(\(|!|:)'
emit "Refactor"      '^refactor(\(|!|:)'
emit "Documentation" '^docs(\(|!|:)'
emit "Build & CI"    '^(build|ci)(\(|!|:)'
other="$(printf '%s\n' "${SUBJECTS[@]:-}" \
  | grep -vE '^(feat|fix|perf|refactor|docs|build|ci|chore|style|test)(\(|!|:)' \
  | grep -vE '^[[:space:]]*$' || true)"
if [[ -n "$other" ]]; then
  { echo ""; echo "### Other"; } >> "$section"
  while IFS= read -r l; do [[ -n "$l" ]] && echo "- $l" >> "$section"; done <<< "$other"
fi
echo "" >> "$section"

# ---- prepend the section after the marker ----------------------------------
awk -v marker="$MARKER" -v f="$section" '
  { print }
  $0==marker { print ""; while ((getline line < f) > 0) print line }
' "$CHANGELOG" > "$CHANGELOG.tmp" && mv "$CHANGELOG.tmp" "$CHANGELOG"

# ---- outputs ---------------------------------------------------------------
echo "Bumped $CUR -> $NEW (level: $level, range: $RANGE)"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "version=$NEW"
    echo "semver=$SEMVER"
    echo "tag=$TAG"
    echo "level=$level"
    echo "changelog<<__CHANGELOG_EOF__"
    cat "$section"
    echo "__CHANGELOG_EOF__"
  } >> "$GITHUB_OUTPUT"
fi
rm -f "$section"
