#!/usr/bin/env bash
# Fails if any production project under src/ references Microsoft.ML.Tokenizers, directly or
# transitively. Comparison implementations belong in test or tool projects only (AGENTS.md).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
status=0
shopt -s nullglob

for proj in "$root"/src/*/*.csproj; do
    dotnet restore "$proj" --verbosity quiet >/dev/null
    if dotnet list "$proj" package --include-transitive | grep -qi 'Microsoft\.ML\.Tokenizers'; then
        echo "FAIL: ${proj#"$root"/} references Microsoft.ML.Tokenizers" >&2
        status=1
    else
        echo "ok:   ${proj#"$root"/}"
    fi
done

exit "$status"
