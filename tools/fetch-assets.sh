#!/usr/bin/env bash
# Downloads the pinned tokenizer.json assets listed in tools/pinned-assets.txt and verifies their
# SHA-256. Assets are model-owner files and never committed.
#
#   tools/fetch-assets.sh                       # into <repo>/.models (ignored by Git)
#   STARLING_MODELS=/data/models tools/fetch-assets.sh
#
# Layout: $STARLING_MODELS/<hub repo>/<full commit SHA>/<file>. A file already present with the right
# digest is not downloaded again.
#
# Derived assets are runtime pipelines that no Hub file holds. The qwen35-bpe-chat profile's
# tokenizer.json is the pipeline transformers 5.18.0 builds for Qwen/Qwen3.5-0.8B at 2fc06364
# (Qwen3_5Tokenizer, serialized with backend_tokenizer.to_str(pretty=True); reproduced by
# reference/qwen-bpe/run_qwen_bpe_tokenizer.py --write-live). It differs from the Kev-0.8B
# tokenizer.json only in the Split regex (\p{M} added), so it is derived from that file by replacing
# the regex line and accepted only if its SHA-256 matches the pinned digest.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${STARLING_MODELS:-$root/.models}"

# repo|commit|path|sha256, from tools/pinned-assets.txt (comments and blank lines skipped)
assets=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    assets+=("$line")
done <"$root/tools/pinned-assets.txt"

# target repo|commit|path|sha256 <- source repo|commit|path (the source must be in assets above)
derived=(
    "Qwen/Qwen3.5-0.8B|2fc06364715b967f1860aea9cf38778875588b17|transformers-5.18.0/tokenizer.json|a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f|jaredpalmer/kev-0.8b|9a45d25eb2ab761841196625383fa1dff0e56c1e|tokenizer.json"
)
# The part of the Split regex (JSON text, exactly once in the source) that differs between the two.
qwen2_regex_part='[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*'
qwen35_regex_part='[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*'


sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

status=0
for entry in "${assets[@]}"; do
    IFS='|' read -r repo commit path digest <<<"$entry"
    target="$dest/$repo/$commit/$path"
    if [[ -f "$target" && "$(sha256 "$target")" == "$digest" ]]; then
        echo "ok:   $repo@$commit/$path (present)"
        continue
    fi
    mkdir -p "$(dirname "$target")"
    tmp="$target.partial"
    curl --fail --silent --show-error --location --retry 3 \
        -o "$tmp" "https://huggingface.co/$repo/resolve/$commit/$path"
    actual="$(sha256 "$tmp")"
    if [[ "$actual" != "$digest" ]]; then
        echo "FAIL: $repo@$commit/$path sha256 $actual, expected $digest" >&2
        rm -f "$tmp"
        status=1
        continue
    fi
    mv "$tmp" "$target"
    echo "ok:   $repo@$commit/$path (downloaded)"
done

for entry in "${derived[@]}"; do
    IFS='|' read -r repo commit path digest src_repo src_commit src_path <<<"$entry"
    target="$dest/$repo/$commit/$path"
    if [[ -f "$target" && "$(sha256 "$target")" == "$digest" ]]; then
        echo "ok:   $repo@$commit/$path (present)"
        continue
    fi
    source="$dest/$src_repo/$src_commit/$src_path"
    if [[ ! -f "$source" ]]; then
        echo "FAIL: $repo@$commit/$path needs $src_repo@$src_commit/$src_path" >&2
        status=1
        continue
    fi
    mkdir -p "$(dirname "$target")"
    tmp="$target.partial"
    FROM="$qwen2_regex_part" TO="$qwen35_regex_part" \
        perl -0777 -pe 'my $n = s/\Q$ENV{FROM}\E/$ENV{TO}/g; die "expected one regex, found $n\n" if $n != 1' \
        "$source" >"$tmp"
    actual="$(sha256 "$tmp")"
    if [[ "$actual" != "$digest" ]]; then
        echo "FAIL: $repo@$commit/$path sha256 $actual, expected $digest" >&2
        rm -f "$tmp"
        status=1
        continue
    fi
    mv "$tmp" "$target"
    echo "ok:   $repo@$commit/$path (derived from $src_repo@$src_commit/$src_path)"
done

exit "$status"
