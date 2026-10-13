#!/usr/bin/env bash
# Release drill for Starling.Tokenizers (docs/releasing.md).
#
# Proves the tokenizer releases on its own: from a clean clone of the committed HEAD, with no optional
# Microsoft comparator and no upstream patch merged, it
#   1. checks that no tokenizer project references Microsoft.ML.Tokenizers (the comparator is absent),
#   2. runs the full test suite with the pinned assets required (STARLING_REQUIRE_ASSETS=1, locked restore),
#   3. packs the package and inspects the nuspec (version, MIT OR Apache-2.0, no dependencies, notices, readme, profiles),
#   4. pushes it to a local folder feed only (never nuget.org),
#   5. builds a throwaway consumer outside the repository whose NuGet.config knows only that feed and
#      nuget.org (package source mapping sends Starling.* to the local feed only), with an isolated
#      empty NuGet package cache: no project references, no repository paths, no hidden cache,
#   6. proves Microsoft.ML.Tokenizers is absent from the consumer's restore graph and package cache,
#   7. runs the consumer: it verifies the pinned tokenizer.json digest and compares ids, tokens, masks,
#      offsets, decode, truncation and padded batch output with the protected e5 fixtures.
#
#   tools/release-drill.sh                 # assets from $STARLING_MODELS or <repo>/.models
#   DRILL_KEEP=1 tools/release-drill.sh    # keep the temporary directory for inspection
#
# Uncommitted changes are not part of the drill: it builds exactly the committed HEAD.
# The packed .nupkg/.snupkg and the consumer output are copied to <repo>/artifacts/release-drill/ (ignored by Git).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
models="${STARLING_MODELS:-$root/.models}"
upstream_url="https://github.com/starlinghq/tokenizers.git"
e5_repo="intfloat/multilingual-e5-small"
e5_commit="614241f622f53c4eeff9890bdc4f31cfecc418b3"
e5_sha256="0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
fixtures_rel="conformance/xlmr-unigram-e5"

export DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

work="$(mktemp -d "${TMPDIR:-/tmp}/starling-tokenizers-drill.XXXXXX")"
cleanup() {
    if [[ "${DRILL_KEEP:-0}" == "1" ]]; then
        echo "kept: $work"
    else
        rm -rf -- "$work"
    fi
}
trap cleanup EXIT

step() { printf '\n== %s\n' "$*"; }
fail() {
    echo "DRILL FAIL: $*" >&2
    exit 1
}

step "0. Inputs"
sha="$(git -C "$root" rev-parse HEAD)"
echo "commit:    $sha"
if [[ -n "$(git -C "$root" status --porcelain)" ]]; then
    echo "note:      working tree has uncommitted changes; the drill ignores them and builds $sha"
fi
echo "dotnet:    $(command -v dotnet) ($(dotnet --version))"
echo "work dir:  $work"
STARLING_MODELS="$models" "$root/tools/fetch-assets.sh"
tokenizer_json="$models/$e5_repo/$e5_commit/tokenizer.json"
[[ -f "$tokenizer_json" ]] || fail "pinned asset missing: $tokenizer_json"

step "1. Clean checkout of $sha"
src="$work/src"
git clone --quiet --no-hardlinks "$root" "$src"
git -C "$src" -c advice.detachedHead=false checkout --quiet --detach "$sha"
# SourceLink maps sources to the public repository, as in a GitHub checkout.
git -C "$src" remote set-url origin "$upstream_url"
[[ -z "$(git -C "$src" status --porcelain --ignored)" ]] || fail "clone is not clean"
echo "ok:   no build outputs, caches or untracked files in the checkout"

step "2. Optional Microsoft comparator disabled"
# Item elements only: descriptions and comments may name the package.
if grep -rIn --include='*.csproj' --include='*.props' --include='*.targets' --include='*.slnx' -Ei \
    '<(PackageReference|PackageVersion|ProjectReference|Reference|Project)[[:space:]][^>]*Microsoft\.ML\.Tokenizers' \
    "$src"; then
    fail "a tokenizer project references Microsoft.ML.Tokenizers; the drill runs with no comparator"
fi
echo "ok:   no project, props, targets or solution file references Microsoft.ML.Tokenizers"
echo "ok:   local changes shipped and upstream status: PROVENANCE.md"

# Every NuGet operation from here on uses an empty, private package cache: nothing comes from ~/.nuget.
export NUGET_PACKAGES="$work/nuget-build"

step "3. Test with assets required (locked restore)"
(
    cd "$src"
    CI=true STARLING_MODELS="$models" STARLING_REQUIRE_ASSETS=1 \
        dotnet test --solution Tokenizers.slnx --configuration Release 2>&1 | tee "$work/test.log"
) || fail "tests failed"
grep -Eqi 'skipped:[[:space:]]*[1-9]' "$work/test.log" && fail "tests were skipped; the drill requires every asset"
(cd "$src" && tools/check-independence.sh) || fail "independence check failed"

step "4. Pack"
version="$(cd "$src" && dotnet msbuild src/Starling.Tokenizers -getProperty:Version)"
echo "version:   $version"
(cd "$src" && CI=true dotnet pack src/Starling.Tokenizers --configuration Release \
    --output "$work/pack" >/dev/null) || fail "pack failed"
nupkg="$work/pack/Starling.Tokenizers.$version.nupkg"
[[ -f "$nupkg" ]] || fail "expected $nupkg"
[[ -f "${nupkg%.nupkg}.snupkg" ]] || fail "symbol package missing"
nuspec="$(unzip -p "$nupkg" Starling.Tokenizers.nuspec)"
contents="$(unzip -Z1 "$nupkg")"
grep -q "<version>$version</version>" <<<"$nuspec" || fail "nuspec version"
grep -q '<license type="expression">MIT OR Apache-2.0</license>' <<<"$nuspec" || fail "nuspec license"
grep -q "commit=\"$sha\"" <<<"$nuspec" || fail "nuspec repository commit"
grep -q '<group targetFramework="net10.0" />' <<<"$nuspec" || fail "package has dependencies; expected none"
profile_files=()
for f in "$src"/profiles/*.yaml; do profile_files+=("profiles/$(basename "$f")"); done
for f in lib/net10.0/Starling.Tokenizers.dll lib/net10.0/Starling.Tokenizers.xml \
    README.md THIRD-PARTY-NOTICES.md "${profile_files[@]}"; do
    grep -qx "$f" <<<"$contents" || fail "package lacks $f"
done
if grep -Eqi '\.(onnx|safetensors|bin)$|tokenizer\.json$' <<<"$contents"; then
    fail "package contains model assets"
fi
echo "ok:   $(basename "$nupkg") sha256 $(sha256sum "$nupkg" | cut -d' ' -f1)"
echo "ok:   nuspec: MIT OR Apache-2.0 expression, repository commit $sha, no dependencies; packs README, notices, profiles"

step "5. Push to a local folder feed (never nuget.org)"
feed="$work/feed"
mkdir -p "$feed"
dotnet nuget push "$nupkg" --source "$feed" >/dev/null
ls "$feed"

step "6. Clean consumer outside the repository"
consumer="$work/consumer"
mkdir -p "$consumer" "$work/fixtures"
cp "$root/tools/release-drill/consumer/Program.cs" "$consumer/"
sed "s/@VERSION@/$version/" "$root/tools/release-drill/consumer/Consumer.csproj.in" >"$consumer/Consumer.csproj"
cp "$src/$fixtures_rel/tokenizer.jsonl" "$src/$fixtures_rel/tokenizer-batch.json" "$work/fixtures/"
cat >"$consumer/NuGet.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="drill-local" value="$feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <packageSourceMapping>
    <packageSource key="drill-local">
      <package pattern="Starling.*" />
    </packageSource>
    <packageSource key="nuget.org">
      <package pattern="*" />
    </packageSource>
  </packageSourceMapping>
</configuration>
EOF
export NUGET_PACKAGES="$work/nuget-consumer"
(cd "$consumer" && dotnet restore >/dev/null) || fail "consumer restore failed"
(cd "$consumer" && dotnet list package --include-transitive) | tee "$work/consumer-packages.txt"
grep -qi 'Microsoft\.ML\.Tokenizers' "$work/consumer-packages.txt" && fail "Microsoft.ML.Tokenizers in the consumer graph"
python3 - "$consumer/obj/project.assets.json" "$nupkg" <<'EOF' || fail "assets file check failed"
import base64, hashlib, json, sys
assets = json.load(open(sys.argv[1]))
libraries = sorted(assets["libraries"])
print("assets libraries:", ", ".join(libraries))
assert not any("microsoft.ml.tokenizers" in name.lower() for name in libraries), "Microsoft.ML.Tokenizers restored"
assert all(assets["libraries"][n]["type"] == "package" for n in libraries), "non-package library (project reference)"
assert [n.split("/")[0] for n in libraries] == ["Starling.Tokenizers"], "unexpected libraries"
lib = assets["libraries"][libraries[0]]
packed = base64.b64encode(hashlib.sha512(open(sys.argv[2], "rb").read()).digest()).decode()
assert lib["sha512"] == packed, "restored package differs from the drill's packed package"
print("ok:   restored package sha512 equals the packed .nupkg (from the local feed)")
EOF
if find "$NUGET_PACKAGES" -maxdepth 1 -iname 'microsoft.ml.tokenizers*' | grep -q .; then
    fail "Microsoft.ML.Tokenizers present in the consumer package cache"
fi
echo "ok:   consumer package cache holds only: $(ls "$NUGET_PACKAGES" | tr '\n' ' ')"
(cd "$consumer" && dotnet build --configuration Release --no-restore >/dev/null) || fail "consumer build failed"
if grep -rIl --exclude='*.log' -F "$root" "$consumer"; then
    fail "consumer files mention the repository path"
fi
echo "ok:   no file in the consumer (sources, NuGet.config, obj, bin) names the repository path"

step "7. Run the consumer against the pinned asset and the protected fixtures"
(cd "$consumer" && dotnet run --configuration Release --no-build -- "$tokenizer_json" "$e5_sha256" "$work/fixtures") \
    | tee "$work/consumer.log" || fail "consumer checks failed"
grep -qx 'RESULT: PASS' "$work/consumer.log" || fail "consumer did not pass"

out="$root/artifacts/release-drill"
mkdir -p "$out"
cp "$nupkg" "${nupkg%.nupkg}.snupkg" "$work/consumer.log" "$work/consumer-packages.txt" "$out/"
step "DRILL PASS: Starling.Tokenizers $version from $sha (packages and logs in artifacts/release-drill/)"
