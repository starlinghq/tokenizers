# Adding a tokenizer

A new model is supported in one of two ways:

- **Its `tokenizer.json` already loads.** Adding it is data: a pin, reference fixtures and a profile file. No library code changes, and the release is a PATCH version.
- **It does not load.** One or more components are missing. Implement them first (see [Adding a component](#adding-a-component)), then add the profile.

Plan on about an hour for the first case. Most of that is reading what the model actually runs.

## 1. Inspect the file at a pinned revision

Pin the revision as a full commit SHA, never `main`:

```sh
repo=BAAI/bge-small-en-v1.5
rev=$(curl -s "https://huggingface.co/api/models/$repo/revision/main" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')
mkdir -p ".models/$repo/$rev"
curl -sSL -o ".models/$repo/$rev/tokenizer.json" "https://huggingface.co/$repo/resolve/$rev/tokenizer.json"
dotnet run tools/inspect.cs -- ".models/$repo/$rev/tokenizer.json"
```

`inspect.cs` prints the SHA-256, the pipeline components, vocabulary sizes and a sample encode. If the file does not load, it prints the missing component, for example `UNSUPPORTED pre_tokenizer.type=Digits`.

Gated repositories (for example Gemma or Llama) need a Hugging Face token: add `-H "Authorization: Bearer $HF_TOKEN"` to both `curl` calls. Read the model's license before you pin it. The file is never committed, but CI downloads it.

## 2. Confirm which pipeline the model actually runs

The Hub `tokenizer.json` is not always what the model uses. transformers can rebuild the pipeline from a tokenizer class and ignore the file; that is the case for Qwen3.5. Some runtimes also add or strip special tokens around the post-processor. Check before pinning:

```sh
python3 -m venv /tmp/tok-check && /tmp/tok-check/bin/pip install -q transformers tokenizers
/tmp/tok-check/bin/python - "$repo" "$rev" ".models/$repo/$rev/tokenizer.json" <<'EOF'
import json, sys
from tokenizers import Tokenizer
from transformers import AutoTokenizer
repo, rev, path = sys.argv[1:]
live = AutoTokenizer.from_pretrained(repo, revision=rev).backend_tokenizer
same = json.loads(live.to_str()) == json.loads(Tokenizer.from_file(path).to_str())
print("live pipeline equals the Hub file:", same, "|", type(AutoTokenizer.from_pretrained(repo, revision=rev)).__name__)
EOF
```

If the two differ, the profile's asset is the live pipeline, not the Hub file. Record it as a derived asset, as `qwen35-bpe-chat` does: `tools/fetch-assets.sh` derives the file, and the result is accepted only if it matches the pinned digest. Note in the profile which transformers version and class produced the pipeline.

## 3. Pin it

Add one line to [`tools/pinned-assets.txt`](../tools/pinned-assets.txt):

```text
BAAI/bge-small-en-v1.5|5c38ec7c405ec4b44b94cc5a9bb96e735b38267a|tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
```

Add the same pin as a field in `tests/StarlingHq.Tokenizers.Tests/TestAssets.cs`:

```csharp
public static readonly PinnedAsset BgeSmallTokenizerJson = new(
    "BAAI/bge-small-en-v1.5",
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "tokenizer.json",
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66");
```

Then run `tools/fetch-assets.sh`. It downloads the file into `.models/` and verifies the digest. `PinnedAssetsTests` fails if the list, the test pins and the profiles disagree.

## 4. Generate the reference fixtures

The generic runner records what huggingface/tokenizers produces for every corpus case, in the format `ProfileConformanceTests` reads:

```sh
uv venv /tmp/ref && VIRTUAL_ENV=/tmp/ref uv pip install --require-hashes -r reference/tokenizer-json/requirements.txt
/tmp/ref/bin/python -I reference/tokenizer-json/run_tokenizer_json.py \
    --profile bert-wordpiece-bge-small \
    --asset ".models/$repo/$rev/tokenizer.json" \
    --corpus conformance/corpus/tokenizer-corpus-v1.jsonl \
    --out conformance/bert-wordpiece-bge-small \
    --truncate 512
```

- **Profile id.** Use `<family>-<algorithm>-<model>`, for example `qwen35-bpe-chat` or `gemma-bpe-embeddinggemma2`.
- **Corpus.** `tokenizer-corpus-v1.jsonl` (49 cases) is the base for every profile. Byte-level BPE profiles also take `tokenizer-corpus-qwen-bpe-v1.jsonl` (88 cases). If the model needs cases the corpora lack, such as a script, a special-token pattern or a long input, propose them in a separate pull request that adds a new corpus file, with the source of each case.
- **`--truncate`.** Set it to the model's maximum input length when long inputs are truncated in practice, as for encoders.
- **Reference version.** The runner pins tokenizers 0.23.3. Bump it only on purpose, and recheck every profile that uses the runner.

Add a `PROPOSED.md` next to the fixtures with these facts:
- Status: proposed.
- The exact runner command.
- The runner's repository commit, from `environment.json`.
- The reference version.
- The asset (repository, revision, SHA-256).
- The corpus files and their case counts.

A maintainer accepts the fixtures by reviewing them (`CODEOWNERS` covers `conformance/`). Once accepted, they are never edited to make code pass.

## 5. Write the profile

Copy the closest existing profile under [`profiles/`](../profiles/) and edit it. The tests read `profile`, the `asset` block and `conformance.fixtures`. The rest is the record users and reviewers rely on:

```yaml
profile: bert-wordpiece-bge-small
status: experimental   # supported | experimental | untested | unsupported
library: StarlingHq.Tokenizers 0.2.1-preview.1

asset:
  repo: BAAI/bge-small-en-v1.5
  revision: 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
  file: tokenizer.json
  sha256: d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
  size: 711396
  format_version: "1.0"
  acquisition: tools/fetch-assets.sh

reference:
  implementation: huggingface/tokenizers 0.23.3 (Python binding, Tokenizer.from_file)
  offsets_unit: unicode-scalars

pipeline:   # exactly what the pinned file declares (tools/inspect.cs lists the components)
  normalizer: BertNormalizer(clean_text, handle_chinese_chars, lowercase; strip_accents null)
  pre_tokenizer: BertPreTokenizer
  model: WordPiece(vocab 30522, unk "[UNK]", prefix "##", max_input_chars_per_word 100)
  post_processor: TemplateProcessing(single "[CLS] $A [SEP]", pair "[CLS] $A [SEP] $B:1 [SEP]:1")
  decoder: WordPiece(prefix "##", cleanup true)

conformance:
  fixtures: conformance/bert-wordpiece-bge-small/   # proposed
  corpus: conformance/corpus/tokenizer-corpus-v1.jsonl (49 cases)
  tests: tests/StarlingHq.Tokenizers.Tests/ProfileConformanceTests.cs
  result:
    encode: 49/49 cases exact
    decode: 49/49 cases exact

capabilities:   # every capability gets an explicit status; missing never means supported
  encode_single: supported
  encode_pair: supported
  decode_skip_special_tokens: supported
  truncation_right_longest_first_single: supported
  incremental_decode_stream: unsupported

known_divergences:
  against_reference: none on the conformance corpus
```

## 6. Run the tests

```sh
STARLING_REQUIRE_ASSETS=1 dotnet test --solution Tokenizers.slnx -- --filter-class '*ProfileConformanceTests'
STARLING_REQUIRE_ASSETS=1 dotnet test --solution Tokenizers.slnx
```

Every case must match. A mismatch is a bug in the library or a component it implements incompletely. Never treat it as a fixture to adjust.
1. Find the smallest input that differs.
2. Read the upstream source at the reference revision.
3. Fix the port, with a component test that reproduces the case.
4. If the behavior cannot be matched yet, make the loader reject the option, and mark the capability `unsupported` in the profile.

For large vocabularies or new pre-tokenizer logic, add a randomized differential run, as the Qwen and Gemma profiles do (`reference/qwen-bpe/`, `reference/gemma-bpe/`, `*DifferentialTests.cs`). It catches what a 49-case corpus cannot.

## 7. Document and release

- Add the profile to the table in [`README.md`](../README.md) and to the summary in [`llms.txt`](../llms.txt).
- Add a `CHANGELOG.md` entry under `Unreleased`, for example `Profile bert-wordpiece-bge-small (BAAI/bge-small-en-v1.5), experimental: 49/49 cases exact`.
- Open the pull request. New fixtures for a new profile can come in the same pull request as the profile. Changing existing fixtures always needs its own pull request.
- After merge, release with a PATCH bump ([releasing.md](releasing.md)).

## Adding a component

When `inspect.cs` reports `UNSUPPORTED`, the component needs a port:

1. **Read the reference.** Find the upstream implementation in huggingface/tokenizers at the reference revision (`tokenizers/src/normalizers/`, `pre_tokenizers/`, `models/`, `processors/`, `decoders/`) and its serde field names and defaults.
2. **Accept exactly what you implement.** Add a case to `Pipeline/TokenizerJsonLoader.cs` that `AllowKeys` the fields you implement and throws `UnsupportedTokenizerComponentException` for any option value you do not.
3. **Port the stage** into the matching `Pipeline/` file, keeping upstream's alignment and offset behavior. Put the upstream file and blob in the file header and a row in `PROVENANCE.md`.
4. **Add reference evidence.** Write a component test with cases generated by the reference (see `GemmaBpeComponentTests` and `BertWordPieceComponentTests`), covering the option variants you accept.
5. **Update the docs.** Edit the row in [supported-components.md](supported-components.md) and add a `CHANGELOG.md` entry. A new component is a PATCH release, unless the public API changes.

Then continue with step 3 above for the model that needed it.
