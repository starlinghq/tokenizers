# Reference runners

Python programs that run the reference implementation (huggingface/tokenizers, and transformers where a model's runtime pipeline needs it) at pinned versions, and write the fixtures the .NET tests compare against. They never run in CI. Their output is committed under `conformance/` or `tests/*/TestData/` after maintainer review.

Each runner has its own `requirements.txt`, compiled with hashes from `requirements.in` (`uv pip compile --python-version 3.11 --generate-hashes requirements.in -o requirements.txt`), because upstream projects need incompatible versions. Install with:

```sh
uv venv /tmp/ref && VIRTUAL_ENV=/tmp/ref uv pip install --require-hashes -r reference/<runner>/requirements.txt
/tmp/ref/bin/python -I reference/<runner>/<script>.py --help
```

All runners share the vendored helpers in [`fixtures.py`](fixtures.py): corpus loading, offset-unit conversion, file digests, the environment record and JSON Lines writing.

| Runner | Use | Environment | Output |
| --- | --- | --- | --- |
| [`tokenizer-json/run_tokenizer_json.py`](tokenizer-json/run_tokenizer_json.py) | **Any new profile**: `Tokenizer.from_file(asset)` over the corpus; optional `--truncate N`, `--without-special-tokens`, and a padded batch (`--batch`, `--pair-batch`, `--pad-token`) | tokenizers 0.23.3 | `conformance/<profile>/tokenizer.jsonl`, `environment.json`, with `--batch` also `tokenizer-batch.json`; without the optional flags it reproduces the e5, Gemma and Whisper fixtures byte for byte |
| [`qwen-bpe/run_qwen_bpe_tokenizer.py`](qwen-bpe/run_qwen_bpe_tokenizer.py) | `qwen2-bpe-kev`, `qwen35-bpe-chat`: also checks the live transformers pipeline equals the asset; randomized differential dumps; the regex class probe | `requirements-kev.txt` (transformers 5.17.0), `requirements-chat.txt` (transformers 5.18.0); tokenizers 0.23.2 | `conformance/qwen*/`, `tests/*/TestData/qwen-bpe/` |
| [`gemma-bpe/run_gemma_bpe_tokenizer.py`](gemma-bpe/run_gemma_bpe_tokenizer.py) | `gemma-bpe-embeddinggemma2`: randomized differential dumps and the synthetic component reference | tokenizers 0.23.3 | `tests/*/TestData/gemma-bpe/` |
| [`gliner2/run_gliner2_tokenizer.py`](gliner2/run_gliner2_tokenizer.py) | `gliner2-base-v1-unigram`: per-word tokenization as GLiNER2 calls it | the planning repository's GLiNER2 environment (transformers 4.57.6, tokenizers 0.22.2, gliner2 at a pinned commit); not carried over | `conformance/gliner2-base-v1-unigram/` |
| [`compare_tokenizers.py`](compare_tokenizers.py) | Differences between implementations (slow tokenizers, SentencePiece) and `tokenizer.json` | any environment with `tokenizers` | a report, not fixtures |

The `xlmr-unigram-e5` fixtures were written by the planning repository's e5 encoder runner, which also records model embeddings. `run_tokenizer_json.py --truncate 512` reproduces them byte for byte.

## Corpora

`conformance/corpus/` holds the inputs:

- `tokenizer-corpus-v1.jsonl`: 49 cases, the base for every profile. Covers ASCII, multilingual scripts, combining marks, emoji and ZWJ sequences, surrogate pairs, whitespace and control characters, empty input, special-token literals and look-alikes, pairs, and long inputs.
- `tokenizer-corpus-qwen-bpe-v1.jsonl`: 88 byte-level BPE cases. Covers contractions in mixed case, digit, whitespace and newline runs, Indic, Thai, Arabic and CJK text, astral letters, every Qwen added token and near-misses, and long words.

Each line is `{"id", "category", "text"[, "pair"]}`. Case ids are unique across the corpora a profile uses. Propose new cases as a new corpus file in their own pull request, with the source of each case. Never change an existing corpus that fixtures were generated from.

## Fixture format (version 1)

- JSON Lines, UTF-8, LF, one compact object per line, non-ASCII kept literal. Split on `\n` only: corpus text contains U+0085 and U+2028, which `str.splitlines()` and some readers treat as line breaks.
- Every case carries its input `text` (and `pair`), so a fixture is usable without the corpus file.
- Each case has the fields `id`, `category`, `text`, optional `pair`, and `encoding`. The `encoding` object holds `ids`, `tokens`, `type_ids`, `attention_mask`, `special_tokens_mask` and `offsets`, plus `sequence_ids` for pairs. Each case also has `decode` (`skip_special_tokens`, `keep_special_tokens`), and an optional `truncated` object (`max_length`, `strategy`, `direction`, `encoding`). With `--without-special-tokens`, each case also has `without_special_tokens`: the `encoding` with `add_special_tokens=False` and, where it is longer than `--truncate`, its own `truncated` object.
- Offsets are exclusive-end and recorded in three units: `unicode-scalars`, `utf16-code-units` and `utf8-bytes`. The upstream unit is Unicode scalar values (indices into a Python `str`); the other two are conversions of it. For a pair, each offset indexes the input its `sequence_ids` entry names. Special tokens have offsets `(0, 0)`.
- `environment.json` names the runner, the repository commit it ran from (`-dirty` if the tree had uncommitted changes), the asset and corpus digests, and interpreter, package and CPU details.
