# Architecture

Starlings.Tokenizers is an interpreter for the `tokenizer.json` format. It does not reimplement any particular model's tokenizer. It reproduces huggingface/tokenizers stage by stage, so any file built only from implemented components behaves like the reference. The reference is the specification: each stage is ported from its upstream Rust source, keeps upstream's offset alignment, and reproduces upstream's quirks rather than "fixing" them.

## Loading

`Tokenizer.FromFile`, `FromStream` and `FromUtf8Json` all go through `Pipeline/TokenizerJsonLoader.cs`, which is the gate:

- Each slot (`normalizer`, `pre_tokenizer`, `model`, `post_processor`, `decoder`, `added_tokens`, `truncation`, `padding`) is read by type tag.
- `AllowKeys` rejects any field the implementation does not read. Unknown fields are errors, not silently ignored.
- An unimplemented type or option value throws `UnsupportedTokenizerComponentException`, whose `Component` names the JSON path (for example `model.type=WordLevel`). A malformed document throws `TokenizerFormatException`.
- Every accepted stage appends a `slot:Type` name to `Tokenizer.Components`. Profiles and tests use the list to state exactly which pipeline ran.

**Fast path.** Large vocabularies (the Unigram `model.vocab` array, the BPE `vocab` object and `merges` array) are scanned directly from the UTF-8 bytes. Only the rest of the document goes through `JsonDocument`. Pieces stay UTF-8 and strings are created on demand. The fast path accepts only input that the document path would read to the same values. Anything else (malformed entries, duplicate keys, invalid JSON) falls back to the unchanged document path, so errors and strictness are identical. `LoadFastPathTests` and `BpeLoaderTests` compare the two paths.

The result is an immutable `TokenizerPipeline`. `WithTruncation` and `WithPadding` create new `Tokenizer` instances that share it, so a loaded vocabulary is never copied.

## Encoding

For one text (`Tokenizer.EncodeCore`), in upstream order:

1. **Validate.** Unpaired UTF-16 surrogates are rejected, and the text is converted to UTF-8. All internal offsets are UTF-8 byte offsets, as upstream.
2. **Added tokens.** `AddedVocabulary.ExtractAndNormalize` splits out added-token literals (leftmost-longest, over a byte trie) and normalizes the remaining segments. Tokens marked `normalized` are matched after normalization, the others before, as upstream.
3. **Normalize.** The normalizer stages rewrite a `NormalizedString`, which keeps an alignment from every normalized byte back to its original byte range. This is how offsets survive NFC, `Precompiled` charsmaps, `Replace` and lowercasing.
4. **Pre-tokenize.** The pre-tokenizer stages split a `PreTokenizedString` into pieces, each with its own alignment.
5. **Model.** Each piece is tokenized independently by `IModel`: `Unigram` (a Viterbi best path over a double-array trie, adopted from dotnet/machinelearning), `BPE` (merge-rank priority queue, optional byte fallback and unknown fusion), or `WordPiece` (greedy longest-match-first).
6. **Truncate.** Truncation runs before post-processing, as upstream, after reserving room for the special tokens the post-processor will add.
7. **Post-process.** `TemplateProcessing` or `ByteLevel` adds special tokens, type ids and sequence ids. `ByteLevel` can also trim offsets.
8. **Pad.** Single encodes pad on their own. `EncodeBatch` pads every row to the longest row, a fixed length, or a multiple of either.
9. **Offsets out.** `TokenEncoding` converts the byte offsets into UTF-16 code units (`Offsets`) and Unicode scalar values (`ScalarOffsets`).

**Decoding** maps ids to tokens (added vocabulary first, then the model), drops special tokens if asked, and runs the decoder chain (`Metaspace`, `ByteLevel`, `Replace`, `ByteFallback`, `Fuse`, `WordPiece`, `Sequence`) with upstream's `decode_chain` semantics.

## Source map

| File | Role |
| --- | --- |
| `Tokenizer.cs` | Public entry point: load, encode, batch, truncate, pad, decode, vocabulary lookups |
| `TokenEncoding.cs`, `TokenizerOptions.cs`, `TokenizerExceptions.cs` | Public result, options and exception types |
| `Pipeline/TokenizerJsonLoader.cs` | Document reader and component gate (above) |
| `Pipeline/NormalizedString.cs`, `PreTokenizedString.cs`, `EncodingBuilder.cs` | Upstream's aligned string, split string and mutable encoding |
| `Pipeline/Normalizers.cs`, `NfcNormalizer.cs` | `Sequence`, `Precompiled`, `Strip`, `Replace`, `NFC` (Unicode 9.0.0 tables, as the reference) |
| `Pipeline/PreTokenizers.cs`, `QwenSplitPattern.cs` | `Sequence`, `Metaspace`, `Split`, `ByteLevel`; a hand-written scanner for the two Qwen split regexes and the Llama 3 one |
| `Pipeline/AddedVocabulary.cs` | Added-token matching and ids |
| `Pipeline/UnigramVocabulary*.cs`, `VocabularyTrie.cs`, `ThirdParty/MLNet/` | Unigram vocabulary, trie construction, best-path search |
| `Pipeline/BpeModel.cs`, `BpeModelReader.cs`, `BpeVocabulary.cs` | BPE model, vocabulary and merges reader, UTF-8 hash tables |
| `Pipeline/BertTokenizerComponents.cs` | `BertNormalizer`, `BertPreTokenizer`, `WordPiece` model and decoder |
| `Pipeline/PostProcessorsAndDecoders.cs`, `Decoders.cs` | `TemplateProcessing`, `ByteLevel` and `Sequence` post-processors (upstream `process_encodings` shape; `PostProcessing.Process` is the trait's provided `process`), and the decoders |
| `Pipeline/SerdeJsonNumber.cs` | Parses JSON numbers with `serde_json`'s rounding (which can differ from a correctly rounded parse by one unit in the last place), so Unigram scores sum and compare exactly as in the reference |
| `Pipeline/*.g.cs` | Generated tables: NFC data from the reference's crate, and the `\p{L}`, `\p{N}`, `\p{M}` classes probed from the reference regex engine (`tools/generate-unicode-tables.py`) |

## Why not .NET built-ins

Each of these choices comes from a measured difference with the reference. The details are in `docs/design/source-audit.md` and `PROVENANCE.md`.

- **Unicode version.** `string.Normalize` follows the runtime's Unicode version (16 in .NET 10). The reference normalizes with Unicode 9.0.0 data, so NFC uses tables copied from the reference's crate.
- **Regex.** .NET `Regex` matches UTF-16 code units, so `\p{L}` misses letters outside the BMP. The reference uses Oniguruma over code points. The Qwen split patterns therefore use a scanner whose character classes were probed from the reference over all 1,112,064 scalar values. Other regexes fail at load rather than risk a mismatch.
- **Microsoft.ML.Tokenizers.** It differs from the reference on unknown-token fusion and empty input, among others. Only its double-array trie and Unigram search were adopted, with local changes recorded in `PROVENANCE.md`. The package itself is never referenced from `src/`.

## Evidence layers

| Layer | Where | What it proves |
| --- | --- | --- |
| Profile conformance | `conformance/<profile>/`, `ProfileConformanceTests` and profile-specific classes | Exact output on a curated multilingual corpus for one pinned file |
| Randomized differential | `TestData/*/differential.jsonl.gz`, `*DifferentialTests` | Exact output on thousands of seeded generated strings and option variants; larger external dumps through `STARLING_*_DIFFERENTIAL_DIR` |
| Component reference | `*ComponentTests` | Each component's options against reference outputs, including variants no profile uses |
| Loader | `LoadFastPathTests`, `BpeLoaderTests`, `PipelineBehaviorTests` | Strictness, fast-path equivalence, error messages |
| Package | `tools/release-drill.sh` | The packed bytes install and pass the e5 fixtures in a clean consumer |
