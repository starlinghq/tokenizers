# Supported `tokenizer.json` components

Behavior follows huggingface/tokenizers v0.23.2 (`88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3`); the Gemma and Qwen3 profiles are checked against 0.23.3 (`9ab7691f497eff905f961aba51a4bb9be3e9555c`), whose library changes against 0.23.2 touch only truncation with overflowing windows (not offered here). Anything not listed as supported fails at load with `UnsupportedTokenizerComponentException`, whose `Component` names the JSON path and type tag (for example `pre_tokenizer.type=ByteLevel`). Unknown fields are rejected too, not ignored.

Every component listed as supported is matched against huggingface/tokenizers on the conformance fixtures of at least one profile ([profiles/](../profiles/)), except where marked experimental. Experimental means the component is implemented and covered by component-level reference cases, but no profile with corpus fixtures uses it yet.

The BERT/WordPiece components (`BertNormalizer`, `BertPreTokenizer`, `WordPiece` model and decoder) are covered by component reference cases and by loading three pinned BERT-family files (`sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2`); they have no profile or corpus fixtures yet.


| Slot | Supported | Unsupported (fails at load) |
| --- | --- | --- |
| `version` | `"1.0"` | anything else, including `"2.0"` |
| `normalizer` | `Sequence`, `Precompiled`, `Strip`, `Replace` with a non-empty `String` pattern (any content, multi-byte replacements aligned as upstream) or the `Regex` `" {2,}"`, `NFC` (Unicode 9.0.0 data, as the reference), `BertNormalizer` with its defaults only (`clean_text`, `handle_chinese_chars` and `lowercase` true, `strip_accents` null; experimental) | `BertNormalizer` with any other option, `StripAccents`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `Nmt`, `Prepend`, `ByteLevel`, other `Replace` regexes |
| `pre_tokenizer` | `Sequence`, `Metaspace` (legacy `add_prefix_space` and current `prepend_scheme`/`split` forms), `Split` with a non-empty `String` pattern (every behavior: `Removed`, `Isolated`, `MergedWithPrevious`, `MergedWithNext`, `Contiguous`; `invert` either), `Split` with one of the two Qwen regexes (below), `Isolated`, `invert` false, `ByteLevel` with `use_regex` false (`add_prefix_space` either), `BertPreTokenizer` (experimental) | `ByteLevel` with `use_regex` true (or absent), `CharDelimiterSplit`, `Whitespace`, `Split` with an empty `String`, any other `Regex`, or a Qwen regex with another behavior or `invert`, `Punctuation`, `WhitespaceSplit`, `Digits`, `UnicodeScripts`, `FixedLength` |
| `model` | `Unigram` (`byte_fallback` false or absent); `BPE` with `dropout` null, empty or absent prefix and suffix, `byte_fallback`, `fuse_unk` and `ignore_merges` either, `unk_token` null or a token of the vocabulary, merges as pairs or legacy `"a b"` strings; `WordPiece` (`unk_token`, `continuing_subword_prefix`, `max_input_chars_per_word`; experimental) | `WordLevel`, Unigram `byte_fallback: true`, BPE dropout or affixes, a BPE `unk_token` missing from the vocabulary (upstream fails only when an unknown character is encoded), duplicate vocabulary keys or ids, untyped models |
| `post_processor` | `TemplateProcessing`, `ByteLevel` (`trim_offsets` either), `Sequence` of supported post-processors (nested allowed, at most one `TemplateProcessing` in total; below), or none | `BertProcessing`, `RobertaProcessing`, a `Sequence` with more than one `TemplateProcessing` |
| `decoder` | `Metaspace`, `ByteLevel`, `Sequence`, `Replace` (same patterns as the normalizer), `ByteFallback`, `Fuse`, `WordPiece` (experimental), or none | `BPEDecoder`, `CTC`, `Strip` |
| `added_tokens` | `special`, `normalized` | `single_word`, `lstrip`, `rstrip` set to true; an `id` that differs from the id the reference would assign |
| `truncation` | `max_length`, `strategy`, `direction` | `stride` other than 0 (no overflowing encodings) |
| `padding` | `BatchLongest`, `Fixed`, `direction`, `pad_to_multiple_of`, ids and token | |

Not offered: pre-tokenized input, word ids, incremental (streaming) decode, encoding special tokens as plain text.

## Post-processor `Sequence` (Qwen3 Embedding)

`{"type": "Sequence", "processors": [...]}` runs each child's `process_encodings` on the previous child's output, as upstream `processors/sequence.rs`; the number of tokens it adds (`added_tokens(is_pair)`, which truncation reserves) is the sum over its children. Before the first child, input `i` gets sequence id and type id `i`; after the last, the resulting encodings are merged in order. A `TemplateProcessing` returns one encoding per template piece, so a `ByteLevel` after it gives every piece its own sequence id, as upstream does (`SequencePostProcessorTests`). A second `TemplateProcessing` would choose its template by that piece count (and upstream panics above two), so a `Sequence` with more than one is rejected at load. `qwen3-bpe-embedding` uses `Sequence[ByteLevel(trim_offsets false), TemplateProcessing("$A <|endoftext|>")]`: truncation keeps `max_length - 1` tokens and the encoding still ends in `<|endoftext|>` (151643).

## Byte fallback (Gemma BPE)

With `byte_fallback` true, a character missing from the BPE vocabulary becomes one `<0xNN>` token per UTF-8 byte (upper-case hex, as upstream's `format!("<{b:#04X}>")`), each carrying the offsets of the whole character; only if one of those byte tokens is missing does the unknown token apply (consecutive unknown characters fused into one with `fuse_unk`). Upstream emits a pending fused unknown token only at the next in-vocabulary character or the end of the word, after any byte tokens produced in between; that order is reproduced (`GemmaBpeComponentTests`). The `ByteFallback` decoder turns each run of consecutive byte tokens into the string they spell if it is valid UTF-8, otherwise into one U+FFFD per byte token (not per ill-formed subsequence as the `ByteLevel` decoder does); token names parse as Rust `u8::from_str_radix(_, 16)` does, so `<0xab>` and `<0x+A>` count as byte tokens.

Streaming: there is no incremental decode API. A caller that decodes growing prefixes (as a generation loop's incremental detokenizer does) sees an incomplete character as trailing U+FFFD and can hold it back. Such prefix decoding is not always prefix-stable with byte fallback: a complete character followed by a stray continuation byte token (for example `<0xC3> <0xA9> <0x80>`) decodes as one invalid run, three U+FFFD, so the earlier `é` is replaced; upstream's `DecodeStream` reports the same situation as an invalid prefix.

## Qwen pre-tokenizer regexes

`Split` accepts exactly two `Regex` patterns, matched by a hand-written scanner (`Pipeline/QwenSplitPattern.cs`) over Unicode scalar values, not by .NET `Regex` (which matches UTF-16 code units, so `\p{L}` misses supplementary-plane letters; P-023 section 3.4):

- Qwen2/Qwen3 (`qwen2-bpe-kev`): `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
- Qwen3.5 (`qwen35-bpe-chat`): the same with `[\p{L}\p{M}]+` and `[^\s\p{L}\p{M}\p{N}]+`.

The scanner reproduces the reference engine's semantics (leftmost match, alternatives in order, greedy quantifiers with backtracking, the lookahead). Its `\p{L}`, `\p{N}` and `\p{M}` are tables generated by probing the reference (Oniguruma in tokenizers 0.23.2) with every Unicode scalar value (`tools/generate-unicode-tables.py`), so they do not depend on the .NET runtime's Unicode version; `\s` is the 25 White_Space characters the probe found; the case-insensitive contractions also accept U+017F for `s`, as the probe found. Verification: the class tables equal the probe for all 1,112,064 scalar values (`QwenSplitPatternTests`), the 137-case fixtures, and the randomized differential runs (`QwenBpeDifferentialTests`).
