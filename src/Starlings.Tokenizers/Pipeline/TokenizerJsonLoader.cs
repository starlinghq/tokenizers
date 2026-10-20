// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Reads tokenizer.json format version 1.0 as written by huggingface/tokenizers 0.x. Field
// names, defaults and validation follow the serde definitions of tokenizers v0.23.2
// (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3), Apache License 2.0. The reader is strict:
// unknown components, unknown keys and unimplemented options throw
// UnsupportedTokenizerComponentException instead of being ignored or approximated.

using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Starlings.Tokenizers.ThirdParty.MLNet;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>The parsed pipeline: every stage of a <c>tokenizer.json</c>.</summary>
internal sealed record TokenizerPipeline(
    INormalizer? Normalizer,
    IPreTokenizer? PreTokenizer,
    IModel Model,
    IPostProcessor? PostProcessor,
    IDecoder? Decoder,
    AddedVocabulary AddedVocabulary,
    TruncationOptions? Truncation,
    PaddingOptions? Padding,
    IReadOnlyList<string> Components,
    bool VocabularyFastPath)
{
    /// <summary>The model as Unigram, or null.</summary>
    public UnigramModel? Unigram => Model as UnigramModel;

    /// <summary>The model as BPE, or null.</summary>
    public BpeModel? Bpe => Model as BpeModel;
}

internal static class TokenizerJsonLoader
{
    private static readonly JsonDocumentOptions _documentOptions = new() { MaxDepth = 64 };

    public static TokenizerPipeline Load(ReadOnlyMemory<byte> utf8Json) => Load(utf8Json, useVocabularyFastPath: true);

    /// <param name="utf8Json">The document.</param>
    /// <param name="useVocabularyFastPath">False forces the document-model path (tests compare the two).</param>
    internal static TokenizerPipeline Load(ReadOnlyMemory<byte> utf8Json, bool useVocabularyFastPath)
    {
        // The Unigram vocabulary is almost all of the file. The fast path reads it from the bytes
        // and parses the rest as a document; when it declines, or the rest is not what it assumed,
        // the unmodified file is loaded through the document model alone, as before.
        if (useVocabularyFastPath && UnigramVocabularyReader.TryRead(utf8Json.Span, out UnigramVocabulary? vocab, out byte[]? rest))
        {
            JsonDocument? restDocument = null;
            try
            {
                restDocument = JsonDocument.Parse(rest, _documentOptions);
            }
            catch (JsonException)
            {
            }

            if (restDocument is not null)
            {
                using (restDocument)
                {
                    if (UnigramVocabularyReader.HasSingleVocab(restDocument.RootElement))
                    {
                        return Load(restDocument.RootElement, vocab, null);
                    }
                }
            }
        }
        else if (useVocabularyFastPath && BpeModelReader.TryRead(utf8Json.Span, out BpeModelData? bpe, out byte[]? bpeRest))
        {
            // The BPE vocab object and merges array are almost all of the file: same scheme.
            JsonDocument? restDocument = null;
            try
            {
                restDocument = JsonDocument.Parse(bpeRest, _documentOptions);
            }
            catch (JsonException)
            {
            }

            if (restDocument is not null)
            {
                using (restDocument)
                {
                    if (BpeModelReader.HasSingleVocabAndMerges(restDocument.RootElement))
                    {
                        return Load(restDocument.RootElement, null, bpe);
                    }
                }
            }
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(utf8Json, _documentOptions);
        }
        catch (JsonException e)
        {
            throw new TokenizerFormatException("tokenizer.json is not valid JSON.", e);
        }

        using (document)
        {
            return Load(document.RootElement, null, null);
        }
    }

    private static TokenizerPipeline Load(JsonElement root, UnigramVocabulary? scannedVocab, BpeModelData? scannedBpe)
    {
        RequireObject(root, "tokenizer.json");
        AllowKeys(root, "tokenizer.json", "version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model");

        string version = RequireString(root, "version", "tokenizer.json");
        if (version != "1.0")
        {
            throw new UnsupportedTokenizerComponentException($"version={version}", "only tokenizer.json format version 1.0 is supported.");
        }

        var components = new List<string>();
        INormalizer? normalizer = ReadOptional(root, "normalizer", e => ReadNormalizer(e, "normalizer", components));
        IPreTokenizer? preTokenizer = ReadOptional(root, "pre_tokenizer", e => ReadPreTokenizer(e, "pre_tokenizer", components));
        IModel model = ReadModel(Require(root, "model", "tokenizer.json"), scannedVocab, scannedBpe, components);
        AddedVocabulary added = ReadAddedTokens(root, model, normalizer, components);
        IPostProcessor? post = ReadOptional(root, "post_processor", e => ReadPostProcessor(e, components));
        IDecoder? decoder = ReadOptional(root, "decoder", e => ReadDecoder(e, components));
        TruncationOptions? truncation = ReadOptional(root, "truncation", ReadTruncation);
        PaddingOptions? padding = ReadOptional(root, "padding", ReadPadding);

        if (truncation is not null && post is not null && truncation.MaxLength < Math.Max(post.AddedSingle, post.AddedPair))
        {
            throw new UnsupportedTokenizerComponentException("truncation.max_length", "is smaller than the number of special tokens the post-processor adds.");
        }

        return new TokenizerPipeline(
            normalizer, preTokenizer, model, post, decoder, added, truncation, padding, components, scannedVocab is not null || scannedBpe is not null);
    }

    private static INormalizer ReadNormalizer(JsonElement e, string path, List<string> components)
    {
        string type = RequireType(e, path);
        string where = $"{path}.type={type}";
        switch (type)
        {
            case "Sequence":
                {
                    AllowKeys(e, where, "type", "normalizers");
                    var list = new List<INormalizer>();
                    int i = 0;
                    foreach (JsonElement child in RequireArray(e, "normalizers", where))
                    {
                        list.Add(ReadNormalizer(child, $"{path}.normalizers[{i++}]", components));
                    }

                    components.Add("normalizer:Sequence");
                    return new SequenceNormalizer(list);
                }

            case "Precompiled":
                {
                    AllowKeys(e, where, "type", "precompiled_charsmap");
                    string b64 = RequireString(e, "precompiled_charsmap", where);
                    byte[] charsmap;
                    try
                    {
                        charsmap = Convert.FromBase64String(b64);
                    }
                    catch (FormatException ex)
                    {
                        throw new TokenizerFormatException($"{where}: precompiled_charsmap is not valid base64.", ex);
                    }

                    components.Add("normalizer:Precompiled");
                    return new PrecompiledNormalizer(charsmap);
                }

            case "Replace":
                {
                    (IPattern matcher, string content) = ReadReplace(e, where);
                    components.Add("normalizer:Replace");
                    return new ReplaceNormalizer(matcher, content);
                }

            case "Strip":
                {
                    AllowKeys(e, where, "type", "strip_left", "strip_right");
                    components.Add("normalizer:Strip");
                    return new StripNormalizer(RequireBool(e, "strip_left", where), RequireBool(e, "strip_right", where));
                }

            case "NFC":
                {
                    AllowKeys(e, where, "type");
                    components.Add("normalizer:NFC");
                    return new NfcNormalizer();
                }

            case "BertNormalizer":
                {
                    AllowKeys(e, where, "type", "clean_text", "handle_chinese_chars", "strip_accents", "lowercase");
                    bool cleanText = OptionalBool(e, "clean_text", where, defaultValue: true);
                    bool handleChineseChars = OptionalBool(e, "handle_chinese_chars", where, defaultValue: true);
                    bool? stripAccents = OptionalNullableBool(e, "strip_accents", where);
                    bool lowercase = OptionalBool(e, "lowercase", where, defaultValue: true);
                    if (!cleanText || !handleChineseChars || stripAccents is not null || !lowercase)
                    {
                        throw new UnsupportedTokenizerComponentException(
                            where,
                            "only the pinned encoder profile is supported (clean_text=true, handle_chinese_chars=true, strip_accents=null, lowercase=true).");
                    }

                    components.Add("normalizer:BertNormalizer");
                    return new BertNormalizer(cleanText, handleChineseChars, stripAccents, lowercase);
                }

            default:
                throw new UnsupportedTokenizerComponentException(where, "normalizer type is not implemented.");
        }
    }

    /// <summary>
    /// <c>Replace</c> fields (normalizer and decoder share upstream's <c>Replace</c>): a non-empty
    /// <c>String</c> pattern, or the <c>Regex</c> <c>" {2,}"</c>.
    /// </summary>
    private static (IPattern Pattern, string Content) ReadReplace(JsonElement e, string where)
    {
        AllowKeys(e, where, "type", "pattern", "content");
        JsonElement pattern = Require(e, "pattern", where);
        RequireObject(pattern, $"{where}.pattern");
        string content = RequireString(e, "content", where);
        if (pattern.TryGetProperty("String", out JsonElement literal) && CountProperties(pattern) == 1)
        {
            string s = literal.GetString() ?? string.Empty;
            if (s.Length == 0)
            {
                throw new UnsupportedTokenizerComponentException(where, "an empty String pattern is not supported.");
            }

            return (SimplePattern.Literal(s), content);
        }

        if (pattern.TryGetProperty("Regex", out JsonElement regex) && CountProperties(pattern) == 1)
        {
            string r = regex.GetString() ?? string.Empty;
            return r switch
            {
                " {2,}" => (SimplePattern.ByteRun((byte)' ', 2), content),
                _ => throw new UnsupportedTokenizerComponentException(
                    where, $"Regex pattern '{r}' is not on the supported list (\" {{2,}}\"); Oniguruma regexes are not emulated."),
            };
        }

        throw new TokenizerFormatException($"{where}: pattern must be {{\"String\": ...}} or {{\"Regex\": ...}}.");
    }

    private static IPreTokenizer ReadPreTokenizer(JsonElement e, string path, List<string> components)
    {
        string type = RequireType(e, path);
        string where = $"{path}.type={type}";
        switch (type)
        {
            case "Sequence":
                {
                    AllowKeys(e, where, "type", "pretokenizers");
                    var list = new List<IPreTokenizer>();
                    int i = 0;
                    foreach (JsonElement child in RequireArray(e, "pretokenizers", where))
                    {
                        list.Add(ReadPreTokenizer(child, $"{path}.pretokenizers[{i++}]", components));
                    }

                    components.Add("pre_tokenizer:Sequence");
                    return new SequencePreTokenizer(list);
                }

            case "Metaspace":
                {
                    (Rune replacement, PrependScheme scheme, bool split) = ReadMetaspace(e, where);
                    components.Add("pre_tokenizer:Metaspace");
                    return new MetaspacePreTokenizer(replacement, scheme, split);
                }

            case "Split":
                {
                    AllowKeys(e, where, "type", "pattern", "behavior", "invert");
                    JsonElement pattern = Require(e, "pattern", where);
                    RequireObject(pattern, $"{where}.pattern");
                    if (pattern.TryGetProperty("String", out JsonElement literal) && CountProperties(pattern) == 1)
                    {
                        // A String pattern is an escaped literal for the reference's regex engine:
                        // leftmost, non-overlapping occurrences. Every behavior and invert apply.
                        string s = literal.ValueKind == JsonValueKind.String
                            ? literal.GetString()!
                            : throw new TokenizerFormatException($"{where}.pattern: String must be a string.");
                        if (s.Length == 0)
                        {
                            throw new UnsupportedTokenizerComponentException($"{where}.pattern", "an empty String pattern is not supported.");
                        }

                        SplitDelimiterBehavior splitBehavior = ReadSplitBehavior(e, where);
                        IPattern stringPattern = SimplePattern.Literal(s);
                        if (RequireBool(e, "invert", where))
                        {
                            stringPattern = new InvertedPattern(stringPattern);
                        }

                        components.Add("pre_tokenizer:Split");
                        return new SplitPreTokenizer(stringPattern, splitBehavior);
                    }

                    if (!pattern.TryGetProperty("Regex", out JsonElement regex) || CountProperties(pattern) != 1 || regex.ValueKind != JsonValueKind.String)
                    {
                        throw new UnsupportedTokenizerComponentException($"{where}.pattern", "only a String or a Regex pattern is supported.");
                    }

                    string r = regex.GetString()!;
                    QwenSplitPattern matcher = QwenSplitPattern.TryCreate(r) ?? throw new UnsupportedTokenizerComponentException(
                        $"{where}.pattern", $"Regex '{r}' is not one of the supported Qwen pre-tokenizer patterns; Oniguruma regexes are not emulated in general.");
                    string behavior = RequireString(e, "behavior", where);
                    if (behavior != "Isolated")
                    {
                        throw new UnsupportedTokenizerComponentException($"{where}.behavior", $"behavior '{behavior}' is not implemented for Regex patterns (only Isolated).");
                    }

                    if (RequireBool(e, "invert", where))
                    {
                        throw new UnsupportedTokenizerComponentException($"{where}.invert", "invert true is not implemented.");
                    }

                    components.Add("pre_tokenizer:Split");
                    return new SplitPreTokenizer(matcher, SplitDelimiterBehavior.Isolated);
                }

            case "ByteLevel":
                {
                    (bool addPrefixSpace, _, bool useRegex) = ReadByteLevel(e, where);
                    components.Add("pre_tokenizer:ByteLevel");
                    return new ByteLevelPreTokenizer(addPrefixSpace, useRegex);
                }

            case "BertPreTokenizer":
                AllowKeys(e, where, "type");
                components.Add("pre_tokenizer:BertPreTokenizer");
                return new BertPreTokenizer();

            default:
                throw new UnsupportedTokenizerComponentException(where, "pre-tokenizer type is not implemented.");
        }
    }

    /// <summary>Upstream <c>SplitDelimiterBehavior</c> names.</summary>
    private static SplitDelimiterBehavior ReadSplitBehavior(JsonElement e, string where) =>
        RequireString(e, "behavior", where) switch
        {
            "Removed" => SplitDelimiterBehavior.Removed,
            "Isolated" => SplitDelimiterBehavior.Isolated,
            "MergedWithPrevious" => SplitDelimiterBehavior.MergedWithPrevious,
            "MergedWithNext" => SplitDelimiterBehavior.MergedWithNext,
            "Contiguous" => SplitDelimiterBehavior.Contiguous,
            var other => throw new TokenizerFormatException($"{where}.behavior: unknown value '{other}'."),
        };

    /// <summary>Metaspace fields, legacy (<c>add_prefix_space</c>) and current forms.</summary>
    private static (Rune Replacement, PrependScheme Scheme, bool Split) ReadMetaspace(JsonElement e, string where)
    {
        AllowKeys(e, where, "type", "replacement", "add_prefix_space", "prepend_scheme", "split", "str_rep");
        string replacementText = RequireString(e, "replacement", where);
        if (!Rune.TryGetRuneAt(replacementText, 0, out Rune replacement) || replacement.Utf16SequenceLength != replacementText.Length)
        {
            throw new TokenizerFormatException($"{where}: replacement must be exactly one character.");
        }

        PrependScheme scheme = PrependScheme.Always;
        if (e.TryGetProperty("prepend_scheme", out JsonElement ps))
        {
            scheme = ps.GetString() switch
            {
                "always" => PrependScheme.Always,
                "first" => PrependScheme.First,
                "never" => PrependScheme.Never,
                var other => throw new UnsupportedTokenizerComponentException($"{where}.prepend_scheme", $"value '{other}' is not known."),
            };
        }

        if (e.TryGetProperty("add_prefix_space", out JsonElement aps) && aps.ValueKind != JsonValueKind.Null)
        {
            if (!aps.GetBoolean())
            {
                if (scheme != PrependScheme.Never)
                {
                    throw new TokenizerFormatException($"{where}: add_prefix_space does not match declared prepend_scheme");
                }

                scheme = PrependScheme.Never;
            }
        }

        bool split = !e.TryGetProperty("split", out JsonElement sp) || sp.ValueKind == JsonValueKind.Null || sp.GetBoolean();
        return (replacement, scheme, split);
    }

    /// <summary><c>ByteLevel</c> fields; <c>use_regex</c> defaults to true as upstream.</summary>
    private static (bool AddPrefixSpace, bool TrimOffsets, bool UseRegex) ReadByteLevel(JsonElement e, string where)
    {
        AllowKeys(e, where, "type", "add_prefix_space", "trim_offsets", "use_regex");
        bool useRegex = true;
        if (e.TryGetProperty("use_regex", out JsonElement ur))
        {
            useRegex = ur.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => throw new TokenizerFormatException($"{where}: field 'use_regex' must be a boolean."),
            };
        }

        return (RequireBool(e, "add_prefix_space", where), RequireBool(e, "trim_offsets", where), useRegex);
    }

    private static IModel ReadModel(JsonElement e, UnigramVocabulary? scannedVocab, BpeModelData? scannedBpe, List<string> components)
    {
        RequireObject(e, "model");
        if (!e.TryGetProperty("type", out JsonElement typeElement))
        {
            throw new UnsupportedTokenizerComponentException("model", "a model without a type tag is not supported.");
        }

        string type = typeElement.GetString() ?? string.Empty;
        string where = $"model.type={type}";
        if (type == "BPE")
        {
            return ReadBpeModel(e, where, scannedBpe, components);
        }

        if (type == "WordPiece")
        {
            return ReadWordPieceModel(e, where, components);
        }

        if (type != "Unigram" || scannedBpe is not null)
        {
            throw new UnsupportedTokenizerComponentException(where, "model type is not implemented.");
        }

        AllowKeys(e, where, "type", "unk_id", "vocab", "byte_fallback");
        if (e.TryGetProperty("byte_fallback", out JsonElement bf) && bf.ValueKind == JsonValueKind.True)
        {
            throw new UnsupportedTokenizerComponentException($"{where}.byte_fallback", "byte fallback is not implemented.");
        }

        int? unkId = null;
        if (e.TryGetProperty("unk_id", out JsonElement unk) && unk.ValueKind != JsonValueKind.Null)
        {
            unkId = unk.GetInt32();
        }

        // With a scanned vocabulary the document's vocab is the empty placeholder array.
        JsonElement.ArrayEnumerator entries = RequireArray(e, "vocab", where);
        UnigramVocabulary vocab = scannedVocab ?? new UnigramVocabulary();
        foreach (JsonElement entry in entries)
        {
            if (entry.ValueKind != JsonValueKind.Array || entry.GetArrayLength() != 2)
            {
                throw new TokenizerFormatException($"{where}: vocab entries must be [piece, score].");
            }

            if (entry[0].ValueKind != JsonValueKind.String || entry[1].ValueKind != JsonValueKind.Number)
            {
                throw new TokenizerFormatException($"{where}: vocab entries must be [string, number].");
            }

            // Scores are parsed as the reference's serde_json does, not correctly rounded.
            vocab.Add(entry[0].GetString()!, SerdeJsonNumber.ParseDouble(JsonMarshal.GetRawUtf8Value(entry[1])));
        }

        components.Add("model:Unigram");
        return new UnigramModel(vocab, unkId);
    }

    private static WordPieceModel ReadWordPieceModel(JsonElement e, string where, List<string> components)
    {
        AllowKeys(e, where, "type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word", "vocab");
        string unkToken = OptionalString(e, "unk_token", where, "[UNK]");
        string prefix = OptionalString(e, "continuing_subword_prefix", where, "##");
        int maxInputChars = OptionalInt32(e, "max_input_chars_per_word", where, 100);
        if (maxInputChars <= 0)
        {
            throw new TokenizerFormatException($"{where}: max_input_chars_per_word must be a positive integer.");
        }

        JsonElement vocabElement = Require(e, "vocab", where);
        RequireObject(vocabElement, $"{where}.vocab");
        var vocab = new BpeVocabulary(CountProperties(vocabElement), expectedBytes: CountProperties(vocabElement) * 8);
        foreach (JsonProperty entry in vocabElement.EnumerateObject())
        {
            if (entry.Value.ValueKind != JsonValueKind.Number || !entry.Value.TryGetInt32(out int id) || id < 0)
            {
                throw new TokenizerFormatException($"{where}.vocab: token ids must be nonnegative signed 32-bit integers.");
            }

            vocab.Add(entry.Name, id);
        }

        vocab.Seal();
        components.Add("model:WordPiece");
        return new WordPieceModel(vocab, unkToken, prefix, maxInputChars);
    }

    /// <summary>
    /// BPE model (upstream <c>BPEVisitor</c> and <c>BpeBuilder::build</c>). Supported: no dropout,
    /// empty or absent prefix and suffix, either merges form (pairs or legacy <c>"a b"</c> strings),
    /// <c>ignore_merges</c>, <c>byte_fallback</c> and <c>fuse_unk</c> true or false, an
    /// <c>unk_token</c> that is in the vocabulary (or null).
    /// </summary>
    private static BpeModel ReadBpeModel(JsonElement e, string where, BpeModelData? scanned, List<string> components)
    {
        AllowKeys(e, where, "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges");
        if (e.TryGetProperty("dropout", out JsonElement dropout) && dropout.ValueKind != JsonValueKind.Null)
        {
            throw new UnsupportedTokenizerComponentException($"{where}.dropout", "BPE dropout is not implemented (only null).");
        }

        string? unkToken = null;
        if (e.TryGetProperty("unk_token", out JsonElement unk) && unk.ValueKind != JsonValueKind.Null)
        {
            unkToken = unk.ValueKind == JsonValueKind.String
                ? unk.GetString()!
                : throw new TokenizerFormatException($"{where}: field 'unk_token' must be a string or null.");
        }

        foreach (string affix in new[] { "continuing_subword_prefix", "end_of_word_suffix" })
        {
            if (e.TryGetProperty(affix, out JsonElement a) && a.ValueKind != JsonValueKind.Null
                && !(a.ValueKind == JsonValueKind.String && a.GetString()!.Length == 0))
            {
                throw new UnsupportedTokenizerComponentException($"{where}.{affix}", "a non-empty value is not implemented.");
            }
        }

        bool byteFallback = OptionalBool(e, "byte_fallback", where);
        bool fuseUnk = OptionalBool(e, "fuse_unk", where);
        bool ignoreMerges = OptionalBool(e, "ignore_merges", where);

        JsonElement vocabElement = Require(e, "vocab", where);
        JsonElement mergesElement = Require(e, "merges", where);
        BpeModelData data;
        if (scanned is not null)
        {
            // The document holds placeholders where the scanned vocab and merges were.
            if (vocabElement.ValueKind != JsonValueKind.Object || mergesElement.ValueKind != JsonValueKind.Array
                || CountProperties(vocabElement) != 0 || mergesElement.GetArrayLength() != 0)
            {
                throw new InvalidOperationException("BPE fast path placeholders are missing.");
            }

            data = scanned;
        }
        else
        {
            data = BpeModelReader.ReadDocument(vocabElement, mergesElement, where);
        }

        int unkId = -1;
        if (unkToken is not null)
        {
            // Upstream looks the token up only when an unknown character occurs and fails the
            // encode then; a missing one is rejected here instead of failing on some inputs.
            unkId = data.Vocabulary.Find(unkToken);
            if (unkId < 0)
            {
                throw new UnsupportedTokenizerComponentException($"{where}.unk_token", $"unk_token '{unkToken}' is not in the vocabulary.");
            }
        }

        components.Add("model:BPE");
        return new BpeModel(data.Vocabulary, data.Merges, ignoreMerges, unkId, fuseUnk, byteFallback);
    }

    private static bool OptionalBool(JsonElement e, string name, string where)
    {
        if (!e.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
        {
            return false;
        }

        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new TokenizerFormatException($"{where}: field '{name}' must be a boolean."),
        };
    }

    private static AddedVocabulary ReadAddedTokens(JsonElement root, IModel model, INormalizer? normalizer, List<string> components)
    {
        var tokens = new List<(int, AddedToken)>();
        if (root.TryGetProperty("added_tokens", out JsonElement list) && list.ValueKind != JsonValueKind.Null)
        {
            int i = 0;
            foreach (JsonElement t in list.EnumerateArray())
            {
                string where = $"added_tokens[{i++}]";
                RequireObject(t, where);
                AllowKeys(t, where, "id", "content", "single_word", "lstrip", "rstrip", "normalized", "special");
                if (RequireBool(t, "single_word", where))
                {
                    throw new UnsupportedTokenizerComponentException($"{where}.single_word", "added-token option is not implemented.");
                }

                var token = new AddedToken(
                    RequireString(t, "content", where),
                    RequireBool(t, "special", where),
                    RequireBool(t, "normalized", where),
                    RequireBool(t, "lstrip", where),
                    RequireBool(t, "rstrip", where));
                tokens.Add((Require(t, "id", where).GetInt32(), token));
            }
        }

        components.Add("added_tokens");
        return new AddedVocabulary(tokens, model, normalizer);
    }

    private static IPostProcessor ReadPostProcessor(JsonElement e, List<string> components)
    {
        string type = RequireType(e, "post_processor");
        string where = $"post_processor.type={type}";
        if (type == "ByteLevel")
        {
            (bool addPrefixSpace, bool trimOffsets, _) = ReadByteLevel(e, where);
            components.Add("post_processor:ByteLevel");
            return new ByteLevelPostProcessor(addPrefixSpace, trimOffsets);
        }

        if (type != "TemplateProcessing")
        {
            throw new UnsupportedTokenizerComponentException(where, "post-processor type is not implemented.");
        }

        AllowKeys(e, where, "type", "single", "pair", "special_tokens");
        var special = new Dictionary<string, TemplateSpecialToken>(StringComparer.Ordinal);
        JsonElement specialElement = Require(e, "special_tokens", where);
        RequireObject(specialElement, $"{where}.special_tokens");
        foreach (JsonProperty p in specialElement.EnumerateObject())
        {
            string sw = $"{where}.special_tokens.{p.Name}";
            AllowKeys(p.Value, sw, "id", "ids", "tokens");
            int[] ids = [.. RequireArray(p.Value, "ids", sw).Select(x => x.GetInt32())];
            string[] toks = [.. RequireArray(p.Value, "tokens", sw).Select(x => x.GetString() ?? string.Empty)];
            if (ids.Length != toks.Length)
            {
                throw new TokenizerFormatException($"{sw}: ids and tokens differ in length.");
            }

            special[p.Name] = new TemplateSpecialToken(RequireString(p.Value, "id", sw), ids, toks);
        }

        List<TemplatePiece> single = ReadTemplate(Require(e, "single", where), $"{where}.single", special, allowB: false);
        List<TemplatePiece> pair = ReadTemplate(Require(e, "pair", where), $"{where}.pair", special, allowB: true);
        components.Add("post_processor:TemplateProcessing");
        return new TemplateProcessing(single, pair, special);
    }

    private static List<TemplatePiece> ReadTemplate(
        JsonElement e, string where, Dictionary<string, TemplateSpecialToken> special, bool allowB)
    {
        var pieces = new List<TemplatePiece>();
        foreach (JsonElement item in e.EnumerateArray())
        {
            RequireObject(item, where);
            if (item.TryGetProperty("Sequence", out JsonElement seq) && CountProperties(item) == 1)
            {
                AllowKeys(seq, where, "id", "type_id");
                int index = RequireString(seq, "id", where) switch
                {
                    "A" => 0,
                    "B" => 1,
                    var other => throw new TokenizerFormatException($"{where}: unknown sequence id '{other}'."),
                };
                pieces.Add(new SequencePiece(index, Require(seq, "type_id", where).GetInt32()));
            }
            else if (item.TryGetProperty("SpecialToken", out JsonElement tok) && CountProperties(item) == 1)
            {
                AllowKeys(tok, where, "id", "type_id");
                string id = RequireString(tok, "id", where);
                if (!special.ContainsKey(id))
                {
                    throw new TokenizerFormatException($"{where}: special token '{id}' is missing from special_tokens.");
                }

                pieces.Add(new SpecialTokenPiece(id, Require(tok, "type_id", where).GetInt32()));
            }
            else
            {
                throw new TokenizerFormatException($"{where}: template pieces must be Sequence or SpecialToken.");
            }
        }

        int countA = pieces.OfType<SequencePiece>().Count(p => p.Sequence == 0);
        int countB = pieces.OfType<SequencePiece>().Count(p => p.Sequence == 1);
        if (countA != 1 || countB != (allowB ? 1 : 0))
        {
            throw new UnsupportedTokenizerComponentException(where, "templates must use $A exactly once (and $B exactly once in the pair template).");
        }

        return pieces;
    }

    private static IDecoder ReadDecoder(JsonElement e, List<string> components) => ReadDecoder(e, "decoder", components);

    private static IDecoder ReadDecoder(JsonElement e, string path, List<string> components)
    {
        string type = RequireType(e, path);
        string where = $"{path}.type={type}";
        switch (type)
        {
            case "ByteLevel":
                // The decoder ignores all three options (upstream decode_chain).
                _ = ReadByteLevel(e, where);
                components.Add("decoder:ByteLevel");
                return new ByteLevelDecoder();

            case "Metaspace":
                {
                    (Rune replacement, PrependScheme scheme, _) = ReadMetaspace(e, where);
                    components.Add("decoder:Metaspace");
                    return new MetaspaceDecoder(replacement, scheme);
                }

            case "Sequence":
                {
                    AllowKeys(e, where, "type", "decoders");
                    var list = new List<IDecoder>();
                    int i = 0;
                    foreach (JsonElement child in RequireArray(e, "decoders", where))
                    {
                        list.Add(ReadDecoder(child, $"{path}.decoders[{i++}]", components));
                    }

                    components.Add("decoder:Sequence");
                    return new SequenceDecoder(list);
                }

            case "Replace":
                {
                    (IPattern pattern, string content) = ReadReplace(e, where);
                    components.Add("decoder:Replace");
                    return new ReplaceDecoder(pattern, content);
                }

            case "ByteFallback":
                AllowKeys(e, where, "type");
                components.Add("decoder:ByteFallback");
                return new ByteFallbackDecoder();

            case "Fuse":
                AllowKeys(e, where, "type");
                components.Add("decoder:Fuse");
                return new FuseDecoder();

            case "WordPiece":
                AllowKeys(e, where, "type", "prefix", "cleanup");
                string prefix = OptionalString(e, "prefix", where, "##");
                bool cleanup = OptionalBool(e, "cleanup", where, defaultValue: true);
                components.Add("decoder:WordPiece");
                return new WordPieceDecoder(prefix, cleanup);

            default:
                throw new UnsupportedTokenizerComponentException(where, "decoder type is not implemented.");
        }
    }

    private static TruncationOptions ReadTruncation(JsonElement e)
    {
        const string where = "truncation";
        AllowKeys(e, where, "direction", "max_length", "strategy", "stride");
        if (Require(e, "stride", where).GetInt32() != 0)
        {
            throw new UnsupportedTokenizerComponentException("truncation.stride", "a non-zero stride (overflowing windows) is not implemented.");
        }

        TruncationDirection direction = TruncationDirection.Right;
        if (e.TryGetProperty("direction", out JsonElement d))
        {
            direction = d.GetString() switch
            {
                "Right" => TruncationDirection.Right,
                "Left" => TruncationDirection.Left,
                var other => throw new TokenizerFormatException($"truncation.direction: unknown value '{other}'."),
            };
        }

        TruncationStrategy strategy = RequireString(e, "strategy", where) switch
        {
            "LongestFirst" => TruncationStrategy.LongestFirst,
            "OnlyFirst" => TruncationStrategy.OnlyFirst,
            "OnlySecond" => TruncationStrategy.OnlySecond,
            var other => throw new TokenizerFormatException($"truncation.strategy: unknown value '{other}'."),
        };

        return new TruncationOptions(Require(e, "max_length", where).GetInt32(), strategy, direction);
    }

    private static PaddingOptions ReadPadding(JsonElement e)
    {
        const string where = "padding";
        AllowKeys(e, where, "strategy", "direction", "pad_to_multiple_of", "pad_id", "pad_type_id", "pad_token");
        JsonElement strategy = Require(e, "strategy", where);
        int? fixedLength = null;
        if (strategy.ValueKind == JsonValueKind.Object && strategy.TryGetProperty("Fixed", out JsonElement f))
        {
            fixedLength = f.GetInt32();
        }
        else if (!(strategy.ValueKind == JsonValueKind.String && strategy.GetString() == "BatchLongest"))
        {
            throw new TokenizerFormatException("padding.strategy: expected \"BatchLongest\" or {\"Fixed\": n}.");
        }

        PaddingDirection direction = RequireString(e, "direction", where) switch
        {
            "Right" => PaddingDirection.Right,
            "Left" => PaddingDirection.Left,
            var other => throw new TokenizerFormatException($"padding.direction: unknown value '{other}'."),
        };

        int? multiple = e.TryGetProperty("pad_to_multiple_of", out JsonElement m) && m.ValueKind != JsonValueKind.Null
            ? m.GetInt32()
            : null;

        return new PaddingOptions(
            Require(e, "pad_id", where).GetInt32(),
            RequireString(e, "pad_token", where),
            fixedLength,
            direction,
            Require(e, "pad_type_id", where).GetInt32(),
            multiple);
    }

    private static T? ReadOptional<T>(JsonElement root, string name, Func<JsonElement, T> read)
        where T : class =>
        root.TryGetProperty(name, out JsonElement e) && e.ValueKind != JsonValueKind.Null ? read(e) : null;

    private static JsonElement Require(JsonElement e, string name, string where) =>
        e.TryGetProperty(name, out JsonElement value)
            ? value
            : throw new TokenizerFormatException($"{where}: missing field '{name}'.");

    private static string RequireString(JsonElement e, string name, string where)
    {
        JsonElement value = Require(e, name, where);
        return value.ValueKind == JsonValueKind.String
            ? value.GetString()!
            : throw new TokenizerFormatException($"{where}: field '{name}' must be a string.");
    }

    private static bool RequireBool(JsonElement e, string name, string where)
    {
        JsonElement value = Require(e, name, where);
        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new TokenizerFormatException($"{where}: field '{name}' must be a boolean."),
        };
    }

    private static bool OptionalBool(JsonElement e, string name, string where, bool defaultValue)
    {
        if (!e.TryGetProperty(name, out JsonElement value))
        {
            return defaultValue;
        }

        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new TokenizerFormatException($"{where}: field '{name}' must be a boolean or null."),
        };
    }

    private static bool? OptionalNullableBool(JsonElement e, string name, string where)
    {
        if (!e.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw new TokenizerFormatException($"{where}: field '{name}' must be a boolean or null."),
        };
    }

    private static string OptionalString(JsonElement e, string name, string where, string defaultValue)
    {
        if (!e.TryGetProperty(name, out JsonElement value))
        {
            return defaultValue;
        }

        return value.ValueKind == JsonValueKind.String
            ? value.GetString()!
            : throw new TokenizerFormatException($"{where}: field '{name}' must be a string or null.");
    }

    private static int OptionalInt32(JsonElement e, string name, string where, int defaultValue)
    {
        if (!e.TryGetProperty(name, out JsonElement value))
        {
            return defaultValue;
        }

        return value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int parsed)
            ? parsed
            : throw new TokenizerFormatException($"{where}: field '{name}' must be a 32-bit integer or null.");
    }

    private static JsonElement.ArrayEnumerator RequireArray(JsonElement e, string name, string where)
    {
        JsonElement value = Require(e, name, where);
        return value.ValueKind == JsonValueKind.Array
            ? value.EnumerateArray()
            : throw new TokenizerFormatException($"{where}: field '{name}' must be an array.");
    }

    private static string RequireType(JsonElement e, string path)
    {
        RequireObject(e, path);
        return RequireString(e, "type", path);
    }

    private static void RequireObject(JsonElement e, string where)
    {
        if (e.ValueKind != JsonValueKind.Object)
        {
            throw new TokenizerFormatException($"{where}: expected a JSON object.");
        }
    }

    private static void AllowKeys(JsonElement e, string where, params string[] allowed)
    {
        foreach (JsonProperty p in e.EnumerateObject())
        {
            if (Array.IndexOf(allowed, p.Name) < 0)
            {
                throw new UnsupportedTokenizerComponentException($"{where}.{p.Name}", "field is not recognized; refusing to ignore it.");
            }
        }
    }

    private static int CountProperties(JsonElement e)
    {
        int n = 0;
        foreach (JsonProperty _ in e.EnumerateObject())
        {
            n++;
        }

        return n;
    }
}
