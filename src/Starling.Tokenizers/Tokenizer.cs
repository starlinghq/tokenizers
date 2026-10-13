// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Encode, post-process, truncation, padding and decode flow ported from huggingface/tokenizers
// v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3) tokenizers/src/tokenizer/mod.rs and
// tokenizers/src/utils/{truncation,padding}.rs, Copyright The HuggingFace Inc. team,
// Apache License 2.0 (see THIRD-PARTY-NOTICES.md). Modified by StarlingHq: C# API.

using Starling.Tokenizers.Pipeline;

namespace Starling.Tokenizers;

/// <summary>
/// A tokenizer built from a Hugging Face <c>tokenizer.json</c> (format version 1.0). The file's
/// pipeline (normalizer, pre-tokenizer, model, post-processor, decoder, added tokens) is
/// interpreted stage by stage; a component this library does not implement is rejected with
/// <see cref="UnsupportedTokenizerComponentException"/> when the file is loaded.
/// </summary>
/// <remarks>Instances are immutable and safe to use from multiple threads.</remarks>
public sealed class Tokenizer
{
    private readonly TokenizerPipeline _pipeline;

    private Tokenizer(TokenizerPipeline pipeline, TruncationOptions? truncation, PaddingOptions? padding)
    {
        _pipeline = pipeline;
        Truncation = truncation;
        Padding = padding;
    }

    /// <summary>Truncation applied by every encode call, or null for none.</summary>
    public TruncationOptions? Truncation { get; }

    /// <summary>Padding applied by every encode call, or null for none.</summary>
    public PaddingOptions? Padding { get; }

    /// <summary>The pipeline components this tokenizer runs, as <c>slot:Type</c> names.</summary>
    public IReadOnlyList<string> Components => _pipeline.Components;

    /// <summary>Loads a <c>tokenizer.json</c> file.</summary>
    /// <exception cref="UnsupportedTokenizerComponentException">The file uses an unsupported component or option.</exception>
    /// <exception cref="TokenizerFormatException">The file is malformed.</exception>
    public static Tokenizer FromFile(string path) => FromUtf8Json(File.ReadAllBytes(path));

    /// <summary>Loads a <c>tokenizer.json</c> document from a stream.</summary>
    public static Tokenizer FromStream(Stream utf8Json)
    {
        ArgumentNullException.ThrowIfNull(utf8Json);
        using var buffer = new MemoryStream();
        utf8Json.CopyTo(buffer);
        return FromUtf8Json(buffer.ToArray());
    }

    /// <summary>Loads a <c>tokenizer.json</c> document from UTF-8 bytes.</summary>
    public static Tokenizer FromUtf8Json(ReadOnlyMemory<byte> utf8Json)
    {
        TokenizerPipeline pipeline = TokenizerJsonLoader.Load(utf8Json);
        return new Tokenizer(pipeline, pipeline.Truncation, pipeline.Padding);
    }

    /// <summary>Returns a tokenizer with the same pipeline and different truncation.</summary>
    public Tokenizer WithTruncation(TruncationOptions? truncation)
    {
        if (truncation is not null)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(truncation.MaxLength, nameof(truncation));
            int added = _pipeline.PostProcessor is { } p ? Math.Max(p.AddedSingle, p.AddedPair) : 0;
            if (truncation.MaxLength < added)
            {
                throw new ArgumentException(
                    $"MaxLength {truncation.MaxLength} is smaller than the {added} special tokens the post-processor adds.",
                    nameof(truncation));
            }
        }

        return new Tokenizer(_pipeline, truncation, Padding);
    }

    /// <summary>Returns a tokenizer with the same pipeline and different padding.</summary>
    public Tokenizer WithPadding(PaddingOptions? padding)
    {
        if (padding is not null)
        {
            ArgumentNullException.ThrowIfNull(padding.PadToken, nameof(padding));
        }

        return new Tokenizer(_pipeline, Truncation, padding);
    }

    /// <summary>Encodes one text.</summary>
    /// <param name="text">Input text. Unpaired UTF-16 surrogates are rejected.</param>
    /// <param name="addSpecialTokens">Apply the post-processor's special tokens (reference default: true).</param>
    public TokenEncoding Encode(string text, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        return EncodeCore(text, null, addSpecialTokens).Build();
    }

    /// <summary>Encodes a text pair (for example a query and a passage).</summary>
    public TokenEncoding Encode(string text, string pair, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(pair);
        return EncodeCore(text, pair, addSpecialTokens).Build();
    }

    /// <summary>Encodes several texts. With padding configured, the batch is padded together.</summary>
    public IReadOnlyList<TokenEncoding> EncodeBatch(IReadOnlyList<string> texts, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var builders = new EncodingBuilder[texts.Count];
        for (int i = 0; i < builders.Length; i++)
        {
            builders[i] = EncodeCore(texts[i] ?? throw new ArgumentException("Batch contains a null text.", nameof(texts)), null, addSpecialTokens);
        }

        return FinishBatch(builders);
    }

    /// <summary>Encodes several text pairs. With padding configured, the batch is padded together.</summary>
    public IReadOnlyList<TokenEncoding> EncodeBatch(IReadOnlyList<(string Text, string Pair)> pairs, bool addSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        var builders = new EncodingBuilder[pairs.Count];
        for (int i = 0; i < builders.Length; i++)
        {
            (string text, string pair) = pairs[i];
            if (text is null || pair is null)
            {
                throw new ArgumentException("Batch contains a null text.", nameof(pairs));
            }

            builders[i] = EncodeCore(text, pair, addSpecialTokens);
        }

        return FinishBatch(builders);
    }

    /// <summary>Decodes ids back to text with the file's decoder.</summary>
    /// <param name="ids">Token ids. Ids unknown to the vocabulary are skipped, as in the reference.</param>
    /// <param name="skipSpecialTokens">Drop special added tokens (for example <c>&lt;s&gt;</c>) before decoding.</param>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecialTokens)
    {
        var tokens = new List<string>(ids.Length);
        foreach (int id in ids)
        {
            string? token = _pipeline.AddedVocabulary.IdToToken(id) ?? _pipeline.Model.IdToToken(id);
            if (token is null || (skipSpecialTokens && _pipeline.AddedVocabulary.IsSpecialToken(token)))
            {
                continue;
            }

            tokens.Add(token);
        }

        return _pipeline.Decoder is { } decoder
            ? string.Concat(decoder.DecodeChain(tokens))
            : string.Join(' ', tokens);
    }

    /// <summary>Id of a token string, checking added tokens first, or null.</summary>
    public int? TokenToId(string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        return _pipeline.AddedVocabulary.TokenToId(token) ?? _pipeline.Model.TokenToId(token);
    }

    /// <summary>Token string of an id, checking added tokens first, or null.</summary>
    public string? IdToToken(int id) => _pipeline.AddedVocabulary.IdToToken(id) ?? _pipeline.Model.IdToToken(id);

    /// <summary>Vocabulary size, optionally counting added tokens that are not in the model vocabulary.</summary>
    public int GetVocabularySize(bool includeAddedTokens)
    {
        int size = _pipeline.Model.VocabularySize;
        if (!includeAddedTokens)
        {
            return size;
        }

        int overlapping = _pipeline.AddedVocabulary.TokenToIdMap.Keys.Count(t => _pipeline.Model.TokenToId(t) is not null);
        return size + _pipeline.AddedVocabulary.Count - overlapping;
    }

    private TokenEncoding[] FinishBatch(EncodingBuilder[] builders)
    {
        if (Padding is { } padding)
        {
            PadEncodings(builders, padding);
        }

        var result = new TokenEncoding[builders.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = builders[i].Build();
        }

        return result;
    }

    private EncodingBuilder EncodeCore(string text, string? pair, bool addSpecialTokens)
    {
        EncodingBuilder encoding = EncodeSingleSequence(text, 0, nameof(text));
        EncodingBuilder? pairEncoding = pair is null ? null : EncodeSingleSequence(pair, 1, nameof(pair));
        return PostProcess(encoding, pairEncoding, addSpecialTokens);
    }

    private EncodingBuilder EncodeSingleSequence(string sequence, int typeId, string paramName)
    {
        byte[] utf8 = Utf8Util.EncodeStrict(sequence, paramName);
        PreTokenizedString pretokenized = _pipeline.AddedVocabulary.ExtractAndNormalize(_pipeline.Normalizer, utf8);
        _pipeline.PreTokenizer?.PreTokenize(pretokenized);
        IModel model = _pipeline.Model;
        pretokenized.Tokenize(normalized => model.Tokenize(normalized.Normalized));
        return pretokenized.IntoEncoding(typeId);
    }

    private EncodingBuilder PostProcess(EncodingBuilder encoding, EncodingBuilder? pair, bool addSpecialTokens)
    {
        if (Truncation is { } truncation)
        {
            int added = _pipeline.PostProcessor?.AddedTokens(pair is not null) ?? 0;
            int maxLength = addSpecialTokens && added > 0 ? truncation.MaxLength - added : truncation.MaxLength;
            TruncateEncodings(encoding, pair, maxLength, truncation);
        }

        EncodingBuilder final;
        if (_pipeline.PostProcessor is { } processor)
        {
            final = processor.Process(encoding, pair, addSpecialTokens);
        }
        else if (pair is null)
        {
            final = encoding;
        }
        else
        {
            final = new EncodingBuilder();
            encoding.SetSequenceId(0);
            pair.SetSequenceId(1);
            final.Append(encoding);
            final.Append(pair);
        }

        if (Padding is { } padding)
        {
            PadEncodings([final], padding);
        }

        return final;
    }

    /// <summary>Upstream <c>truncate_encodings</c> (stride 0).</summary>
    private static void TruncateEncodings(EncodingBuilder encoding, EncodingBuilder? pair, int maxLength, TruncationOptions options)
    {
        if (maxLength == 0)
        {
            encoding.Truncate(0, options.Direction);
            pair?.Truncate(0, options.Direction);
            return;
        }

        int total = encoding.Count + (pair?.Count ?? 0);
        if (total <= maxLength)
        {
            return;
        }

        int toRemove = total - maxLength;
        switch (options.Strategy)
        {
            case TruncationStrategy.LongestFirst when pair is not null:
                {
                    int n1 = encoding.Count;
                    int n2 = pair.Count;
                    bool swap = false;
                    if (n1 > n2)
                    {
                        swap = true;
                        (n1, n2) = (n2, n1);
                    }

                    n2 = n1 > maxLength ? n1 : Math.Max(n1, maxLength - n1);
                    if (n1 + n2 > maxLength)
                    {
                        n1 = maxLength / 2;
                        n2 = n1 + (maxLength % 2);
                    }

                    if (swap)
                    {
                        (n1, n2) = (n2, n1);
                    }

                    encoding.Truncate(n1, options.Direction);
                    pair.Truncate(n2, options.Direction);
                    break;
                }

            case TruncationStrategy.LongestFirst:
                encoding.Truncate(total - toRemove, options.Direction);
                break;
            default:
                {
                    EncodingBuilder target = options.Strategy == TruncationStrategy.OnlyFirst
                        ? encoding
                        : pair ?? throw new InvalidOperationException("Truncation error: Second sequence not provided");
                    if (target.Count <= toRemove)
                    {
                        throw new InvalidOperationException("Truncation error: Sequence to truncate too short to respect the provided max_length");
                    }

                    target.Truncate(target.Count - toRemove, options.Direction);
                    break;
                }
        }
    }

    /// <summary>Upstream <c>pad_encodings</c>.</summary>
    private static void PadEncodings(EncodingBuilder[] encodings, PaddingOptions padding)
    {
        if (encodings.Length == 0)
        {
            return;
        }

        int padLength = padding.FixedLength ?? encodings.Max(e => e.Count);
        if (padding.PadToMultipleOf is int multiple && multiple > 0 && padLength % multiple > 0)
        {
            padLength += multiple - (padLength % multiple);
        }

        foreach (EncodingBuilder e in encodings)
        {
            e.Pad(padLength, padding.PadId, padding.PadTypeId, padding.PadToken, padding.Direction);
        }
    }
}
