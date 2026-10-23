// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3):
// tokenizers/src/processors/template.rs, tokenizers/src/processors/sequence.rs,
// tokenizers/src/tokenizer/mod.rs (PostProcessor::process),
// tokenizers/src/pre_tokenizers/metaspace.rs (Decoder impl), tokenizers/src/pre_tokenizers/byte_level.rs
// (PostProcessor and Decoder impls, process_offsets), Copyright The HuggingFace Inc. team,
// Apache License 2.0 (see THIRD-PARTY-NOTICES.md). Modified by StarlingHq: translated to C#.

using System.Runtime.InteropServices;
using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>A piece of a <c>TemplateProcessing</c> template.</summary>
internal abstract record TemplatePiece(int TypeId);

/// <summary><c>{"Sequence": {"id": "A"|"B", "type_id": n}}</c>.</summary>
internal sealed record SequencePiece(int Sequence, int TypeId) : TemplatePiece(TypeId);

/// <summary><c>{"SpecialToken": {"id": "...", "type_id": n}}</c>.</summary>
internal sealed record SpecialTokenPiece(string Id, int TypeId) : TemplatePiece(TypeId);

/// <summary>An entry of <c>special_tokens</c>: one template id may expand to several tokens.</summary>
internal sealed record TemplateSpecialToken(string Id, IReadOnlyList<int> Ids, IReadOnlyList<string> Tokens);

/// <summary>The <c>post_processor</c> stage (upstream <c>PostProcessor</c> trait).</summary>
internal interface IPostProcessor
{
    int AddedSingle { get; }

    int AddedPair { get; }

    /// <summary>Upstream <c>added_tokens(is_pair)</c>.</summary>
    int AddedTokens(bool isPair);

    /// <summary>
    /// Upstream <c>process_encodings</c>: transforms a list of encodings (one per input sequence on the
    /// first call; possibly one per template piece after a <c>TemplateProcessing</c>) into another list,
    /// which <see cref="PostProcessing.Process"/> merges.
    /// </summary>
    List<EncodingBuilder> ProcessEncodings(List<EncodingBuilder> encodings, bool addSpecialTokens);
}

/// <summary>The provided method of upstream's <c>PostProcessor</c> trait.</summary>
internal static class PostProcessing
{
    /// <summary>
    /// Upstream <c>PostProcessor::process</c>: sets sequence id and type id <c>i</c> on input
    /// <c>i</c>, runs <c>process_encodings</c>, then merges the result in order
    /// (<c>Encoding::merge(encodings, false)</c>).
    /// </summary>
    public static EncodingBuilder Process(this IPostProcessor processor, EncodingBuilder encoding, EncodingBuilder? pair, bool addSpecialTokens)
    {
        List<EncodingBuilder> encodings = pair is null ? [encoding] : [encoding, pair];
        for (int i = 0; i < encodings.Count; i++)
        {
            encodings[i].SetSequenceId(i);
            encodings[i].SetTypeIds(i);
        }

        List<EncodingBuilder> processed = processor.ProcessEncodings(encodings, addSpecialTokens);
        if (processed.Count == 1)
        {
            return processed[0];
        }

        var merged = new EncodingBuilder();
        foreach (EncodingBuilder e in processed)
        {
            merged.Append(e);
        }

        return merged;
    }
}

/// <summary><c>TemplateProcessing</c> post-processor.</summary>
internal sealed class TemplateProcessing : IPostProcessor
{
    private readonly IReadOnlyList<TemplatePiece> _single;
    private readonly IReadOnlyList<TemplatePiece> _pair;
    private readonly IReadOnlyDictionary<string, TemplateSpecialToken> _specialTokens;

    public TemplateProcessing(
        IReadOnlyList<TemplatePiece> single,
        IReadOnlyList<TemplatePiece> pair,
        IReadOnlyDictionary<string, TemplateSpecialToken> specialTokens)
    {
        _single = single;
        _pair = pair;
        _specialTokens = specialTokens;
        AddedSingle = CountAdded(single);
        AddedPair = CountAdded(pair);
    }

    public int AddedSingle { get; }

    public int AddedPair { get; }

    public int AddedTokens(bool isPair) => isPair ? AddedPair : AddedSingle;

    /// <summary>
    /// Upstream <c>process_encodings</c> and <c>apply_template</c>: one encoding per template piece, in
    /// template order (special-token pieces only with <paramref name="addSpecialTokens"/>), not merged.
    /// </summary>
    public List<EncodingBuilder> ProcessEncodings(List<EncodingBuilder> encodings, bool addSpecialTokens)
    {
        IReadOnlyList<TemplatePiece> template = encodings.Count switch
        {
            1 => _single,
            2 => _pair,
            _ => throw new InvalidOperationException("TemplateProcessing takes one or two encodings."),
        };

        var result = new List<EncodingBuilder>(template.Count);
        foreach (TemplatePiece piece in template)
        {
            switch (piece)
            {
                case SequencePiece seq:
                    // The loader admits each of $A and $B exactly once, so the encoding is used, not cloned.
                    EncodingBuilder source = encodings[seq.Sequence];
                    source.SetTypeIds(seq.TypeId);
                    source.SetSequenceId(seq.Sequence);
                    result.Add(source);
                    break;
                case SpecialTokenPiece special when addSpecialTokens:
                    TemplateSpecialToken tok = _specialTokens[special.Id];
                    var added = new EncodingBuilder();
                    for (int k = 0; k < tok.Ids.Count; k++)
                    {
                        added.AddSpecial(tok.Ids[k], tok.Tokens[k], special.TypeId);
                    }

                    result.Add(added);
                    break;
            }
        }

        return result;
    }

    private int CountAdded(IReadOnlyList<TemplatePiece> template) =>
        template.OfType<SpecialTokenPiece>().Sum(p => _specialTokens[p.Id].Ids.Count);
}

internal interface IDecoder
{
    List<string> DecodeChain(List<string> tokens);
}

/// <summary><c>Metaspace</c> decoder.</summary>
internal sealed class MetaspaceDecoder(Rune replacement, PrependScheme prependScheme) : IDecoder
{
    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        var builder = new StringBuilder();
        for (int i = 0; i < tokens.Count; i++)
        {
            builder.Clear();
            foreach (Rune c in tokens[i].EnumerateRunes())
            {
                if (c == replacement)
                {
                    if (!(i == 0 && prependScheme != PrependScheme.Never))
                    {
                        builder.Append(' ');
                    }
                }
                else
                {
                    builder.Append(c.ToString());
                }
            }

            result.Add(builder.ToString());
        }

        return result;
    }
}

/// <summary>
/// <c>ByteLevel</c> post-processor: adds no tokens; with <c>trim_offsets</c> it removes leading
/// and trailing spaces from token offsets of each encoding (upstream <c>process_offsets</c>), then
/// sets sequence id <c>i</c> on encoding <c>i</c>. Merging is left to <see cref="PostProcessing.Process"/>.
/// </summary>
internal sealed class ByteLevelPostProcessor(bool addPrefixSpace, bool trimOffsets) : IPostProcessor
{
    public int AddedSingle => 0;

    public int AddedPair => 0;

    public int AddedTokens(bool isPair) => 0;

    public List<EncodingBuilder> ProcessEncodings(List<EncodingBuilder> encodings, bool addSpecialTokens)
    {
        if (trimOffsets)
        {
            foreach (EncodingBuilder e in encodings)
            {
                e.TrimByteLevelOffsets(addPrefixSpace);
            }
        }

        for (int i = 0; i < encodings.Count; i++)
        {
            encodings[i].SetSequenceId(i);
        }

        return encodings;
    }
}

/// <summary>
/// <c>Sequence</c> post-processor: each child's <c>process_encodings</c> runs on the previous child's
/// output, in order; <c>added_tokens(is_pair)</c> is the sum over the children (upstream
/// <c>processors/sequence.rs</c>).
/// </summary>
internal sealed class SequencePostProcessor : IPostProcessor
{
    public SequencePostProcessor(IReadOnlyList<IPostProcessor> processors)
    {
        Processors = processors;
        AddedSingle = processors.Sum(p => p.AddedTokens(isPair: false));
        AddedPair = processors.Sum(p => p.AddedTokens(isPair: true));
    }

    public IReadOnlyList<IPostProcessor> Processors { get; }

    public int AddedSingle { get; }

    public int AddedPair { get; }

    public int AddedTokens(bool isPair) => isPair ? AddedPair : AddedSingle;

    public List<EncodingBuilder> ProcessEncodings(List<EncodingBuilder> encodings, bool addSpecialTokens)
    {
        foreach (IPostProcessor processor in Processors)
        {
            encodings = processor.ProcessEncodings(encodings, addSpecialTokens);
        }

        return encodings;
    }
}

/// <summary>
/// <c>ByteLevel</c> decoder: maps every character of every token back to its byte (a token with a
/// character outside the byte-level alphabet contributes its own UTF-8 bytes), then decodes all
/// bytes at once, replacing ill-formed sequences with U+FFFD as Rust <c>from_utf8_lossy</c> does.
/// </summary>
internal sealed class ByteLevelDecoder : IDecoder
{
    public List<string> DecodeChain(List<string> tokens)
    {
        var bytes = new List<byte>();
        short[] charToByte = ByteLevelAlphabet.CharToByte;
        foreach (string token in tokens)
        {
            int mark = bytes.Count;
            bool mapped = true;
            foreach (char c in token)
            {
                int b = c < charToByte.Length ? charToByte[c] : -1;
                if (b < 0)
                {
                    mapped = false;
                    break;
                }

                bytes.Add((byte)b);
            }

            if (!mapped)
            {
                bytes.RemoveRange(mark, bytes.Count - mark);
                bytes.AddRange(Encoding.UTF8.GetBytes(token));
            }
        }

        // UTF8Encoding's replacement fallback substitutes one U+FFFD per maximal ill-formed
        // subsequence, the same policy as Rust's String::from_utf8_lossy.
        return [Encoding.UTF8.GetString(CollectionsMarshal.AsSpan(bytes))];
    }
}
