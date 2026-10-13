// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3):
// tokenizers/src/processors/template.rs, tokenizers/src/tokenizer/mod.rs (PostProcessor::process),
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

/// <summary>The <c>post_processor</c> stage.</summary>
internal interface IPostProcessor
{
    int AddedSingle { get; }

    int AddedPair { get; }

    /// <summary>Upstream <c>added_tokens(is_pair)</c>.</summary>
    int AddedTokens(bool isPair);

    /// <summary>Upstream <c>PostProcessor::process</c>: the final encoding.</summary>
    EncodingBuilder Process(EncodingBuilder encoding, EncodingBuilder? pair, bool addSpecialTokens);
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

    /// <summary>Upstream <c>PostProcessor::process</c> followed by <c>apply_template</c> and merge.</summary>
    public EncodingBuilder Process(EncodingBuilder encoding, EncodingBuilder? pair, bool addSpecialTokens)
    {
        EncodingBuilder[] encodings = pair is null ? [encoding] : [encoding, pair];
        for (int i = 0; i < encodings.Length; i++)
        {
            encodings[i].SetSequenceId(i);
            encodings[i].SetTypeIds(i);
        }

        IReadOnlyList<TemplatePiece> template = pair is null ? _single : _pair;
        var result = new EncodingBuilder();
        foreach (TemplatePiece piece in template)
        {
            switch (piece)
            {
                case SequencePiece seq:
                    EncodingBuilder source = encodings[seq.Sequence];
                    source.SetTypeIds(seq.TypeId);
                    source.SetSequenceId(seq.Sequence);
                    result.Append(source);
                    break;
                case SpecialTokenPiece special when addSpecialTokens:
                    TemplateSpecialToken tok = _specialTokens[special.Id];
                    var added = new EncodingBuilder();
                    for (int k = 0; k < tok.Ids.Count; k++)
                    {
                        added.AddSpecial(tok.Ids[k], tok.Tokens[k], special.TypeId);
                    }

                    result.Append(added);
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
/// and trailing spaces from token offsets (upstream <c>process_offsets</c>), then sets sequence ids
/// and merges the encodings.
/// </summary>
internal sealed class ByteLevelPostProcessor(bool addPrefixSpace, bool trimOffsets) : IPostProcessor
{
    public int AddedSingle => 0;

    public int AddedPair => 0;

    public int AddedTokens(bool isPair) => 0;

    public EncodingBuilder Process(EncodingBuilder encoding, EncodingBuilder? pair, bool addSpecialTokens)
    {
        if (trimOffsets)
        {
            encoding.TrimByteLevelOffsets(addPrefixSpace);
            pair?.TrimByteLevelOffsets(addPrefixSpace);
        }

        encoding.SetSequenceId(0);
        if (pair is null)
        {
            return encoding;
        }

        pair.SetSequenceId(1);
        var merged = new EncodingBuilder();
        merged.Append(encoding);
        merged.Append(pair);
        return merged;
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
