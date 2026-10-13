// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3),
// tokenizers/src/tokenizer/pre_tokenizer.rs, Copyright 2020 The HuggingFace Inc. team,
// Apache License 2.0 (see THIRD-PARTY-NOTICES.md). Modified by StarlingHq: translated to
// C#; offsets are converted to Unicode scalar values as the reference Python binding does.

namespace StarlingHq.Tokenizers.Pipeline;

/// <summary>A model token with byte offsets into the normalized split it came from.</summary>
internal readonly record struct ModelToken(int Id, string Value, int Start, int End);

/// <summary>A piece of the input: its normalized string and, once tokenized, its tokens.</summary>
internal sealed class Split(NormalizedString normalized, List<ModelToken>? tokens = null)
{
    public NormalizedString Normalized { get; set; } = normalized;

    public List<ModelToken>? Tokens { get; set; } = tokens;
}

/// <summary>Upstream <c>PreTokenizedString</c>.</summary>
internal sealed class PreTokenizedString
{
    private readonly byte[] _original;
    private List<Split> _splits;

    public PreTokenizedString(byte[] originalUtf8)
    {
        _original = originalUtf8;
        _splits = [new Split(NormalizedString.FromUtf8(originalUtf8))];
    }

    public IReadOnlyList<Split> Splits => _splits;

    /// <summary>Splits every untokenized piece; empty results are dropped.</summary>
    public void SplitEach(Func<NormalizedString, IEnumerable<Split>> splitFn)
    {
        var next = new List<Split>(_splits.Count);
        foreach (Split split in _splits)
        {
            if (split.Tokens is not null)
            {
                next.Add(split);
                continue;
            }

            foreach (Split produced in splitFn(split.Normalized))
            {
                if (!produced.Normalized.IsEmpty)
                {
                    next.Add(produced);
                }
            }
        }

        _splits = next;
    }

    /// <summary>Upstream <c>normalize</c>: transforms every untokenized piece in place.</summary>
    public void NormalizeEach(Action<NormalizedString> normalize)
    {
        foreach (Split split in _splits)
        {
            if (split.Tokens is null)
            {
                normalize(split.Normalized);
            }
        }
    }

    public void Tokenize(Func<NormalizedString, List<ModelToken>> tokenize)
    {
        foreach (Split split in _splits)
        {
            split.Tokens ??= tokenize(split.Normalized);
        }
    }

    /// <summary>Upstream <c>into_encoding</c> with <c>OffsetType::Char</c>.</summary>
    public EncodingBuilder IntoEncoding(int typeId)
    {
        var builder = new EncodingBuilder();
        if (_splits.Count == 0)
        {
            return builder;
        }

        var converter = new OffsetConverter(_original);
        foreach (Split split in _splits)
        {
            NormalizedString normalized = split.Normalized;
            (int baseStart, _) = normalized.OffsetsOriginal();
            foreach (ModelToken token in split.Tokens!)
            {
                (int Start, int End) bytes = normalized.ConvertNormalizedToOriginal(token.Start, token.End) is { } r
                    ? (baseStart + r.Start, baseStart + r.End)
                    : (token.Start, token.End);
                builder.Add(token.Id, token.Value, typeId, converter.ToScalar(bytes), converter);
            }
        }

        return builder;
    }
}

/// <summary>
/// Converts byte offsets of the original input to Unicode scalar offsets exactly as upstream
/// <c>BytesToCharOffsetConverter</c> does, then to UTF-16 code units for C# string slicing.
/// </summary>
internal sealed class OffsetConverter
{
    private readonly int[] _byteToScalar;
    private readonly int[] _scalarToUtf16;
    private readonly int _byteLength;

    public OffsetConverter(byte[] utf8)
    {
        _byteLength = utf8.Length;
        _byteToScalar = new int[utf8.Length];
        var scalarToUtf16 = new List<int>(utf8.Length + 1) { 0 };
        int scalar = 0;
        int utf16 = 0;
        int i = 0;
        while (i < utf8.Length)
        {
            int len = Utf8Util.SequenceLength(utf8, i);
            for (int k = 0; k < len; k++)
            {
                _byteToScalar[i + k] = scalar;
            }

            scalar++;
            utf16 += len == 4 ? 2 : 1;
            scalarToUtf16.Add(utf16);
            i += len;
        }

        _scalarToUtf16 = [.. scalarToUtf16];
    }

    /// <summary>Upstream <c>BytesToCharOffsetConverter::convert</c>, falling back to the input
    /// offsets when the start is not a byte of the original, as upstream does.</summary>
    public (int Start, int End) ToScalar((int Start, int End) bytes)
    {
        bool hasStart = (uint)bytes.Start < (uint)_byteLength;
        if (!hasStart)
        {
            return bytes;
        }

        int start = _byteToScalar[bytes.Start];
        if ((uint)bytes.End < (uint)_byteLength)
        {
            return (start, _byteToScalar[bytes.End]);
        }

        int last = (uint)(bytes.End - 1) < (uint)_byteLength ? _byteToScalar[bytes.End - 1] : start + 1;
        return (start, last + 1);
    }

    public TokenOffset ScalarToUtf16((int Start, int End) scalar) =>
        new(ToUtf16(scalar.Start), ToUtf16(scalar.End));

    private int ToUtf16(int scalar) =>
        _scalarToUtf16[Math.Clamp(scalar, 0, _scalarToUtf16.Length - 1)];
}
