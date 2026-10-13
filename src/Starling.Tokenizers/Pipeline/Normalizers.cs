// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3):
// tokenizers/src/normalizers/{mod,strip,replace,precompiled}.rs, and from
// huggingface/spm_precompiled 0.1.4 src/lib.rs (the crate tokenizers 0.23.2 locks), both
// Copyright The HuggingFace Inc. team, Apache License 2.0. The precompiled charsmap layout
// (u32 trie size, darts-clone trie, NUL-separated replacement strings) follows
// dotnet/machinelearning 5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d
// src/Microsoft.ML.Tokenizers/Normalizer/SentencePieceNormalizer.cs (MIT), which derives from
// google/sentencepiece (Apache License 2.0). See THIRD-PARTY-NOTICES.md.
// Modified by StarlingHq: translated to C# as separate pipeline stages.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using Starling.Tokenizers.ThirdParty.MLNet;

namespace Starling.Tokenizers.Pipeline;

internal interface INormalizer
{
    void Normalize(NormalizedString normalized);
}

/// <summary><c>Sequence</c>: applies each normalizer in order.</summary>
internal sealed class SequenceNormalizer(IReadOnlyList<INormalizer> normalizers) : INormalizer
{
    public void Normalize(NormalizedString normalized)
    {
        foreach (INormalizer n in normalizers)
        {
            n.Normalize(normalized);
        }
    }
}

/// <summary><c>Strip</c>: removes leading and/or trailing Unicode whitespace.</summary>
internal sealed class StripNormalizer(bool left, bool right) : INormalizer
{
    public void Normalize(NormalizedString normalized) => normalized.Strip(left, right);
}

/// <summary>A pattern that yields contiguous, ordered match/non-match spans over UTF-8 bytes.</summary>
internal interface IPattern
{
    List<PatternMatch> FindMatches(ReadOnlySpan<byte> inside);
}

/// <summary>Upstream <c>impl Pattern for char</c> (and <c>Fn(char) -> bool</c>).</summary>
internal sealed class CharPattern(Rune target) : IPattern
{
    public List<PatternMatch> FindMatches(ReadOnlySpan<byte> inside)
    {
        if (inside.IsEmpty)
        {
            return [new PatternMatch(0, 0, false)];
        }

        var matches = new List<PatternMatch>();
        int lastOffset = 0;
        int lastSeen = 0;
        int b = 0;
        while (b < inside.Length)
        {
            Rune c = Utf8Util.DecodeAt(inside, b, out int len);
            lastSeen = b + len;
            if (c == target)
            {
                if (lastOffset < b)
                {
                    matches.Add(new PatternMatch(lastOffset, b, false));
                }

                matches.Add(new PatternMatch(b, b + len, true));
                lastOffset = b + len;
            }

            b += len;
        }

        if (lastSeen > lastOffset)
        {
            matches.Add(new PatternMatch(lastOffset, lastSeen, false));
        }

        return matches;
    }
}

/// <summary>
/// Leftmost, non-overlapping matches of a fixed byte sequence, or of runs of at least
/// <c>minRun</c> repetitions of one byte. These are the only <c>Replace</c> patterns accepted:
/// a literal <c>String</c> pattern, and the <c>Regex</c> <c>" {2,}"</c> used by SentencePiece
/// conversions. Both have one unambiguous meaning, so no regex engine is involved.
/// </summary>
internal sealed class SimplePattern : IPattern
{
    private readonly byte[] _literal;
    private readonly int _minRun;

    private SimplePattern(byte[] literal, int minRun)
    {
        _literal = literal;
        _minRun = minRun;
    }

    public static SimplePattern Literal(string literal) => new(Encoding.UTF8.GetBytes(literal), 0);

    public static SimplePattern ByteRun(byte b, int minRun) => new([b], minRun);

    public List<PatternMatch> FindMatches(ReadOnlySpan<byte> inside)
    {
        if (inside.IsEmpty)
        {
            return [new PatternMatch(0, 0, false)];
        }

        var matches = new List<PatternMatch>();
        int prev = 0;
        int i = 0;
        while (i < inside.Length)
        {
            int matchLength = MatchAt(inside, i);
            if (matchLength == 0)
            {
                i++;
                continue;
            }

            if (prev != i)
            {
                matches.Add(new PatternMatch(prev, i, false));
            }

            matches.Add(new PatternMatch(i, i + matchLength, true));
            i += matchLength;
            prev = i;
        }

        if (prev != inside.Length)
        {
            matches.Add(new PatternMatch(prev, inside.Length, false));
        }

        return matches;
    }

    private int MatchAt(ReadOnlySpan<byte> inside, int i)
    {
        if (_minRun == 0)
        {
            return inside[i..].StartsWith(_literal) ? _literal.Length : 0;
        }

        int run = 0;
        while (i + run < inside.Length && inside[i + run] == _literal[0])
        {
            run++;
        }

        return run >= _minRun ? run : 0;
    }
}

/// <summary>Upstream <c>Invert</c>: the wrapped pattern's spans with the match flags flipped.</summary>
internal sealed class InvertedPattern(IPattern inner) : IPattern
{
    public List<PatternMatch> FindMatches(ReadOnlySpan<byte> inside)
    {
        List<PatternMatch> matches = inner.FindMatches(inside);
        for (int i = 0; i < matches.Count; i++)
        {
            PatternMatch m = matches[i];
            matches[i] = m with { IsMatch = !m.IsMatch };
        }

        return matches;
    }
}

/// <summary><c>Replace</c>: replaces every pattern match with fixed content.</summary>
internal sealed class ReplaceNormalizer(IPattern pattern, string content) : INormalizer
{
    public void Normalize(NormalizedString normalized) =>
        normalized.Replace(pattern.FindMatches(normalized.Normalized), content);
}

/// <summary><c>Precompiled</c>: SentencePiece's precompiled character map, applied per grapheme
/// exactly as <c>tokenizers</c> does (shortest trie match, graphemes under 6 bytes first).</summary>
internal sealed class PrecompiledNormalizer : INormalizer
{
    private readonly DoubleArrayTrie _trie;
    private readonly byte[] _normalized;

    public PrecompiledNormalizer(byte[] charsmap)
    {
        if (charsmap.Length < sizeof(uint))
        {
            throw new TokenizerFormatException("normalizer Precompiled: precompiled_charsmap is too short.");
        }

        uint trieSize = BinaryPrimitives.ReadUInt32LittleEndian(charsmap);
        if (trieSize % 4 != 0 || trieSize > (uint)(charsmap.Length - sizeof(uint)))
        {
            throw new TokenizerFormatException("normalizer Precompiled: trie size exceeds the precompiled_charsmap.");
        }

        var units = new DoubleArrayUnit[trieSize / 4];
        for (int i = 0; i < units.Length; i++)
        {
            units[i] = new DoubleArrayUnit(BinaryPrimitives.ReadUInt32LittleEndian(charsmap.AsSpan(sizeof(uint) + (i * 4))));
        }

        _normalized = charsmap.AsSpan(sizeof(uint) + (int)trieSize).ToArray();
        try
        {
            _ = new UTF8Encoding(false, true).GetCharCount(_normalized);
        }
        catch (DecoderFallbackException e)
        {
            throw new TokenizerFormatException("normalizer Precompiled: replacement strings are not valid UTF-8.", e);
        }

        if (units.Length == 0)
        {
            throw new TokenizerFormatException("normalizer Precompiled: empty trie.");
        }

        _trie = new DoubleArrayTrie(units);
    }

    public void Normalize(NormalizedString normalized)
    {
        string text = normalized.NormalizedText;
        var transformations = new List<CharChange>(text.Length);
        bool modified = false;
        ReadOnlySpan<char> rest = text;
        Span<byte> graphemeBytes = stackalloc byte[8];
        Span<char> chars = stackalloc char[2];

        while (!rest.IsEmpty)
        {
            int length = StringInfo.GetNextTextElementLength(rest);
            ReadOnlySpan<char> grapheme = rest[..length];
            rest = rest[length..];

            int byteCount = Encoding.UTF8.GetByteCount(grapheme);
            if (byteCount < 6)
            {
                Encoding.UTF8.GetBytes(grapheme, graphemeBytes);
                if (Transform(graphemeBytes[..byteCount]) is { } norm)
                {
                    modified = true;
                    Replace(transformations, grapheme, norm);
                    continue;
                }
            }

            foreach (Rune c in grapheme.EnumerateRunes())
            {
                int len = c.EncodeToUtf8(graphemeBytes);
                if (Transform(graphemeBytes[..len]) is { } norm)
                {
                    modified = true;
                    int charLen = c.EncodeToUtf16(chars);
                    Replace(transformations, chars[..charLen], norm);
                }
                else
                {
                    transformations.Add(new CharChange(c, 0));
                }
            }
        }

        if (modified)
        {
            normalized.Transform(transformations, 0);
        }
    }

    /// <summary>spm_precompiled <c>transform</c>: the shortest key that prefixes the chunk.</summary>
    private string? Transform(ReadOnlySpan<byte> chunk)
    {
        int nul = chunk.IndexOf((byte)0);
        if (nul >= 0)
        {
            chunk = chunk[..nul];
        }

        Span<DoubleArrayResultPair> results = stackalloc DoubleArrayResultPair[1];
        if (_trie.CommonPrefixSearch(chunk, results) == 0)
        {
            return null;
        }

        int index = results[0].Value;
        int end = index;
        while (end < _normalized.Length && _normalized[end] != 0)
        {
            end++;
        }

        return Encoding.UTF8.GetString(_normalized, index, end - index);
    }

    /// <summary>tokenizers <c>precompiled.rs</c> <c>replace</c>.</summary>
    private static void Replace(List<CharChange> transformations, ReadOnlySpan<char> oldPart, string newPart)
    {
        int oldCount = CountRunes(oldPart);
        int newCount = CountRunes(newPart);
        int diff = newCount - oldCount;

        foreach (Rune c in newPart.EnumerateRunes())
        {
            transformations.Add(new CharChange(c, 0));
        }

        if (diff > 0)
        {
            for (int k = 0; k < diff && k < transformations.Count; k++)
            {
                int i = transformations.Count - 1 - k;
                transformations[i] = transformations[i] with { Change = 1 };
            }
        }
        else if (diff < 0 && transformations.Count > 0)
        {
            CharChange last = transformations[^1];
            transformations[^1] = last with { Change = last.Change + diff };
        }
    }

    private static int CountRunes(ReadOnlySpan<char> s)
    {
        int count = 0;
        foreach (Rune _ in s.EnumerateRunes())
        {
            count++;
        }

        return count;
    }
}
