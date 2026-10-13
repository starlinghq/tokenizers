// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// A hand-written matcher for two Split regexes, written fresh from the regex semantics (leftmost
// match, alternatives tried in order, greedy quantifiers with backtracking) of the reference's
// engine, Oniguruma as used by huggingface/tokenizers v0.23.2. No regex engine is involved: .NET
// Regex matches UTF-16 code units, so \p{L} does not match supplementary-plane letters
// (docs/design/source-audit.md section 3.4).

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace StarlingHq.Tokenizers.Pipeline;

/// <summary>
/// The Qwen pre-tokenizer regexes, matched over Unicode scalar values with the reference's
/// character classes:
/// <list type="bullet">
/// <item><see cref="Qwen2"/>: <c>(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+</c></item>
/// <item><see cref="Qwen35"/>: the same with <c>[\p{L}\p{M}]+</c> and <c>[^\s\p{L}\p{M}\p{N}]+</c>.</item>
/// </list>
/// </summary>
/// <remarks>
/// Case-insensitive contractions follow the reference's case folding, probed over every scalar
/// value: <c>s</c> also matches U+017F LATIN SMALL LETTER LONG S; the other letters match only
/// their ASCII upper and lower case.
/// </remarks>
internal sealed class QwenSplitPattern : IPattern
{
    public const string Qwen2 = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    public const string Qwen35 = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    // Per-character flags.
    private const byte Letter = 1;
    private const byte Number = 2;
    private const byte Mark = 4;
    private const byte Space = 8;          // \s
    private const byte CrLf = 16;          // \r or \n
    private const byte Blank = 32;         // U+0020

    private readonly byte _letterish;      // \p{L}, or [\p{L}\p{M}]
    private readonly byte _notOther;       // classes excluded from the "other" run of alternative 4

    private QwenSplitPattern(bool marksAreLetters)
    {
        _letterish = marksAreLetters ? (byte)(Letter | Mark) : Letter;
        _notOther = marksAreLetters ? (byte)(Space | Letter | Number | Mark) : (byte)(Space | Letter | Number);
    }

    public static QwenSplitPattern? TryCreate(string regex) => regex switch
    {
        Qwen2 => new QwenSplitPattern(marksAreLetters: false),
        Qwen35 => new QwenSplitPattern(marksAreLetters: true),
        _ => null,
    };

    /// <summary>Upstream <c>impl Pattern for &amp;SysRegex</c>: matches and the gaps between them.</summary>
    public List<PatternMatch> FindMatches(ReadOnlySpan<byte> inside)
    {
        if (inside.IsEmpty)
        {
            return [new PatternMatch(0, 0, false)];
        }

        int n = 0;
        int[] cps = ArrayPool<int>.Shared.Rent(inside.Length);
        int[] starts = ArrayPool<int>.Shared.Rent(inside.Length + 1);
        byte[] flags = ArrayPool<byte>.Shared.Rent(inside.Length + 1);
        try
        {
            int b = 0;
            while (b < inside.Length)
            {
                int c;
                int len;
                byte first = inside[b];
                if (first < 0x80)
                {
                    c = first;
                    len = 1;
                }
                else
                {
                    c = Utf8Util.DecodeAt(inside, b, out len).Value;
                }

                cps[n] = c;
                starts[n] = b;
                flags[n] = Classify(c);
                n++;
                b += len;
            }

            starts[n] = inside.Length;
            flags[n] = 0;
            return Scan(cps.AsSpan(0, n), starts.AsSpan(0, n + 1), flags.AsSpan(0, n + 1));
        }
        finally
        {
            ArrayPool<int>.Shared.Return(cps);
            ArrayPool<int>.Shared.Return(starts);
            ArrayPool<byte>.Shared.Return(flags);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private List<PatternMatch> Scan(ReadOnlySpan<int> cps, ReadOnlySpan<int> starts, ReadOnlySpan<byte> flags)
    {
        int n = cps.Length;
        var matches = new List<PatternMatch>();
        int prev = 0;
        int pos = 0;
        while (pos < n)
        {
            int end = MatchAt(cps, flags, pos);
            if (end < 0)
            {
                // Unreachable for these patterns (every character starts a match); kept so a gap
                // is reported as the reference would.
                pos++;
                continue;
            }

            if (prev != pos)
            {
                matches.Add(new PatternMatch(starts[prev], starts[pos], false));
            }

            matches.Add(new PatternMatch(starts[pos], starts[end], true));
            prev = end;
            pos = end;
        }

        if (prev != n)
        {
            matches.Add(new PatternMatch(starts[prev], starts[n], false));
        }

        return matches;
    }

    /// <summary>End (exclusive, in characters) of the match starting at <paramref name="i"/>, or -1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int MatchAt(ReadOnlySpan<int> cps, ReadOnlySpan<byte> flags, int i)
    {
        int n = cps.Length;
        int c = cps[i];
        byte f = flags[i];

        // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && i + 1 < n)
        {
            int c1 = cps[i + 1];
            switch (c1)
            {
                case 's' or 'S' or 0x17F or 't' or 'T' or 'm' or 'M' or 'd' or 'D':
                    return i + 2;
                case 'r' or 'R' or 'v' or 'V' when i + 2 < n && cps[i + 2] is 'e' or 'E':
                    return i + 3;
                case 'l' or 'L' when i + 2 < n && cps[i + 2] is 'l' or 'L':
                    return i + 3;
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+   (Qwen3.5: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+)
        if ((f & (CrLf | Letter | Number)) == 0 && (flags[i + 1] & _letterish) != 0)
        {
            return RunEnd(flags, i + 1, _letterish);
        }

        if ((f & _letterish) != 0)
        {
            return RunEnd(flags, i, _letterish);
        }

        // 3. \p{N}
        if ((f & Number) != 0)
        {
            return i + 1;
        }

        // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*'   (Qwen3.5 also excludes \p{M})
        if ((f & Blank) != 0 && i + 1 < n && (flags[i + 1] & _notOther) == 0)
        {
            return RunEnd(flags, RunEndNot(flags, i + 1, _notOther, n), CrLf);
        }

        if ((f & _notOther) == 0)
        {
            return RunEnd(flags, RunEndNot(flags, i, _notOther, n), CrLf);
        }

        // Only \s remains.
        int spaceEnd = RunEnd(flags, i, Space);

        // 5. \s*[\r\n]+: ends after the last \r or \n of the whitespace run.
        for (int k = spaceEnd - 1; k >= i; k--)
        {
            if ((flags[k] & CrLf) != 0)
            {
                return k + 1;
            }
        }

        // 6. \s+(?!\S): the whole run at the end of the text, else all but its last character.
        if (spaceEnd == n)
        {
            return spaceEnd;
        }

        if (spaceEnd - i >= 2)
        {
            return spaceEnd - 1;
        }

        // 7. \s+
        return spaceEnd;
    }

    /// <summary>First index at or after <paramref name="i"/> whose flags have none of <paramref name="mask"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RunEnd(ReadOnlySpan<byte> flags, int i, byte mask)
    {
        // flags has a terminating 0 entry, so the loop stops at the end of the text.
        while ((flags[i] & mask) != 0)
        {
            i++;
        }

        return i;
    }

    /// <summary>First index at or after <paramref name="i"/> (at most <paramref name="n"/>) whose flags have any of <paramref name="mask"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RunEndNot(ReadOnlySpan<byte> flags, int i, byte mask, int n)
    {
        while (i < n && (flags[i] & mask) == 0)
        {
            i++;
        }

        return i;
    }

    private static byte Classify(int c)
    {
        int cls = RegexUnicodeClasses.Classify(c);
        byte f = (byte)(((cls & RegexUnicodeClasses.Letter) != 0 ? Letter : 0)
            | ((cls & RegexUnicodeClasses.Number) != 0 ? Number : 0)
            | ((cls & RegexUnicodeClasses.Mark) != 0 ? Mark : 0));
        if (UnicodeData.IsWhiteSpace(new Rune(c)))
        {
            f |= Space;
            if (c is '\r' or '\n')
            {
                f |= CrLf;
            }
            else if (c == ' ')
            {
                f |= Blank;
            }
        }

        return f;
    }
}

/// <summary>
/// <c>\p{L}</c>, <c>\p{N}</c> and <c>\p{M}</c> as the reference's regex engine matches them
/// (generated tables, <see cref="RegexUnicodeClassData"/>), independent of the .NET runtime's
/// Unicode version.
/// </summary>
internal static class RegexUnicodeClasses
{
    public const int Letter = 1;
    public const int Number = 2;
    public const int Mark = 4;

    private static readonly byte[] _bmp = BuildBmp();

    public static int Classify(int c)
    {
        if (c < 0x10000)
        {
            return _bmp[c];
        }

        int cls = 0;
        if (InRanges(RegexUnicodeClassData.LetterRanges, c))
        {
            cls |= Letter;
        }

        if (InRanges(RegexUnicodeClassData.NumberRanges, c))
        {
            cls |= Number;
        }

        if (InRanges(RegexUnicodeClassData.MarkRanges, c))
        {
            cls |= Mark;
        }

        return cls;
    }

    private static bool InRanges(ReadOnlySpan<int> ranges, int c)
    {
        int lo = 0;
        int hi = (ranges.Length / 2) - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (c < ranges[2 * mid])
            {
                hi = mid - 1;
            }
            else if (c > ranges[(2 * mid) + 1])
            {
                lo = mid + 1;
            }
            else
            {
                return true;
            }
        }

        return false;
    }

    private static byte[] BuildBmp()
    {
        var bmp = new byte[0x10000];
        Fill(bmp, RegexUnicodeClassData.LetterRanges, Letter);
        Fill(bmp, RegexUnicodeClassData.NumberRanges, Number);
        Fill(bmp, RegexUnicodeClassData.MarkRanges, Mark);
        return bmp;
    }

    private static void Fill(byte[] bmp, ReadOnlySpan<int> ranges, int bit)
    {
        for (int i = 0; i < ranges.Length; i += 2)
        {
            int end = Math.Min(ranges[i + 1], 0xFFFF);
            for (int c = ranges[i]; c <= end; c++)
            {
                bmp[c] |= (byte)bit;
            }
        }
    }
}
