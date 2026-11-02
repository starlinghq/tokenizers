// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// A hand-written matcher for the o200k Split regex, written fresh from the regex semantics (leftmost
// match, alternatives tried in order, greedy quantifiers with backtracking) of the reference's engine,
// Oniguruma as used by huggingface/tokenizers v0.23.3, like QwenSplitPattern. No regex engine is
// involved: .NET Regex matches UTF-16 code units, so \p{L} does not match supplementary-plane letters.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>
/// The o200k pre-tokenizer regex (tiktoken's <c>o200k_base</c> pattern, as microsoft/Phi-4-mini-instruct
/// declares it), matched over Unicode scalar values with the reference's character classes:
/// <code>
/// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
/// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
/// |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// Every character starts a match (a letter or mark matches alternative 1 or 2, a number 3, any other
/// non-space 4, a space 5 to 7), so there are no gaps: <c>Removed</c> with <c>invert</c> true (the form
/// Phi-4-mini declares) and <c>Isolated</c> with <c>invert</c> false give the same pieces.
/// </para>
/// <para>
/// Case-insensitive contractions follow the reference's case folding (QwenSplitPattern): <c>s</c> also
/// matches U+017F; the other letters match only their ASCII upper and lower case.
/// </para>
/// </remarks>
internal sealed class O200kSplitPattern : IPattern
{
    public const string O200k =
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        + @"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        + @"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    // Per-character flags.
    private const byte Letter = 1;         // \p{L}
    private const byte Number = 2;         // \p{N}
    private const byte Upper = 4;          // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    private const byte Lower = 8;          // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    private const byte Space = 16;         // \s
    private const byte CrLf = 32;          // \r or \n
    private const byte Blank = 64;         // U+0020
    private const byte Slash = 128;        // '/'

    private const byte NotPrefix = CrLf | Letter | Number;     // excluded by [^\r\n\p{L}\p{N}]
    private const byte NotOther = Space | Letter | Number;     // excluded by [^\s\p{L}\p{N}]

    private const int UpperClasses = RegexUnicodeClasses.UppercaseLetter | RegexUnicodeClasses.TitlecaseLetter
        | RegexUnicodeClasses.ModifierLetter | RegexUnicodeClasses.OtherLetter | RegexUnicodeClasses.Mark;

    private const int LowerClasses = RegexUnicodeClasses.LowercaseLetter | RegexUnicodeClasses.ModifierLetter
        | RegexUnicodeClasses.OtherLetter | RegexUnicodeClasses.Mark;

    private static readonly O200kSplitPattern _instance = new();

    private O200kSplitPattern()
    {
    }

    public static O200kSplitPattern? TryCreate(string regex) => regex == O200k ? _instance : null;

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
    private static List<PatternMatch> Scan(ReadOnlySpan<int> cps, ReadOnlySpan<int> starts, ReadOnlySpan<byte> flags)
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
                // Unreachable for this pattern (every character starts a match); kept so a gap is
                // reported as the reference would.
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
    private static int MatchAt(ReadOnlySpan<int> cps, ReadOnlySpan<byte> flags, int i)
    {
        int n = cps.Length;
        byte f = flags[i];
        bool prefix = (f & NotPrefix) == 0;

        // 1. [^\r\n\p{L}\p{N}]?[Upper]*[Lower]+(contraction)?   (the optional prefix is tried taken first)
        if (prefix)
        {
            int end = UpperStarLowerPlus(cps, flags, i + 1);
            if (end >= 0)
            {
                return end;
            }
        }

        int end1 = UpperStarLowerPlus(cps, flags, i);
        if (end1 >= 0)
        {
            return end1;
        }

        // 2. [^\r\n\p{L}\p{N}]?[Upper]+[Lower]*(contraction)?
        if (prefix && (flags[i + 1] & Upper) != 0)
        {
            return UpperPlusLowerStar(cps, flags, i + 1);
        }

        if ((f & Upper) != 0)
        {
            return UpperPlusLowerStar(cps, flags, i);
        }

        // 3. \p{N}{1,3}, greedy
        if ((f & Number) != 0)
        {
            int end = i + 1;
            while (end - i < 3 && (flags[end] & Number) != 0)
            {
                end++;
            }

            return end;
        }

        // 4. ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
        if ((f & Blank) != 0 && i + 1 < n && (flags[i + 1] & NotOther) == 0)
        {
            return RunEnd(flags, RunEndNot(flags, i + 1, NotOther, n), CrLf | Slash);
        }

        if ((f & NotOther) == 0)
        {
            return RunEnd(flags, RunEndNot(flags, i, NotOther, n), CrLf | Slash);
        }

        if ((f & Space) == 0)
        {
            return -1;
        }

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

    /// <summary>
    /// <c>[Upper]*[Lower]+(contraction)?</c> at <paramref name="j"/>, or -1. The greedy <c>[Upper]*</c>
    /// takes the whole Upper run; if the next character is not Lower (then it is not Ll), it gives back
    /// characters until the last one it gave back is Lower (an Lm, Lo or M, which both classes hold),
    /// and <c>[Lower]+</c> is that one character, because what follows it in the run is Lu or Lt.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int UpperStarLowerPlus(ReadOnlySpan<int> cps, ReadOnlySpan<byte> flags, int j)
    {
        int r = RunEnd(flags, j, Upper);
        int end;
        if ((flags[r] & Lower) != 0)
        {
            end = RunEnd(flags, r, Lower);
        }
        else
        {
            int p = r - 1;
            while (p >= j && (flags[p] & Lower) == 0)
            {
                p--;
            }

            if (p < j)
            {
                return -1;
            }

            end = p + 1;
        }

        return Contraction(cps, end);
    }

    /// <summary><c>[Upper]+[Lower]*(contraction)?</c> at <paramref name="j"/>, whose character is Upper.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int UpperPlusLowerStar(ReadOnlySpan<int> cps, ReadOnlySpan<byte> flags, int j) =>
        Contraction(cps, RunEnd(flags, RunEnd(flags, j, Upper), Lower));

    /// <summary>The optional, greedy <c>(?i:'s|'t|'re|'ve|'m|'ll|'d)?</c> at <paramref name="i"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Contraction(ReadOnlySpan<int> cps, int i)
    {
        int n = cps.Length;
        if (i + 1 < n && cps[i] == '\'')
        {
            switch (cps[i + 1])
            {
                case 's' or 'S' or 0x17F or 't' or 'T' or 'm' or 'M' or 'd' or 'D':
                    return i + 2;
                case 'r' or 'R' or 'v' or 'V' when i + 2 < n && cps[i + 2] is 'e' or 'E':
                    return i + 3;
                case 'l' or 'L' when i + 2 < n && cps[i + 2] is 'l' or 'L':
                    return i + 3;
            }
        }

        return i;
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
            | ((cls & UpperClasses) != 0 ? Upper : 0)
            | ((cls & LowerClasses) != 0 ? Lower : 0));
        if (c == '/')
        {
            f |= Slash;
        }
        else if (UnicodeData.IsWhiteSpace(new Rune(c)))
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
