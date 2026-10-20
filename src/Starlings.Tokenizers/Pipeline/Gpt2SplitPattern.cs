// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// A hand-written matcher for the GPT-2 byte-level split regex, written fresh from the regex
// semantics (leftmost match, alternatives tried in order, greedy quantifiers with backtracking)
// of the reference's engine, Oniguruma as used by huggingface/tokenizers v0.23.2
// (pre_tokenizers/byte_level.rs). No regex engine is involved: .NET Regex matches UTF-16 code
// units, so \p{L} does not match supplementary-plane letters (docs/design/source-audit.md §3.4).

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>
/// The GPT-2 byte-level split regex (upstream <c>byte_level.rs</c> <c>RE</c>):
/// <c>'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+</c>,
/// matched over Unicode scalar values with the reference's character classes.
/// </summary>
internal sealed class Gpt2SplitPattern : IPattern
{
    public const string Gpt2 = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    // Per-character flags.
    private const byte Letter = 1;
    private const byte Number = 2;
    private const byte Space = 4;          // \s
    private const byte Blank = 8;          // U+0020, the literal space of the ` ?` prefixes
    private const byte OtherMask = Space | Letter | Number; // excluded by [^\s\p{L}\p{N}]

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
                // Unreachable for this pattern (every character starts a match).
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
    private static int MatchAt(ReadOnlySpan<int> cps, ReadOnlySpan<byte> flags, int i)
    {
        int n = cps.Length;
        int c = cps[i];
        byte f = flags[i];

        // 1. 's | 't | 're | 've | 'm | 'll | 'd   (case-sensitive)
        if (c == '\'' && i + 1 < n)
        {
            int c1 = cps[i + 1];
            switch (c1)
            {
                case 's' or 't' or 'm' or 'd':
                    return i + 2;
                case 'r' or 'v' when i + 2 < n && cps[i + 2] == 'e':
                    return i + 3;
                case 'l' when i + 2 < n && cps[i + 2] == 'l':
                    return i + 3;
            }
        }

        // 2. ' ?\p{L}+'
        if ((f & Letter) != 0)
        {
            return RunEnd(flags, i, Letter);
        }

        if ((f & Blank) != 0 && i + 1 < n && (flags[i + 1] & Letter) != 0)
        {
            return RunEnd(flags, i + 1, Letter);
        }

        // 3. ' ?\p{N}+'
        if ((f & Number) != 0)
        {
            return RunEnd(flags, i, Number);
        }

        if ((f & Blank) != 0 && i + 1 < n && (flags[i + 1] & Number) != 0)
        {
            return RunEnd(flags, i + 1, Number);
        }

        // 4. ' ?[^\s\p{L}\p{N}]+'
        if ((f & OtherMask) == 0)
        {
            return RunEndNot(flags, i, OtherMask, n);
        }

        if ((f & Blank) != 0 && i + 1 < n && (flags[i + 1] & OtherMask) == 0)
        {
            return RunEndNot(flags, i + 1, OtherMask, n);
        }

        // Only \s remains.
        int spaceEnd = RunEnd(flags, i, Space);

        // 5. '\s+(?!\S)': the whole run at the end of the text, else all but its last character.
        if (spaceEnd == n)
        {
            return spaceEnd;
        }

        if (spaceEnd - i >= 2)
        {
            return spaceEnd - 1;
        }

        // 6. '\s+'
        return spaceEnd;
    }

    /// <summary>First index at or after <paramref name="i"/> whose flags have none of <paramref name="mask"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int RunEnd(ReadOnlySpan<byte> flags, int i, byte mask)
    {
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
            | ((cls & RegexUnicodeClasses.Number) != 0 ? Number : 0));
        if (UnicodeData.IsWhiteSpace(new Rune(c)))
        {
            f |= Space;
            if (c == ' ')
            {
                f |= Blank;
            }
        }

        return f;
    }
}
