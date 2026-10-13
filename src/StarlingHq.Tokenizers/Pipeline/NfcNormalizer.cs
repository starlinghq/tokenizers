// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from the crate unicode-normalization-alignments 0.1.12 (src/decompose.rs, recompose.rs,
// normalize.rs; MIT OR Apache-2.0, used under MIT; Copyright 2012-2015 The Rust Project
// Developers and the crate's contributors), which huggingface/tokenizers v0.23.2
// (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3) uses for NormalizedString::nfc
// (tokenizers/src/tokenizer/normalizer.rs, Apache License 2.0). See THIRD-PARTY-NOTICES.md.
// Modified by StarlingHq: translated to C# as a batch computation over a character list
// (the crate is an iterator; the emitted sequence is the same); Unicode 9.0.0 tables from the
// crate in Unicode9NormalizationData.g.cs.

using System.Text;

namespace StarlingHq.Tokenizers.Pipeline;

/// <summary><c>NFC</c> normalizer: upstream <c>normalized.nfc()</c>, Unicode 9.0.0 data, with the
/// crate's alignment changes (decomposed trailing characters are insertions, compositions merge).</summary>
internal sealed class NfcNormalizer : INormalizer
{
    public void Normalize(NormalizedString normalized)
    {
        byte[] s = normalized.Normalized;
        if (Unicode9Normalization.IsNfcInert(s))
        {
            // Every character maps to itself with change 0, which leaves the string and its
            // alignments unchanged (upstream still runs the transform; the result is identical).
            return;
        }

        List<Rune> chars = Utf8Util.DecodeRunes(s);
        normalized.Transform(Unicode9Normalization.Nfc(chars), 0);
    }
}

/// <summary>Canonical decomposition, ordering and composition with Unicode 9.0.0 data, as the crate does.</summary>
internal static class Unicode9Normalization
{
    private const int SBase = 0xAC00;
    private const int LBase = 0x1100;
    private const int VBase = 0x1161;
    private const int TBase = 0x11A7;
    private const int LCount = 19;
    private const int VCount = 21;
    private const int TCount = 28;
    private const int NCount = VCount * TCount;
    private const int SCount = LCount * NCount;

    private static readonly Lazy<Tables> _tables = new(BuildTables);

    /// <summary>True if NFC maps every character of the UTF-8 text to itself with change 0.</summary>
    /// <remarks>A character is inert if its combining class is 0, it has no canonical
    /// decomposition (Hangul syllables included), and it is never the second character of a
    /// composition. Composition needs a non-inert second character, so a text of inert
    /// characters composes nothing.</remarks>
    public static bool IsNfcInert(ReadOnlySpan<byte> utf8)
    {
        int i = 0;
        while (i < utf8.Length && utf8[i] < 0x80)
        {
            i++;
        }

        if (i == utf8.Length)
        {
            return true;
        }

        Tables t = _tables.Value;
        while (i < utf8.Length)
        {
            Rune r = Utf8Util.DecodeAt(utf8, i, out int len);
            i += len;
            int c = r.Value;
            if (c < 0x10000)
            {
                if ((t.BmpNotInert[c >> 6] & (1UL << (c & 63))) != 0)
                {
                    return false;
                }
            }
            else if (t.AstralNotInert.Contains(c))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Crate <c>Recompositions</c> over <c>Decompositions</c> (canonical): characters with
    /// their upstream change values.</summary>
    public static List<CharChange> Nfc(List<Rune> input)
    {
        List<(int Ccc, int C, int Change)> decomposed = Decompose(input);
        var output = new List<CharChange>(decomposed.Count);
        var buffer = new List<(int C, int Change)>();
        (int C, int Change)? composee = null;
        int? lastCcc = null;

        foreach ((int chClass, int ch, int change) in decomposed)
        {
            if (composee is not { } k)
            {
                if (chClass != 0)
                {
                    output.Add(new CharChange(new Rune(ch), change));
                    continue;
                }

                composee = (ch, change);
                continue;
            }

            if (lastCcc is not int lClass)
            {
                if (Compose(k.C, ch) is int r)
                {
                    composee = (r, k.Change + change - 1);
                    continue;
                }

                if (chClass == 0)
                {
                    composee = (ch, change);
                    output.Add(new CharChange(new Rune(k.C), k.Change));
                    continue;
                }

                buffer.Add((ch, change));
                lastCcc = chClass;
                continue;
            }

            if (lClass >= chClass)
            {
                if (chClass == 0)
                {
                    // Emit the composee, then purge the blocked characters.
                    composee = (ch, change);
                    lastCcc = null;
                    output.Add(new CharChange(new Rune(k.C), k.Change));
                    foreach ((int bc, int bChange) in buffer)
                    {
                        output.Add(new CharChange(new Rune(bc), bChange));
                    }

                    buffer.Clear();
                    continue;
                }

                buffer.Add((ch, change));
                lastCcc = chClass;
                continue;
            }

            if (Compose(k.C, ch) is int composed)
            {
                composee = (composed, k.Change + change - 1);
                continue;
            }

            buffer.Add((ch, change));
            lastCcc = chClass;
        }

        if (composee is { } last)
        {
            output.Add(new CharChange(new Rune(last.C), last.Change));
        }

        foreach ((int bc, int bChange) in buffer)
        {
            output.Add(new CharChange(new Rune(bc), bChange));
        }

        return output;
    }

    /// <summary>Canonical decomposition and stable mark ordering used by BERT accent stripping.</summary>
    public static List<CharChange> Nfd(List<Rune> input) =>
        Decompose(input).Select(item => new CharChange(new Rune(item.C), item.Change)).ToList();

    /// <summary>Crate <c>Decompositions</c> (canonical): full decomposition, the first character of
    /// each input character with change 0 and the rest with change 1, non-starters stably sorted
    /// by combining class between starters.</summary>
    private static List<(int Ccc, int C, int Change)> Decompose(List<Rune> input)
    {
        var buffer = new List<(int Ccc, int C, int Change)>(input.Count + 4);
        int readyEnd = 0;
        Tables t = _tables.Value;

        void PushBack(int ch, bool first)
        {
            int cls = CombiningClass(t, ch);
            if (cls == 0)
            {
                SortPending(buffer, readyEnd);
                readyEnd = buffer.Count;
            }

            buffer.Add((cls, ch, first ? 0 : 1));
        }

        foreach (Rune r in input)
        {
            int c = r.Value;
            if (c <= 0x7F)
            {
                PushBack(c, true);
            }
            else if (c >= SBase && c < SBase + SCount)
            {
                int sIndex = c - SBase;
                PushBack(LBase + (sIndex / NCount), true);
                PushBack(VBase + ((sIndex % NCount) / TCount), false);
                int tIndex = sIndex % TCount;
                if (tIndex > 0)
                {
                    PushBack(TBase + tIndex, false);
                }
            }
            else if (t.Decompositions.TryGetValue(c, out int[]? chars))
            {
                for (int k = 0; k < chars.Length; k++)
                {
                    PushBack(chars[k], k == 0);
                }
            }
            else
            {
                PushBack(c, true);
            }
        }

        SortPending(buffer, readyEnd);
        return buffer;
    }

    /// <summary>Stable sort of <c>buffer[from..]</c> by combining class (Rust <c>sort_by_key</c>).</summary>
    private static void SortPending(List<(int Ccc, int C, int Change)> buffer, int from)
    {
        for (int i = from + 1; i < buffer.Count; i++)
        {
            (int Ccc, int C, int Change) item = buffer[i];
            int j = i - 1;
            while (j >= from && buffer[j].Ccc > item.Ccc)
            {
                buffer[j + 1] = buffer[j];
                j--;
            }

            buffer[j + 1] = item;
        }
    }

    /// <summary>Crate <c>compose</c>: Hangul first, then the composition table.</summary>
    private static int? Compose(int a, int b)
    {
        if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        {
            return SBase + ((((a - LBase) * VCount) + (b - VBase)) * TCount);
        }

        if (a >= SBase && a < SBase + SCount && b > TBase && b < TBase + TCount && (a - SBase) % TCount == 0)
        {
            return a + (b - TBase);
        }

        return _tables.Value.Compositions.TryGetValue(((long)a << 32) | (uint)b, out int c) ? c : null;
    }

    internal static int CombiningClass(int c) => CombiningClass(_tables.Value, c);

    private static int CombiningClass(Tables t, int c) =>
        c < 0x300 ? 0 : t.CombiningClasses.TryGetValue(c, out byte cls) ? cls : 0;

    private static Tables BuildTables()
    {
        var ccc = new Dictionary<int, byte>();
        foreach (int kv in Unicode9NormalizationData.CombiningClasses)
        {
            ccc[kv >> 8] = (byte)(kv & 0xFF);
        }

        var decompositions = new Dictionary<int, int[]>();
        ReadOnlySpan<int> index = Unicode9NormalizationData.DecompositionIndex;
        ReadOnlySpan<int> chars = Unicode9NormalizationData.DecompositionChars;
        for (int i = 0; i < index.Length; i += 3)
        {
            decompositions[index[i]] = chars.Slice(index[i + 1], index[i + 2]).ToArray();
        }

        var compositions = new Dictionary<long, int>();
        var seconds = new HashSet<int>();
        ReadOnlySpan<int> comp = Unicode9NormalizationData.Compositions;
        for (int i = 0; i < comp.Length; i += 3)
        {
            compositions[((long)comp[i] << 32) | (uint)comp[i + 1]] = comp[i + 2];
            seconds.Add(comp[i + 1]);
        }

        var bmp = new ulong[0x10000 / 64];
        var astral = new HashSet<int>();
        void Mark(int c)
        {
            if (c < 0x10000)
            {
                bmp[c >> 6] |= 1UL << (c & 63);
            }
            else
            {
                astral.Add(c);
            }
        }

        foreach (int c in ccc.Keys)
        {
            Mark(c);
        }

        foreach (int c in decompositions.Keys)
        {
            Mark(c);
        }

        foreach (int c in seconds)
        {
            Mark(c);
        }

        for (int c = SBase; c < SBase + SCount; c++)
        {
            Mark(c);
        }

        for (int c = VBase; c < VBase + VCount; c++)
        {
            Mark(c);
        }

        for (int c = TBase + 1; c < TBase + TCount; c++)
        {
            Mark(c);
        }

        return new Tables(ccc, decompositions, compositions, bmp, astral);
    }

    /// <summary>Read-only after construction; concurrent reads are safe.</summary>
    private sealed record Tables(
        Dictionary<int, byte> CombiningClasses,
        Dictionary<int, int[]> Decompositions,
        Dictionary<long, int> Compositions,
        ulong[] BmpNotInert,
        HashSet<int> AstralNotInert);
}
