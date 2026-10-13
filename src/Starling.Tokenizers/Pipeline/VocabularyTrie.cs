// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Runtime.CompilerServices;
using Starling.Tokenizers.ThirdParty.MLNet;

namespace Starling.Tokenizers.Pipeline;

/// <summary>
/// Builds the double-array trie of a Unigram vocabulary: every piece maps to its id, and a piece
/// that occurs more than once maps to its last id (upstream <c>token_to_ids</c> is filled in file
/// order, so later ids overwrite earlier ones).
/// </summary>
internal static class VocabularyTrie
{
    private const int InsertionSortThreshold = 12;

    /// <remarks>Pieces must be non-empty and free of NUL bytes (the model checks this first).</remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static DoubleArrayTrie Build(UnigramVocabulary vocab)
    {
        byte[] bytes = vocab.Bytes;
        int[] offsets = vocab.Offsets;
        // Unsigned byte order; a piece sorts before its extensions. Equal pieces end up adjacent.
        int[] order = SortedOrder(bytes, offsets, vocab.Count);

        // Keep one entry per distinct piece, with the largest (last) id among its duplicates, and
        // copy the keys into one buffer in sorted order: the builder reads them range by range.
        var sortedBytes = new byte[offsets[order.Length]];
        var starts = new int[order.Length + 1];
        int unique = 0;
        int position = 0;
        int i0 = 0;
        while (i0 < order.Length)
        {
            int best = order[i0];
            ReadOnlySpan<byte> piece = Piece(bytes, offsets, best);
            int j = i0 + 1;
            while (j < order.Length && Piece(bytes, offsets, order[j]).SequenceEqual(piece))
            {
                best = Math.Max(best, order[j]);
                j++;
            }

            piece.CopyTo(sortedBytes.AsSpan(position));
            starts[unique] = position;
            position += piece.Length;
            order[unique++] = best;
            i0 = j;
        }

        starts[unique] = position;
        return new DoubleArrayTrie(new DoubleArrayKeyset(sortedBytes, starts, order, unique));
    }

    /// <summary>
    /// Piece ids in unsigned byte order of their pieces. A radix sort on the first eight bytes
    /// (zero-padded; pieces contain no NUL, so padding sorts first like the end of a piece) orders
    /// almost everything; runs that share those eight bytes are finished by multikey quicksort.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int[] SortedOrder(byte[] bytes, int[] offsets, int count)
    {
        var keys = new ulong[count];
        var order = new int[count];
        for (int id = 0; id < count; id++)
        {
            ReadOnlySpan<byte> piece = Piece(bytes, offsets, id);
            ulong key = 0;
            int n = Math.Min(piece.Length, 8);
            for (int k = 0; k < n; k++)
            {
                key |= (ulong)piece[k] << (56 - (8 * k));
            }

            keys[id] = key;
            order[id] = id;
        }

        // LSD radix sort of (key, id) by key, one byte per pass; passes where every key has the
        // same byte are skipped.
        var keysOut = new ulong[count];
        var orderOut = new int[count];
        Span<int> histogram = stackalloc int[256];
        for (int shift = 0; shift < 64; shift += 8)
        {
            histogram.Clear();
            for (int i = 0; i < count; i++)
            {
                histogram[(int)(keys[i] >> shift) & 0xFF]++;
            }

            if (count == 0 || histogram[(int)(keys[0] >> shift) & 0xFF] == count)
            {
                continue;
            }

            int sum = 0;
            for (int b = 0; b < 256; b++)
            {
                int c = histogram[b];
                histogram[b] = sum;
                sum += c;
            }

            for (int i = 0; i < count; i++)
            {
                int b = (int)(keys[i] >> shift) & 0xFF;
                int target = histogram[b]++;
                keysOut[target] = keys[i];
                orderOut[target] = order[i];
            }

            (keys, keysOut) = (keysOut, keys);
            (order, orderOut) = (orderOut, order);
        }

        // Runs with equal 8-byte prefixes: order by the bytes after the eighth.
        int start = 0;
        while (start < count)
        {
            int end = start + 1;
            while (end < count && keys[end] == keys[start])
            {
                end++;
            }

            if (end - start > 1)
            {
                Sort(order, start, end, 8, bytes, offsets);
            }

            start = end;
        }

        return order;
    }

    private static ReadOnlySpan<byte> Piece(byte[] bytes, int[] offsets, int id) =>
        bytes.AsSpan(offsets[id], offsets[id + 1] - offsets[id]);

    /// <summary>Piece <paramref name="id"/> from byte <paramref name="depth"/> on (empty past its end).</summary>
    private static ReadOnlySpan<byte> Suffix(byte[] bytes, int[] offsets, int id, int depth)
    {
        ReadOnlySpan<byte> piece = Piece(bytes, offsets, id);
        return depth < piece.Length ? piece[depth..] : [];
    }

    /// <summary>The byte of piece <paramref name="id"/> at <paramref name="depth"/> plus one, or 0 past its end.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Key(byte[] bytes, int[] offsets, int id, int depth)
    {
        int position = offsets[id] + depth;
        return position < offsets[id + 1] ? bytes[position] + 1 : 0;
    }

    /// <summary>
    /// Multikey quicksort (Bentley and Sedgewick, "Fast Algorithms for Sorting and Searching
    /// Strings", 1997) of <c>order[lo..hi)</c>, whose pieces share their first
    /// <paramref name="depth"/> bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Sort(int[] order, int lo, int hi, int depth, byte[] bytes, int[] offsets)
    {
        while (hi - lo > InsertionSortThreshold)
        {
            int a = Key(bytes, offsets, order[lo], depth);
            int b = Key(bytes, offsets, order[lo + ((hi - lo) / 2)], depth);
            int c = Key(bytes, offsets, order[hi - 1], depth);
            int pivot = Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c));

            int lt = lo;
            int gt = hi - 1;
            int i = lo;
            while (i <= gt)
            {
                int k = Key(bytes, offsets, order[i], depth);
                if (k < pivot)
                {
                    (order[lt], order[i]) = (order[i], order[lt]);
                    lt++;
                    i++;
                }
                else if (k > pivot)
                {
                    (order[gt], order[i]) = (order[i], order[gt]);
                    gt--;
                }
                else
                {
                    i++;
                }
            }

            Sort(order, lo, lt, depth, bytes, offsets);
            Sort(order, gt + 1, hi, depth, bytes, offsets);
            if (pivot == 0)
            {
                // [lt, gt] are equal pieces that all end here.
                return;
            }

            lo = lt;
            hi = gt + 1;
            depth++;
        }

        for (int i = lo + 1; i < hi; i++)
        {
            int id = order[i];
            ReadOnlySpan<byte> piece = Suffix(bytes, offsets, id, depth);
            int j = i - 1;
            while (j >= lo && Suffix(bytes, offsets, order[j], depth).SequenceCompareTo(piece) > 0)
            {
                order[j + 1] = order[j];
                j--;
            }

            order[j + 1] = id;
        }
    }
}
