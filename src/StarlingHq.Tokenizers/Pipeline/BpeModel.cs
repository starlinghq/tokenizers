// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3):
// tokenizers/src/models/bpe/model.rs (BpeBuilder::build merge map, merge_word, tokenize,
// ignore_merges) and tokenizers/src/models/bpe/word.rs (Word::merge_all), Copyright The
// HuggingFace Inc. team, Apache License 2.0 (see THIRD-PARTY-NOTICES.md). Modified by
// StarlingHq: translated to C#; vocabulary and merge map are UTF-8 hash tables
// (BpeVocabulary, BpeMergeTable); no word cache (the cache does not change results); dropout,
// continuing_subword_prefix and end_of_word_suffix are rejected at load; an unk_token missing from
// the vocabulary is rejected at load (upstream fails at encode time when it is needed).

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace StarlingHq.Tokenizers.Pipeline;

/// <summary>The <c>BPE</c> model without dropout or affixes, with optional unknown token and byte fallback.</summary>
internal sealed class BpeModel : IModel
{
    private const int MaxCachedChar = 0x800;

    private readonly BpeVocabulary _vocab;
    private readonly BpeMergeTable _merges;
    private readonly bool _ignoreMerges;
    private readonly int _unkId;
    private readonly bool _fuseUnk;

    // Id of the token "<0xNN>" for each byte (upstream format!("<{b:#04X}>")), -1 where it is not in
    // the vocabulary; null without byte fallback.
    private readonly int[]? _byteFallbackIds;

    // Id of each single character below MaxCachedChar (-1: not in the vocabulary). Byte-level
    // pre-tokens consist of U+0021..U+0143 only, so the common path never hashes a character.
    private readonly int[] _charIds;

    /// <param name="vocab">Vocabulary.</param>
    /// <param name="merges">Merge map.</param>
    /// <param name="ignoreMerges">Upstream <c>ignore_merges</c>.</param>
    /// <param name="unkId">Id of <c>unk_token</c>, or -1 for none.</param>
    /// <param name="fuseUnk">Upstream <c>fuse_unk</c>.</param>
    /// <param name="byteFallback">Upstream <c>byte_fallback</c>.</param>
    public BpeModel(BpeVocabulary vocab, BpeMergeTable merges, bool ignoreMerges, int unkId = -1, bool fuseUnk = false, bool byteFallback = false)
    {
        _vocab = vocab;
        _merges = merges;
        _ignoreMerges = ignoreMerges;
        _unkId = unkId;
        _fuseUnk = fuseUnk;
        if (byteFallback)
        {
            _byteFallbackIds = new int[256];
            for (int b = 0; b < 256; b++)
            {
                _byteFallbackIds[b] = vocab.Find(ByteFallbackToken((byte)b));
            }
        }

        _charIds = new int[MaxCachedChar];
        Span<byte> buffer = stackalloc byte[4];
        for (int c = 0; c < MaxCachedChar; c++)
        {
            int len = new Rune(c).EncodeToUtf8(buffer);
            _charIds[c] = vocab.Find(buffer[..len]);
        }
    }

    public int VocabularySize => _vocab.Count;

    public bool IgnoreMerges => _ignoreMerges;

    public bool ByteFallback => _byteFallbackIds is not null;

    public bool FuseUnk => _fuseUnk;

    /// <summary>Id of the unknown token, or -1.</summary>
    public int UnkId => _unkId;

    /// <summary>The byte-fallback token of a byte: <c>&lt;0xNN&gt;</c> with two upper-case hex digits.</summary>
    internal static string ByteFallbackToken(byte b) => $"<0x{b:X2}>";

    internal BpeVocabulary Vocabulary => _vocab;

    internal BpeMergeTable Merges => _merges;

    public int? TokenToId(string token)
    {
        int id = _vocab.Find(token);
        return id >= 0 ? id : null;
    }

    public string? IdToToken(int id) => _vocab.Token(id);

    /// <summary>
    /// Builds the merge map exactly as upstream <c>BpeBuilder::build</c>: rank is the position in
    /// the list, both parts and their concatenation must be in the vocabulary, and a pair listed
    /// again overwrites the earlier entry.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static BpeMergeTable BuildMerges(BpeVocabulary vocab, IReadOnlyList<(byte[] Left, byte[] Right)> merges)
    {
        var table = new BpeMergeTable(merges.Count);
        for (int rank = 0; rank < merges.Count; rank++)
        {
            (byte[] left, byte[] right) = merges[rank];
            AddMerge(vocab, table, left, right, rank);
        }

        return table;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddMerge(BpeVocabulary vocab, BpeMergeTable table, ReadOnlySpan<byte> left, ReadOnlySpan<byte> right, int rank)
    {
        int a = vocab.Find(left);
        if (a < 0)
        {
            throw MergeTokenOutOfVocabulary(left);
        }

        int b = vocab.Find(right);
        if (b < 0)
        {
            throw MergeTokenOutOfVocabulary(right);
        }

        int merged = vocab.FindConcatenation(left, right);
        if (merged < 0)
        {
            throw MergeTokenOutOfVocabulary([.. left, .. right]);
        }

        table.Set(a, b, rank, merged);
    }

    private static TokenizerFormatException MergeTokenOutOfVocabulary(ReadOnlySpan<byte> token) =>
        new($"model.type=BPE: merge token '{Encoding.UTF8.GetString(token)}' is not in the vocabulary.");

    /// <summary>Upstream <c>Model::tokenize</c> for BPE (dropout none).</summary>
    public List<ModelToken> Tokenize(ReadOnlySpan<byte> sequence)
    {
        if (sequence.IsEmpty)
        {
            return [];
        }

        if (_ignoreMerges)
        {
            int whole = _vocab.Find(sequence);
            if (whole >= 0)
            {
                return [new ModelToken(whole, Encoding.UTF8.GetString(sequence), 0, sequence.Length)];
            }
        }

        Symbol[] rented = ArrayPool<Symbol>.Shared.Rent(sequence.Length);
        try
        {
            int count = MergeWord(sequence, rented);
            var tokens = new List<ModelToken>(count);
            int offset = 0;
            for (int i = 0; i < count; i++)
            {
                ref Symbol s = ref rented[i];
                if (s.Len == 0)
                {
                    continue;
                }

                tokens.Add(new ModelToken(s.C, _vocab.Token(s.C)!, offset, offset + s.Len));
                offset += s.Len;
            }

            return tokens;
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Upstream <c>merge_word</c> then <c>Word::merge_all</c>: one symbol per character in the
    /// vocabulary; a character outside it becomes one <c>&lt;0xNN&gt;</c> symbol per UTF-8 byte with
    /// byte fallback (when all its byte tokens exist), else the unknown token (consecutive unknown
    /// characters fused into one with <c>fuse_unk</c>), else it is dropped. A pending unknown symbol
    /// is emitted only when the next in-vocabulary character or the end of the word is reached, as
    /// upstream (byte-fallback symbols in between come first). Then the lowest-ranked pair is merged
    /// first, leftmost on ties, until none is left. Removed symbols keep Len 0.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int MergeWord(ReadOnlySpan<byte> w, Symbol[] symbols)
    {
        int count = 0;
        int i = 0;
        int pendingUnkLen = 0;
        while (i < w.Length)
        {
            int len = Utf8Util.SequenceLength(w, i);
            int id;
            byte first = w[i];
            if (len == 1)
            {
                id = _charIds[first];
            }
            else if (len == 2)
            {
                id = _charIds[((first & 0x1F) << 6) | (w[i + 1] & 0x3F)];
            }
            else
            {
                id = _vocab.Find(w.Slice(i, len));
            }

            if (id >= 0)
            {
                if (pendingUnkLen > 0)
                {
                    AddSymbol(symbols, ref count, _unkId, pendingUnkLen);
                    pendingUnkLen = 0;
                }

                AddSymbol(symbols, ref count, id, len);
            }
            else if (_byteFallbackIds is not null && AllBytesHaveTokens(w.Slice(i, len)))
            {
                for (int k = 0; k < len; k++)
                {
                    AddSymbol(symbols, ref count, _byteFallbackIds[w[i + k]], 1);
                }
            }
            else if (_unkId >= 0)
            {
                if (pendingUnkLen > 0 && !_fuseUnk)
                {
                    AddSymbol(symbols, ref count, _unkId, pendingUnkLen);
                    pendingUnkLen = 0;
                }

                pendingUnkLen += len;
            }

            i += len;
        }

        if (pendingUnkLen > 0)
        {
            AddSymbol(symbols, ref count, _unkId, pendingUnkLen);
        }

        if (count < 2)
        {
            return count;
        }

        var queue = new MergeQueue(count);
        for (int index = 0; index + 1 < count; index++)
        {
            if (_merges.TryGet(symbols[index].C, symbols[index + 1].C, out int rank, out int merged))
            {
                queue.Push(new Merge(index, rank, merged));
            }
        }

        while (queue.TryPop(out Merge top))
        {
            ref Symbol current = ref symbols[top.Pos];
            if (current.Len == 0 || current.Next == -1)
            {
                continue;
            }

            int nextPos = current.Next;
            Symbol right = symbols[nextPos];

            // Expired entry: the pair at this position no longer produces this merge.
            if (!_merges.TryGet(current.C, right.C, out _, out int expected) || expected != top.NewId)
            {
                continue;
            }

            current.C = top.NewId;
            current.Len += right.Len;
            current.Next = right.Next;
            symbols[nextPos].Len = 0;
            if (right.Next > -1 && right.Next < count)
            {
                symbols[right.Next].Prev = top.Pos;
            }

            if (current.Prev >= 0)
            {
                int prev = current.Prev;
                if (_merges.TryGet(symbols[prev].C, current.C, out int rank, out int merged))
                {
                    queue.Push(new Merge(prev, rank, merged));
                }
            }

            int next = current.Next;
            if (next >= 0 && next < count && _merges.TryGet(current.C, symbols[next].C, out int nextRank, out int nextMerged))
            {
                queue.Push(new Merge(top.Pos, nextRank, nextMerged));
            }
        }

        queue.Dispose();
        return count;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddSymbol(Symbol[] symbols, ref int count, int id, int len)
    {
        symbols[count] = new Symbol(id, count - 1, -1, len);
        if (count > 0)
        {
            symbols[count - 1].Next = count;
        }

        count++;
    }

    private bool AllBytesHaveTokens(ReadOnlySpan<byte> bytes)
    {
        foreach (byte b in bytes)
        {
            if (_byteFallbackIds![b] < 0)
            {
                return false;
            }
        }

        return true;
    }

    private struct Symbol(int c, int prev, int next, int len)
    {
        public int C = c;
        public int Prev = prev;
        public int Next = next;
        public int Len = len;
    }

    private readonly record struct Merge(int Pos, int Rank, int NewId);

    /// <summary>Min-heap on (rank, position): the order upstream's <c>Merge</c> defines.</summary>
    private struct MergeQueue : IDisposable
    {
        private Merge[] _heap;
        private int _count;

        public MergeQueue(int capacity)
        {
            _heap = ArrayPool<Merge>.Shared.Rent(Math.Max(capacity, 4));
            _count = 0;
        }

        public void Push(Merge m)
        {
            if (_count == _heap.Length)
            {
                Merge[] bigger = ArrayPool<Merge>.Shared.Rent(_count * 2);
                _heap.AsSpan(0, _count).CopyTo(bigger);
                ArrayPool<Merge>.Shared.Return(_heap);
                _heap = bigger;
            }

            int i = _count++;
            while (i > 0)
            {
                int parent = (i - 1) >> 1;
                if (!Less(m, _heap[parent]))
                {
                    break;
                }

                _heap[i] = _heap[parent];
                i = parent;
            }

            _heap[i] = m;
        }

        public bool TryPop(out Merge top)
        {
            if (_count == 0)
            {
                top = default;
                return false;
            }

            top = _heap[0];
            Merge last = _heap[--_count];
            int i = 0;
            while (true)
            {
                int child = (2 * i) + 1;
                if (child >= _count)
                {
                    break;
                }

                if (child + 1 < _count && Less(_heap[child + 1], _heap[child]))
                {
                    child++;
                }

                if (!Less(_heap[child], last))
                {
                    break;
                }

                _heap[i] = _heap[child];
                i = child;
            }

            if (_count > 0)
            {
                _heap[i] = last;
            }

            return true;
        }

        public void Dispose()
        {
            if (_heap is not null)
            {
                ArrayPool<Merge>.Shared.Return(_heap);
                _heap = null!;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Less(Merge a, Merge b) => a.Rank != b.Rank ? a.Rank < b.Rank : a.Pos < b.Pos;
    }
}
