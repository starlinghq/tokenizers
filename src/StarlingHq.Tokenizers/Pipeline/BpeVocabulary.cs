// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace StarlingHq.Tokenizers.Pipeline;

/// <summary>
/// The BPE <c>model.vocab</c> map (token to id) kept as UTF-8: every token's bytes in one buffer,
/// an id-indexed table, and an open-addressing hash over the bytes. Token strings are created on
/// demand. Built once at load; read-only afterwards.
/// </summary>
internal sealed class BpeVocabulary
{
    private byte[] _bytes;
    private int _byteCount;
    private Entry[] _entries;
    private int _count;

    // Built by Seal(): slot = (hash << 32) | (entry + 1), 0 for empty. Comparing the stored hash
    // first means a lookup usually touches one slot, one entry and the token bytes.
    private ulong[] _slots = [];
    private int _mask;
    private int[] _entryOfId = [];
    private string?[] _tokens = [];

    public BpeVocabulary(int expectedEntries = 16, int expectedBytes = 64)
    {
        _bytes = new byte[Math.Max(expectedBytes, 16)];
        _entries = new Entry[Math.Max(expectedEntries, 4)];
    }

    /// <summary>Number of distinct tokens (upstream <c>vocab.len()</c>).</summary>
    public int Count => _count;

    /// <summary>One past the largest id (ids may have holes).</summary>
    public int IdLimit => _entryOfId.Length;

    /// <summary>Adds a token. Call <see cref="Seal"/> after the last one.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(ReadOnlySpan<byte> utf8, int id)
    {
        if (_count == _entries.Length)
        {
            Array.Resize(ref _entries, _count * 2);
        }

        if (_byteCount + utf8.Length > _bytes.Length)
        {
            Array.Resize(ref _bytes, Math.Max(_bytes.Length * 2, _byteCount + utf8.Length));
        }

        utf8.CopyTo(_bytes.AsSpan(_byteCount));
        _entries[_count] = new Entry(_byteCount, utf8.Length, id);
        _byteCount += utf8.Length;
        _count++;
    }

    public void Add(string token, int id) => Add(Encoding.UTF8.GetBytes(token), id);

    /// <summary>
    /// Builds the hash and the id table. Duplicate tokens and duplicate ids are rejected: upstream
    /// keeps the last duplicate token and an arbitrary token for a duplicated id, which this library
    /// does not reproduce.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Seal()
    {
        int capacity = (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(16, _count * 2));
        _mask = capacity - 1;
        _slots = new ulong[capacity];
        int maxId = -1;
        for (int e = 0; e < _count; e++)
        {
            Entry entry = _entries[e];
            if (entry.Id > maxId)
            {
                maxId = entry.Id;
            }

            ReadOnlySpan<byte> key = _bytes.AsSpan(entry.Start, entry.Length);
            uint hash = Mix(Fnv(key));
            int slot = (int)hash & _mask;
            while (true)
            {
                ulong value = _slots[slot];
                if (value == 0)
                {
                    _slots[slot] = ((ulong)hash << 32) | (uint)(e + 1);
                    break;
                }

                if ((uint)(value >> 32) == hash)
                {
                    Entry existing = _entries[(int)(uint)value - 1];
                    if (key.SequenceEqual(_bytes.AsSpan(existing.Start, existing.Length)))
                    {
                        throw new UnsupportedTokenizerComponentException(
                            "model.vocab", $"token '{Encoding.UTF8.GetString(key)}' appears more than once; duplicate keys are not supported.");
                    }
                }

                slot = (slot + 1) & _mask;
            }
        }

        _entryOfId = new int[maxId + 1];
        _entryOfId.AsSpan().Fill(-1);
        for (int e = 0; e < _count; e++)
        {
            ref int slotOfId = ref _entryOfId[_entries[e].Id];
            if (slotOfId >= 0)
            {
                throw new UnsupportedTokenizerComponentException(
                    "model.vocab", $"id {_entries[e].Id} is assigned to more than one token; duplicate ids are not supported.");
            }

            slotOfId = e;
        }

        _tokens = new string?[_entryOfId.Length];
    }

    /// <summary>Id of the token with exactly these UTF-8 bytes, or -1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Find(ReadOnlySpan<byte> key)
    {
        uint hash = Mix(Fnv(key));
        int slot = (int)hash & _mask;
        while (true)
        {
            ulong value = _slots[slot];
            if (value == 0)
            {
                return -1;
            }

            if ((uint)(value >> 32) == hash)
            {
                Entry e = _entries[(int)(uint)value - 1];
                if (e.Length == key.Length && key.SequenceEqual(_bytes.AsSpan(e.Start, e.Length)))
                {
                    return e.Id;
                }
            }

            slot = (slot + 1) & _mask;
        }
    }

    /// <summary>Id of the token whose bytes are <paramref name="a"/> followed by <paramref name="b"/>, or -1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int FindConcatenation(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        uint hash = Mix(Fnv(b, Fnv(a)));
        int slot = (int)hash & _mask;
        int length = a.Length + b.Length;
        while (true)
        {
            ulong value = _slots[slot];
            if (value == 0)
            {
                return -1;
            }

            if ((uint)(value >> 32) == hash)
            {
                Entry e = _entries[(int)(uint)value - 1];
                if (e.Length == length)
                {
                    ReadOnlySpan<byte> token = _bytes.AsSpan(e.Start, length);
                    if (token[..a.Length].SequenceEqual(a) && token[a.Length..].SequenceEqual(b))
                    {
                        return e.Id;
                    }
                }
            }

            slot = (slot + 1) & _mask;
        }
    }

    public int Find(string token) => Find(Encoding.UTF8.GetBytes(token));

    /// <summary>UTF-8 bytes of the token with this id; empty for an unused id.</summary>
    public ReadOnlySpan<byte> TokenBytes(int id)
    {
        if ((uint)id >= (uint)_entryOfId.Length || _entryOfId[id] < 0)
        {
            return default;
        }

        Entry e = _entries[_entryOfId[id]];
        return _bytes.AsSpan(e.Start, e.Length);
    }

    public bool Contains(int id) => (uint)id < (uint)_entryOfId.Length && _entryOfId[id] >= 0;

    /// <summary>Token string of an id (cached after the first call), or null.</summary>
    public string? Token(int id)
    {
        if (!Contains(id))
        {
            return null;
        }

        return _tokens[id] ??= Encoding.UTF8.GetString(TokenBytes(id));
    }

    /// <summary>FNV-1a over the bytes, continuing from <paramref name="state"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Fnv(ReadOnlySpan<byte> bytes, uint state = 2166136261)
    {
        foreach (byte b in bytes)
        {
            state = (state ^ b) * 16777619;
        }

        return state;
    }

    /// <summary>Spreads the high bits into the low bits used for the slot index.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Mix(uint h) => h ^ (h >> 15);

    private readonly record struct Entry(int Start, int Length, int Id);
}

/// <summary>
/// Upstream <c>MergeMap</c>: <c>(left id, right id)</c> to <c>(rank, merged id)</c>, open addressing
/// on a 64-bit key. A pair listed twice keeps the later rank, as upstream's map insertion does.
/// </summary>
internal sealed class BpeMergeTable
{
    // Key is ((left << 32) | right) + 1, so 0 marks an empty slot.
    private readonly Slot[] _slots;
    private readonly int _shift;
    private int _count;

    public BpeMergeTable(int expected)
    {
        int capacity = (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(16, expected * 2));
        _slots = new Slot[capacity];
        _shift = 64 - BitOperations.Log2((uint)capacity);
    }

    public int Count => _count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(int left, int right, int rank, int merged)
    {
        long key = (((long)left << 32) | (uint)right) + 1;
        int mask = _slots.Length - 1;
        int slot = SlotOf(key);
        while (_slots[slot].Key != 0 && _slots[slot].Key != key)
        {
            slot = (slot + 1) & mask;
        }

        if (_slots[slot].Key == 0)
        {
            if (_count + 1 > _slots.Length / 2)
            {
                throw new InvalidOperationException("BpeMergeTable is full.");
            }

            _count++;
        }

        _slots[slot] = new Slot(key, rank, merged);
    }

    /// <summary>Rank and merged id of a pair, or false.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGet(int left, int right, out int rank, out int merged)
    {
        long key = (((long)left << 32) | (uint)right) + 1;
        int mask = _slots.Length - 1;
        int slot = SlotOf(key);
        while (true)
        {
            ref Slot s = ref _slots[slot];
            if (s.Key == key)
            {
                rank = s.Rank;
                merged = s.Merged;
                return true;
            }

            if (s.Key == 0)
            {
                rank = merged = 0;
                return false;
            }

            slot = (slot + 1) & mask;
        }
    }

    /// <summary>Every entry as (left, right, rank, merged), in no particular order.</summary>
    public IEnumerable<(int Left, int Right, int Rank, int Merged)> Entries()
    {
        foreach (Slot s in _slots)
        {
            if (s.Key != 0)
            {
                long pair = s.Key - 1;
                yield return ((int)(pair >> 32), (int)pair, s.Rank, s.Merged);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int SlotOf(long key) => (int)(((ulong)key * 0x9E3779B97F4A7C15UL) >> _shift);

    private readonly record struct Slot(long Key, int Rank, int Merged);
}
