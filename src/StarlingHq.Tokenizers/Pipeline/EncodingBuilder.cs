// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported in part from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3),
// tokenizers/src/tokenizer/encoding.rs (merge_with, truncate, pad, sequence ids),
// Copyright 2020 The HuggingFace Inc. team, Apache License 2.0 (see THIRD-PARTY-NOTICES.md).
// Modified by StarlingHq: translated to C#; overflowing encodings are not produced.

using System.Runtime.InteropServices;

namespace StarlingHq.Tokenizers.Pipeline;

/// <summary>Mutable encoding used while a pipeline runs (upstream <c>Encoding</c>).</summary>
internal sealed class EncodingBuilder
{
    private List<int> _ids = [];
    private List<string> _tokens = [];
    private List<int> _typeIds = [];
    private List<int> _attentionMask = [];
    private List<int> _specialTokensMask = [];
    private List<(int Start, int End)> _scalarOffsets = [];
    private List<TokenOffset> _utf16Offsets = [];
    private Dictionary<int, (int Start, int End)> _sequenceRanges = [];
    private OffsetConverter? _converter;

    public int Count => _ids.Count;

    public void Add(int id, string token, int typeId, (int Start, int End) scalarOffsets, OffsetConverter converter)
    {
        _ids.Add(id);
        _tokens.Add(token);
        _typeIds.Add(typeId);
        _attentionMask.Add(1);
        _specialTokensMask.Add(0);
        _scalarOffsets.Add(scalarOffsets);
        _utf16Offsets.Add(converter.ScalarToUtf16(scalarOffsets));
        _converter = converter;
    }

    /// <summary>
    /// Upstream <c>byte_level::process_offsets</c> on this (single-sequence) encoding: leading and
    /// trailing <c>Ġ</c> (byte-level space) or whitespace characters of each token are removed from
    /// its offsets, in Unicode scalar values as upstream, and the UTF-16 offsets follow.
    /// </summary>
    public void TrimByteLevelOffsets(bool addPrefixSpace)
    {
        char space = ByteLevelAlphabet.ByteToChar[' '];
        for (int i = 0; i < _tokens.Count; i++)
        {
            List<System.Text.Rune> chars = [.. _tokens[i].EnumerateRunes()];
            static bool IsSpace(System.Text.Rune c, char space) => c.Value == space || UnicodeData.IsWhiteSpace(c);
            int leading = 0;
            while (leading < chars.Count && IsSpace(chars[leading], space))
            {
                leading++;
            }

            int trailing = 0;
            while (trailing < chars.Count && IsSpace(chars[chars.Count - 1 - trailing], space))
            {
                trailing++;
            }

            if (leading == 0 && trailing == 0)
            {
                continue;
            }

            (int start, int end) = _scalarOffsets[i];
            if (leading > 0)
            {
                bool isFirst = i == 0 || start == 0;
                if (isFirst && addPrefixSpace && leading == 1)
                {
                    leading = 0;
                }

                start = Math.Min(start + leading, end);
            }

            if (trailing > 0 && end >= trailing)
            {
                end = Math.Max(end - trailing, start);
            }

            _scalarOffsets[i] = (start, end);
            if (_converter is not null)
            {
                _utf16Offsets[i] = _converter.ScalarToUtf16((start, end));
            }
        }
    }

    public void AddSpecial(int id, string token, int typeId)
    {
        _ids.Add(id);
        _tokens.Add(token);
        _typeIds.Add(typeId);
        _attentionMask.Add(1);
        _specialTokensMask.Add(1);
        _scalarOffsets.Add((0, 0));
        _utf16Offsets.Add(default);
    }

    public void SetTypeIds(int typeId)
    {
        for (int i = 0; i < _typeIds.Count; i++)
        {
            _typeIds[i] = typeId;
        }
    }

    public void SetSequenceId(int sequenceId) => _sequenceRanges[sequenceId] = (0, _ids.Count);

    /// <summary>Upstream <c>merge_with(pair, growing_offsets: false)</c>.</summary>
    public void Append(EncodingBuilder other)
    {
        int shift = _ids.Count;
        foreach ((int seq, (int start, int end)) in other._sequenceRanges)
        {
            _sequenceRanges[seq] = (start + shift, end + shift);
        }

        _ids.AddRange(other._ids);
        _tokens.AddRange(other._tokens);
        _typeIds.AddRange(other._typeIds);
        _attentionMask.AddRange(other._attentionMask);
        _specialTokensMask.AddRange(other._specialTokensMask);
        _scalarOffsets.AddRange(other._scalarOffsets);
        _utf16Offsets.AddRange(other._utf16Offsets);
    }

    /// <summary>Upstream <c>truncate</c> with stride 0; the overflow is discarded.</summary>
    public void Truncate(int maxLength, TruncationDirection direction)
    {
        int length = _ids.Count;
        if (maxLength >= length)
        {
            return;
        }

        if (maxLength == 0)
        {
            _ids = [];
            _tokens = [];
            _typeIds = [];
            _attentionMask = [];
            _specialTokensMask = [];
            _scalarOffsets = [];
            _utf16Offsets = [];
            _sequenceRanges = [];
            return;
        }

        _sequenceRanges.Clear();
        int start = direction == TruncationDirection.Right ? 0 : length - maxLength;
        _ids = _ids.GetRange(start, maxLength);
        _tokens = _tokens.GetRange(start, maxLength);
        _typeIds = _typeIds.GetRange(start, maxLength);
        _attentionMask = _attentionMask.GetRange(start, maxLength);
        _specialTokensMask = _specialTokensMask.GetRange(start, maxLength);
        _scalarOffsets = _scalarOffsets.GetRange(start, maxLength);
        _utf16Offsets = _utf16Offsets.GetRange(start, maxLength);
    }

    /// <summary>Upstream <c>pad</c>.</summary>
    public void Pad(int targetLength, int padId, int padTypeId, string padToken, PaddingDirection direction)
    {
        if (_ids.Count >= targetLength)
        {
            return;
        }

        int padLength = targetLength - _ids.Count;
        int at = direction == PaddingDirection.Right ? _ids.Count : 0;
        _ids.InsertRange(at, Enumerable.Repeat(padId, padLength));
        _tokens.InsertRange(at, Enumerable.Repeat(padToken, padLength));
        _typeIds.InsertRange(at, Enumerable.Repeat(padTypeId, padLength));
        _attentionMask.InsertRange(at, Enumerable.Repeat(0, padLength));
        _specialTokensMask.InsertRange(at, Enumerable.Repeat(1, padLength));
        _scalarOffsets.InsertRange(at, Enumerable.Repeat((0, 0), padLength));
        _utf16Offsets.InsertRange(at, Enumerable.Repeat(default(TokenOffset), padLength));

        if (direction == PaddingDirection.Left)
        {
            foreach (int key in _sequenceRanges.Keys.ToList())
            {
                (int s, int e) = _sequenceRanges[key];
                _sequenceRanges[key] = (s + padLength, e + padLength);
            }
        }
    }

    public TokenEncoding Build()
    {
        // Upstream get_sequence_ids: with no recorded ranges the whole encoding is sequence 0.
        var sequenceIds = new int?[_ids.Count];
        int sequences = _sequenceRanges.Count == 0 ? 1 : _sequenceRanges.Count;
        for (int seq = 0; seq < sequences; seq++)
        {
            (int start, int end) = _sequenceRanges.TryGetValue(seq, out var range) ? range : (0, _ids.Count);
            for (int i = start; i < end && i < sequenceIds.Length; i++)
            {
                sequenceIds[i] = seq;
            }
        }

        var scalar = new TokenOffset[_scalarOffsets.Count];
        for (int i = 0; i < scalar.Length; i++)
        {
            scalar[i] = new TokenOffset(_scalarOffsets[i].Start, _scalarOffsets[i].End);
        }

        return new TokenEncoding(
            ImmutableCollectionsMarshal.AsImmutableArray(_ids.ToArray()),
            ImmutableCollectionsMarshal.AsImmutableArray(_tokens.ToArray()),
            ImmutableCollectionsMarshal.AsImmutableArray(_typeIds.ToArray()),
            ImmutableCollectionsMarshal.AsImmutableArray(_attentionMask.ToArray()),
            ImmutableCollectionsMarshal.AsImmutableArray(_specialTokensMask.ToArray()),
            ImmutableCollectionsMarshal.AsImmutableArray(_utf16Offsets.ToArray()),
            ImmutableCollectionsMarshal.AsImmutableArray(scalar),
            ImmutableCollectionsMarshal.AsImmutableArray(sequenceIds));
    }
}
