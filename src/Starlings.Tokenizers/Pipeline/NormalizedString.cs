// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3),
// tokenizers/src/tokenizer/normalizer.rs, Copyright 2020 The HuggingFace Inc. team,
// Apache License 2.0 (see THIRD-PARTY-NOTICES.md). Modified by StarlingHq: translated to
// C#, limited to the operations the supported components use. Alignment semantics, including
// upstream quirks, are kept byte for byte because offsets are conformance-tested.

using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>Original-string byte range that one normalized byte came from.</summary>
internal readonly record struct Alignment(int Start, int End);

/// <summary>One character of a transformation and how it changes the string.</summary>
/// <remarks><c>Change</c> is 1 for an inserted character, 0 for a replacement, and -N for a
/// replacement followed by N removed characters (upstream <c>transform_range</c> contract).</remarks>
internal readonly record struct CharChange(Rune Char, int Change);

/// <summary>A match produced by a split or replace pattern over normalized UTF-8 bytes.</summary>
internal readonly record struct PatternMatch(int Start, int End, bool IsMatch);

/// <summary>How a split treats the delimiter (upstream <c>SplitDelimiterBehavior</c>).</summary>
internal enum SplitDelimiterBehavior
{
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

/// <summary>
/// A string under normalization: the original UTF-8 bytes, the normalized UTF-8 bytes, and for
/// every normalized byte the original byte range it came from. All offsets are UTF-8 bytes, as
/// upstream.
/// </summary>
internal sealed class NormalizedString
{
    private byte[] _original;
    private byte[] _normalized;
    private Alignment[] _alignments;
    private int _originalShift;

    private NormalizedString(byte[] original, byte[] normalized, Alignment[] alignments, int originalShift)
    {
        _original = original;
        _normalized = normalized;
        _alignments = alignments;
        _originalShift = originalShift;
    }

    public static NormalizedString FromUtf8(byte[] utf8)
    {
        var alignments = new Alignment[utf8.Length];
        int i = 0;
        while (i < utf8.Length)
        {
            int len = Utf8Util.SequenceLength(utf8, i);
            for (int k = 0; k < len; k++)
            {
                alignments[i + k] = new Alignment(i, i + len);
            }

            i += len;
        }

        return new NormalizedString(utf8, (byte[])utf8.Clone(), alignments, 0);
    }

    public static NormalizedString FromString(string s) => FromUtf8(Encoding.UTF8.GetBytes(s));

    /// <summary>Normalized UTF-8 bytes. Callers must not mutate the array.</summary>
    public byte[] Normalized => _normalized;

    public int Length => _normalized.Length;

    public int OriginalLength => _original.Length;

    public bool IsEmpty => _normalized.Length == 0;

    public string NormalizedText => Encoding.UTF8.GetString(_normalized);

    internal Alignment[] Alignments => _alignments;

    /// <summary>Offsets of this (sub)string in the full original input, in bytes.</summary>
    public (int Start, int End) OffsetsOriginal() => (_originalShift, _originalShift + _original.Length);

    public NormalizedString Clone() =>
        new((byte[])_original.Clone(), (byte[])_normalized.Clone(), (Alignment[])_alignments.Clone(), _originalShift);

    /// <summary>Upstream <c>convert_offsets(Range::Original(start..end))</c>.</summary>
    public (int Start, int End)? ConvertOriginalToNormalized(int start, int end)
    {
        if (start == end)
        {
            return (start, end);
        }

        if (start > end)
        {
            return null;
        }

        if (_original.Length == 0 && start == 0 && end == 0)
        {
            return (0, _normalized.Length);
        }

        int? s = null;
        int? e = null;
        for (int i = 0; i < _alignments.Length; i++)
        {
            Alignment a = _alignments[i];
            if (!(end >= a.End))
            {
                break;
            }

            if (s is null && start <= a.Start && a.Start != a.End)
            {
                s = i;
            }

            e = i + 1;
        }

        return (s, e) switch
        {
            (int si, null) => (si, si),
            (null, int ei) => (ei, ei),
            (int si, int ei) => (si, ei),
            _ => null,
        };
    }

    /// <summary>Upstream <c>convert_offsets(Range::Normalized(start..end))</c>.</summary>
    public (int Start, int End)? ConvertNormalizedToOriginal(int start, int end)
    {
        if (start == end)
        {
            return (start, end);
        }

        if (start > end)
        {
            return null;
        }

        if (_normalized.Length == 0 && start == 0 && end == 0)
        {
            return (0, _original.Length);
        }

        if (end > _alignments.Length)
        {
            return null;
        }

        return (_alignments[start].Start, _alignments[end - 1].End);
    }

    /// <summary>Upstream <c>slice(Range::Normalized(start..end))</c>.</summary>
    public NormalizedString? SliceNormalized(int start, int end)
    {
        if (start > end || end > _normalized.Length
            || !Utf8Util.IsCharBoundary(_normalized, start) || !Utf8Util.IsCharBoundary(_normalized, end))
        {
            return null;
        }

        (int Start, int End)? originalRange = ConvertNormalizedToOriginal(start, end);
        if (originalRange is not { } o)
        {
            return null;
        }

        int shift = o.Start;
        byte[] original = o.Start <= o.End && o.End <= _original.Length
            && Utf8Util.IsCharBoundary(_original, o.Start) && Utf8Util.IsCharBoundary(_original, o.End)
            ? _original.AsSpan(o.Start, o.End - o.Start).ToArray()
            : [];

        var alignments = new Alignment[end - start];
        for (int i = 0; i < alignments.Length; i++)
        {
            Alignment a = _alignments[start + i];
            alignments[i] = new Alignment(a.Start - shift, a.End - shift);
        }

        return new NormalizedString(
            original,
            _normalized.AsSpan(start, end - start).ToArray(),
            alignments,
            _originalShift + o.Start);
    }

    /// <summary>Upstream <c>transform</c>: transform over the whole original range.</summary>
    public void Transform(IReadOnlyList<CharChange> dest, int initialOffset)
    {
        if (ConvertOriginalToNormalized(0, _original.Length) is { } range)
        {
            TransformNormalizedRange(range.Start, range.End, dest, initialOffset);
        }
    }

    /// <summary>Upstream <c>transform_range</c> over a normalized byte range.</summary>
    public void TransformNormalizedRange(int start, int end, IReadOnlyList<CharChange> dest, int initialOffset)
    {
        List<Rune> replaced = Utf8Util.DecodeRunes(_normalized.AsSpan(start, end - start));
        int next = 0;

        int initialRemoved = 0;
        for (int k = 0; k < initialOffset && next < replaced.Count; k++)
        {
            initialRemoved += replaced[next++].Utf8SequenceLength;
        }

        int offset = initialRemoved + start;
        var newAlignments = new List<Alignment>(end - start);
        var newBytes = new List<byte>(end - start);
        Span<byte> buffer = stackalloc byte[4];

        foreach (CharChange change in dest)
        {
            int idx = offset;
            Alignment align = change.Change > 0
                ? (idx < 1 ? new Alignment(0, 0) : _alignments[idx - 1])
                : _alignments[idx];

            int replacedSize = 0;
            if (change.Change <= 0 && next < replaced.Count)
            {
                replacedSize = replaced[next++].Utf8SequenceLength;
            }

            int removedBytes = 0;
            if (change.Change < 0)
            {
                for (int k = 0; k < -change.Change && next < replaced.Count; k++)
                {
                    removedBytes += replaced[next++].Utf8SequenceLength;
                }
            }

            offset += replacedSize + removedBytes;

            int written = change.Char.EncodeToUtf8(buffer);
            for (int k = 0; k < written; k++)
            {
                newAlignments.Add(align);
                newBytes.Add(buffer[k]);
            }
        }

        _alignments = Splice(_alignments, start, end, newAlignments);
        _normalized = Splice(_normalized, start, end, newBytes);
    }

    /// <summary>
    /// Upstream <c>ByteLevel</c> pre-tokenizer transform: <c>transform</c> over the whole original
    /// range with, for each character, its UTF-8 bytes mapped to byte-level characters, the first
    /// with change 0 and the others with change 1. Same result as
    /// <see cref="TransformNormalizedRange"/> with that list, computed without it: the first
    /// character takes the alignment of the replaced character's first byte, the inserted ones
    /// the alignment of its last byte.
    /// </summary>
    public void ByteLevelTransform()
    {
        if (ConvertOriginalToNormalized(0, _original.Length) is not { } range)
        {
            return;
        }

        int start = range.Start;
        int end = range.End;
        char[] byteToChar = ByteLevelAlphabet.ByteToChar;
        var newBytes = new List<byte>((end - start) * 2);
        var newAlignments = new List<Alignment>((end - start) * 2);
        int i = start;
        while (i < end)
        {
            int len = Utf8Util.SequenceLength(_normalized, i);
            for (int k = 0; k < len; k++)
            {
                char c = byteToChar[_normalized[i + k]];
                Alignment align = k == 0 ? _alignments[i] : _alignments[i + len - 1];
                if (c < 0x80)
                {
                    newBytes.Add((byte)c);
                    newAlignments.Add(align);
                }
                else
                {
                    newBytes.Add((byte)(0xC0 | (c >> 6)));
                    newBytes.Add((byte)(0x80 | (c & 0x3F)));
                    newAlignments.Add(align);
                    newAlignments.Add(align);
                }
            }

            i += len;
        }

        _alignments = Splice(_alignments, start, end, newAlignments);
        _normalized = Splice(_normalized, start, end, newBytes);
    }

    /// <summary>Upstream <c>replace</c>: replace every match with <paramref name="content"/>.</summary>
    public void Replace(IReadOnlyList<PatternMatch> matches, string content)
    {
        var newNormalized = new List<byte>(_normalized.Length);
        var newAlignments = new List<Alignment>(_alignments.Length);
        int lastEnd = 0;
        byte[] contentBytes = Encoding.UTF8.GetBytes(content);

        foreach (PatternMatch m in matches)
        {
            if (!m.IsMatch)
            {
                continue;
            }

            AppendRange(newNormalized, _normalized, lastEnd, m.Start);
            AppendRange(newAlignments, _alignments, lastEnd, m.Start);

            // Every content character is an insertion (change 1) after all matched characters
            // are removed, so each takes the alignment of the last matched byte.
            int idx = m.End;
            Alignment align = idx < 1 ? new Alignment(0, 0) : _alignments[idx - 1];
            foreach (byte b in contentBytes)
            {
                newNormalized.Add(b);
                newAlignments.Add(align);
            }

            lastEnd = m.End;
        }

        AppendRange(newNormalized, _normalized, lastEnd, _normalized.Length);
        AppendRange(newAlignments, _alignments, lastEnd, _alignments.Length);
        _normalized = [.. newNormalized];
        _alignments = [.. newAlignments];
    }

    /// <summary>Upstream <c>prepend</c>.</summary>
    public void Prepend(string s)
    {
        if (_normalized.Length == 0)
        {
            return;
        }

        Rune first = Utf8Util.DecodeAt(_normalized, 0, out int firstLen);
        var dest = new List<CharChange>();
        int i = 0;
        foreach (Rune r in s.EnumerateRunes())
        {
            dest.Add(new CharChange(r, i != 0 ? 1 : 0));
            i++;
        }

        dest.Add(new CharChange(first, 1));
        TransformNormalizedRange(0, firstLen, dest, 0);
    }

    /// <summary>Upstream <c>lrstrip</c>, including its byte-length comparison.</summary>
    public void Strip(bool left, bool right)
    {
        List<Rune> chars = Utf8Util.DecodeRunes(_normalized);
        int leading = 0;
        if (left)
        {
            while (leading < chars.Count && UnicodeData.IsWhiteSpace(chars[leading]))
            {
                leading++;
            }
        }

        int trailing = 0;
        if (right)
        {
            while (trailing < chars.Count && UnicodeData.IsWhiteSpace(chars[chars.Count - 1 - trailing]))
            {
                trailing++;
            }
        }

        if (leading == 0 && trailing == 0)
        {
            return;
        }

        int count = chars.Count;
        var dest = new List<CharChange>(count);
        for (int i = 0; i < count; i++)
        {
            if (i < leading || i >= count - trailing)
            {
                continue;
            }

            // Upstream compares a character index with the byte length here.
            dest.Add(i == _normalized.Length - trailing - 1
                ? new CharChange(chars[i], -trailing)
                : new CharChange(chars[i], 0));
        }

        Transform(dest, leading);
    }

    /// <summary>Upstream <c>split</c> with a precomputed, contiguous list of matches.</summary>
    public List<NormalizedString> Split(IReadOnlyList<PatternMatch> matches, SplitDelimiterBehavior behavior)
    {
        var splits = new List<(int Start, int End, bool Remove)>(matches.Count);
        switch (behavior)
        {
            case SplitDelimiterBehavior.Isolated:
                foreach (PatternMatch m in matches)
                {
                    splits.Add((m.Start, m.End, false));
                }

                break;
            case SplitDelimiterBehavior.Removed:
                foreach (PatternMatch m in matches)
                {
                    splits.Add((m.Start, m.End, m.IsMatch));
                }

                break;
            case SplitDelimiterBehavior.Contiguous:
                {
                    bool previous = false;
                    foreach (PatternMatch m in matches)
                    {
                        if (m.IsMatch == previous && splits.Count > 0)
                        {
                            (int s, _, bool r) = splits[^1];
                            splits[^1] = (s, m.End, r);
                        }
                        else
                        {
                            splits.Add((m.Start, m.End, false));
                        }

                        previous = m.IsMatch;
                    }

                    break;
                }

            case SplitDelimiterBehavior.MergedWithPrevious:
                {
                    bool previous = false;
                    foreach (PatternMatch m in matches)
                    {
                        if (m.IsMatch && !previous && splits.Count > 0)
                        {
                            (int s, _, bool r) = splits[^1];
                            splits[^1] = (s, m.End, r);
                        }
                        else
                        {
                            splits.Add((m.Start, m.End, false));
                        }

                        previous = m.IsMatch;
                    }

                    break;
                }

            case SplitDelimiterBehavior.MergedWithNext:
                {
                    bool previous = false;
                    for (int i = matches.Count - 1; i >= 0; i--)
                    {
                        PatternMatch m = matches[i];
                        if (m.IsMatch && !previous && splits.Count > 0)
                        {
                            (_, int e, bool r) = splits[^1];
                            splits[^1] = (m.Start, e, r);
                        }
                        else
                        {
                            splits.Add((m.Start, m.End, false));
                        }

                        previous = m.IsMatch;
                    }

                    splits.Reverse();
                    break;
                }

            default:
                throw new ArgumentOutOfRangeException(nameof(behavior));
        }

        var result = new List<NormalizedString>(splits.Count);
        foreach ((int start, int end, bool remove) in splits)
        {
            if (!remove)
            {
                result.Add(SliceNormalized(start, end)
                    ?? throw new InvalidOperationException("NormalizedString bad split"));
            }
        }

        return result;
    }

    private static T[] Splice<T>(T[] source, int start, int end, List<T> insert)
    {
        var result = new T[source.Length - (end - start) + insert.Count];
        source.AsSpan(0, start).CopyTo(result);
        insert.CopyTo(result, start);
        source.AsSpan(end).CopyTo(result.AsSpan(start + insert.Count));
        return result;
    }

    private static void AppendRange<T>(List<T> target, T[] source, int start, int end)
    {
        for (int i = start; i < end; i++)
        {
            target.Add(source[i]);
        }
    }
}
