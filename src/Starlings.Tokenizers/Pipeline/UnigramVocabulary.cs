// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Runtime.CompilerServices;
using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>
/// The <c>[piece, score]</c> list of a Unigram model in file order, with every piece stored as
/// UTF-8 in one shared buffer. Pieces are kept as bytes because that is what the trie is built
/// from and what the encoder matches; strings are created only when an id is turned back into a
/// token.
/// </summary>
internal sealed class UnigramVocabulary
{
    private byte[] _bytes;
    private int[] _offsets;
    private double[] _scores;
    private int _byteCount;

    public UnigramVocabulary(int pieceCapacity = 1024, int byteCapacity = 16 * 1024)
    {
        _bytes = new byte[Math.Max(byteCapacity, 16)];
        _offsets = new int[Math.Max(pieceCapacity, 1) + 1];
        _scores = new double[Math.Max(pieceCapacity, 1)];
    }

    public int Count { get; private set; }

    /// <summary>All pieces, back to back; piece <c>i</c> spans <c>Offsets[i]..Offsets[i + 1]</c>.</summary>
    public byte[] Bytes => _bytes;

    /// <summary><see cref="Count"/> + 1 offsets into <see cref="Bytes"/>.</summary>
    public int[] Offsets => _offsets;

    public ReadOnlySpan<byte> Piece(int id) => _bytes.AsSpan(_offsets[id], _offsets[id + 1] - _offsets[id]);

    public double Score(int id) => _scores[id];

    /// <summary>The scores, indexed by id (longer than <see cref="Count"/> until <see cref="TrimExcess"/>).</summary>
    public double[] Scores => _scores;

    /// <summary>Shrinks the buffers to the pieces added.</summary>
    public void TrimExcess()
    {
        if (_bytes.Length != _byteCount)
        {
            Array.Resize(ref _bytes, _byteCount);
        }

        if (_scores.Length != Count)
        {
            Array.Resize(ref _scores, Count);
            Array.Resize(ref _offsets, Count + 1);
        }
    }

    /// <summary>Adds a piece given as well-formed UTF-8.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Add(ReadOnlySpan<byte> utf8, double score)
    {
        EnsurePieceCapacity();
        if (_byteCount + utf8.Length > _bytes.Length)
        {
            Array.Resize(ref _bytes, Math.Max(_bytes.Length * 2, _byteCount + utf8.Length));
        }

        utf8.CopyTo(_bytes.AsSpan(_byteCount));
        _byteCount += utf8.Length;
        _scores[Count] = score;
        _offsets[++Count] = _byteCount;
    }

    /// <summary>Adds a piece given as a string (from the JSON document); it is stored as UTF-8.</summary>
    public void Add(string piece, double score)
    {
        int max = Encoding.UTF8.GetMaxByteCount(piece.Length);
        if (max <= 256)
        {
            Span<byte> buffer = stackalloc byte[max];
            Add(buffer[..Encoding.UTF8.GetBytes(piece, buffer)], score);
        }
        else
        {
            Add(Encoding.UTF8.GetBytes(piece), score);
        }
    }

    private void EnsurePieceCapacity()
    {
        if (Count == _scores.Length)
        {
            Array.Resize(ref _scores, _scores.Length * 2);
            Array.Resize(ref _offsets, _scores.Length + 1);
        }
    }
}
