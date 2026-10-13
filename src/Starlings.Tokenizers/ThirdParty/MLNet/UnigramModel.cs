// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
//
// Adopted from dotnet/machinelearning at 5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d,
// src/Microsoft.ML.Tokenizers/Model/SentencePieceUnigramModel.cs (MIT; the lattice follows
// google/sentencepiece unigram_model.cc, Apache License 2.0). See THIRD-PARTY-NOTICES.md.
//
// Modified by StarlingHq (see PROVENANCE.md). Only the best-path search (Encode) and the
// double-array trie lookup are kept; the class no longer derives from the ML.NET tokenizer base
// type. Behavior is aligned with huggingface/tokenizers v0.23.2 models/unigram/model.rs
// (Copyright The HuggingFace Inc. team, Apache License 2.0), the fidelity reference:
//   - scores are double, as upstream (ML.NET used float);
//   - every vocabulary piece is in the trie, special pieces included, and there are no
//     Unused or UserDefined piece types;
//   - consecutive unknown pieces are fused into one unknown token whose value is the
//     original text (ML.NET emitted one unknown token per character);
//   - token ids come from a piece-to-id map where a duplicated piece keeps its last id; the map
//     is the vocabulary trie itself (exact UTF-8 match), and pieces are kept as UTF-8 bytes with
//     strings created on demand (StarlingHq load-time change, see PROVENANCE.md L11);
//   - implements the interpreter's IModel stage interface (PROVENANCE.md L13).

using System.Text;
using Starlings.Tokenizers.Pipeline;

namespace Starlings.Tokenizers.ThirdParty.MLNet
{
    internal sealed class UnigramModel : IModel
    {
        private const double UnkPenalty = 10.0;

        private readonly UnigramVocabulary _vocab;
        private readonly double[] _scores;
        private readonly string?[] _pieces;
        private readonly DoubleArrayTrie _trie;
        private readonly double _minScore;
        private readonly int? _unkId;

        public UnigramModel(UnigramVocabulary vocab, int? unkId)
        {
            int count = vocab.Count;
            if (unkId is int u && (count == 0 || u < 0 || u >= count))
            {
                throw new TokenizerFormatException("model Unigram: unk_id is outside the vocabulary.");
            }

            vocab.TrimExcess();
            _vocab = vocab;
            _scores = vocab.Scores;
            _pieces = new string?[count];
            _unkId = unkId;
            _minScore = double.PositiveInfinity;
            for (int id = 0; id < count; id++)
            {
                ReadOnlySpan<byte> piece = vocab.Piece(id);
                if (piece.IsEmpty || piece.Contains((byte)0))
                {
                    throw new UnsupportedTokenizerComponentException(
                        "model.type=Unigram", $"vocabulary piece {id} is empty or contains U+0000.");
                }

                double score = _scores[id];
                if (score < _minScore)
                {
                    _minScore = score;
                }
            }

            _trie = VocabularyTrie.Build(vocab);
        }

        public int VocabularySize => _pieces.Length;

        /// <summary>
        /// Upstream <c>token_to_ids</c> lookup. The trie holds exactly the map's keys (UTF-8) and
        /// values (a duplicated piece keeps its last id), so an exact trie match is the map lookup.
        /// </summary>
        public int? TokenToId(string token)
        {
            // Pieces are non-empty, well-formed Unicode: an empty token or one with unpaired
            // surrogates cannot equal a piece.
            if (token.Length == 0)
            {
                return null;
            }

            byte[]? rented = null;
            int max = token.Length * 3;
            Span<byte> buffer = max <= 512 ? stackalloc byte[max] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(max));
            try
            {
                if (System.Text.Unicode.Utf8.FromUtf16(token, buffer, out _, out int written, replaceInvalidSequences: false)
                    != System.Buffers.OperationStatus.Done)
                {
                    return null;
                }

                int id = _trie.ExactMatch(buffer[..written]);
                return id >= 0 ? id : null;
            }
            finally
            {
                if (rented is not null)
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(rented);
                }
            }
        }

        public double Score(int id) => _scores[id];

        internal DoubleArrayTrie Trie => _trie;

        public string? IdToToken(int id) =>
            (uint)id < (uint)_pieces.Length ? _pieces[id] ??= Encoding.UTF8.GetString(_vocab.Piece(id)) : null;

        /// <summary>Upstream <c>Model::tokenize</c>: byte offsets into <paramref name="sentence"/>.</summary>
        public List<ModelToken> Tokenize(ReadOnlySpan<byte> sentence)
        {
            var tokens = new List<ModelToken>();
            if (sentence.IsEmpty)
            {
                return tokens;
            }

            var bestPathEndsAt = new BestPathNode[sentence.Length + 1];
            Encode(sentence, bestPathEndsAt);

            // Backtrack, fusing consecutive unknown pieces (upstream fuse_unk = true).
            var pieces = new List<(int Start, int End)>();
            int endsAt = sentence.Length;
            int fusedEnd = -1;
            int fusedStart = -1;
            while (endsAt > 0)
            {
                ref BestPathNode node = ref bestPathEndsAt[endsAt];
                if (_unkId is int unk && node.Id == unk)
                {
                    if (fusedEnd < 0)
                    {
                        fusedEnd = endsAt;
                    }

                    fusedStart = node.StartsAt;
                }
                else
                {
                    if (fusedEnd >= 0)
                    {
                        pieces.Add((fusedStart, fusedEnd));
                        fusedEnd = -1;
                    }

                    pieces.Add((node.StartsAt, endsAt));
                }

                endsAt = node.StartsAt;
            }

            if (fusedEnd >= 0)
            {
                pieces.Add((fusedStart, fusedEnd));
            }

            pieces.Reverse();
            int offset = 0;
            foreach ((int start, int end) in pieces)
            {
                string value = Encoding.UTF8.GetString(sentence[start..end]);
                int len = end - start;
                int found = _trie.ExactMatch(sentence[start..end]);
                int id = found >= 0
                    ? found
                    : _unkId ?? throw new InvalidOperationException("Encountered an unknown token but `unk_id` is missing");
                tokens.Add(new ModelToken(id, value, offset, offset + len));
                offset += len;
            }

            return tokens;
        }

        private struct BestPathNode
        {
            // The vocab id.
            public int Id;

            // The total score of the best path ending at this node.
            public double BestPathScore;

            // The starting position (in utf-8) of this node, or -1 when unset.
            public int StartsAt;
        }

        private void Encode(ReadOnlySpan<byte> normalized, Span<BestPathNode> bestPathEndsAt)
        {
            int size = normalized.Length;
            double unkScore = _minScore - UnkPenalty;

            // The ends are exclusive.
            for (int i = 0; i < size + 1; i++)
            {
                bestPathEndsAt[i] = new BestPathNode { Id = 0, BestPathScore = 0, StartsAt = -1 };
            }

            // Generate lattice on-the-fly (not stored) and update best_path_ends_at.
            int startsAt = 0;

            while (startsAt < size)
            {
                int nodePos = 0;
                int keyPos = startsAt;
                double bestPathScoreTillHere = bestPathEndsAt[startsAt].BestPathScore;
                bool hasSingleNode = false;
                int mbLen = Utf8Util.SequenceLength(normalized, startsAt);
                while (keyPos < size)
                {
                    int ret = _trie.Traverse(normalized, ref nodePos, ref keyPos, keyPos + 1);
                    if (ret == -2)
                    {
                        break;
                    }

                    if (ret >= 0)
                    {
                        // Update the best path node.
                        ref BestPathNode targetNode = ref bestPathEndsAt[keyPos];
                        int length = keyPos - startsAt;

                        double score = _scores[ret];
                        double candidateBestPathScore = score + bestPathScoreTillHere;

                        if (targetNode.StartsAt == -1 || candidateBestPathScore > targetNode.BestPathScore)
                        {
                            targetNode.BestPathScore = candidateBestPathScore;
                            targetNode.StartsAt = startsAt;
                            targetNode.Id = ret;
                        }

                        if (!hasSingleNode && length == mbLen)
                        {
                            hasSingleNode = true;
                        }
                    }
                }

                if (!hasSingleNode)
                {
                    ref BestPathNode targetNode = ref bestPathEndsAt[startsAt + mbLen];
                    double candidateBestPathScore = unkScore + bestPathScoreTillHere;

                    if (targetNode.StartsAt == -1 || candidateBestPathScore > targetNode.BestPathScore)
                    {
                        targetNode.BestPathScore = candidateBestPathScore;
                        targetNode.StartsAt = startsAt;
                        targetNode.Id = _unkId ?? throw new InvalidOperationException("Encountered an unknown token but `unk_id` is missing");
                    }
                }

                // Move by one unicode character.
                startsAt += mbLen;
            }
        }
    }
}
