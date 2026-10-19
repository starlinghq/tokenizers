// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3),
// tokenizers/src/tokenizer/added_vocabulary.rs and serialization.rs (id assignment),
// Copyright The HuggingFace Inc. team, Apache License 2.0 (see THIRD-PARTY-NOTICES.md).
// Modified by StarlingHq: translated to C#; leftmost-longest matching uses a byte trie
// instead of an Aho-Corasick automaton (same matches); lstrip and rstrip are implemented;
// single_word is rejected at load time rather than implemented.

using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>An entry of the <c>added_tokens</c> list.</summary>
internal sealed record AddedToken(string Content, bool Special, bool Normalized, bool Lstrip, bool Rstrip);

/// <summary>Upstream <c>AddedVocabulary</c>: tokens matched in the text before the model runs.</summary>
internal sealed class AddedVocabulary
{
    private readonly Dictionary<string, int> _tokenToId = new(StringComparer.Ordinal);
    private readonly Dictionary<int, AddedToken> _idToToken = [];
    private readonly HashSet<string> _specialTokens = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _normalizedCache = [];
    private readonly ByteTrie? _nonNormalizedTrie;
    private readonly ByteTrie? _normalizedTrie;

    /// <summary>Adds the tokens in file order, assigning ids exactly as upstream <c>add_tokens</c>.</summary>
    public AddedVocabulary(IReadOnlyList<(int DeclaredId, AddedToken Token)> tokens, IModel model, INormalizer? normalizer)
    {
        int vocabSize = model.VocabularySize;
        int nextId = vocabSize;

        foreach ((int declaredId, AddedToken token) in tokens)
        {
            if (token.Content.Length == 0)
            {
                continue;
            }

            if (_tokenToId.TryGetValue(token.Content, out int existing) && _idToToken[existing] == token)
            {
                continue;
            }

            int newId;
            if (_tokenToId.TryGetValue(token.Content, out int known))
            {
                newId = known;
            }
            else if (model.TokenToId(token.Content) is int modelId)
            {
                newId = modelId;
            }
            else
            {
                newId = nextId++;
            }

            if (newId != declaredId)
            {
                throw new UnsupportedTokenizerComponentException(
                    "added_tokens",
                    $"token '{token.Content}' declares id {declaredId} but the reference assigns id {newId}.");
            }

            if (token.Normalized && normalizer is not null)
            {
                NormalizedString s = NormalizedString.FromString(token.Content);
                normalizer.Normalize(s);
                string normed = s.NormalizedText;
                if (!string.Equals(normed, token.Content, StringComparison.Ordinal))
                {
                    _normalizedCache[newId] = normed;
                }
            }

            _tokenToId[token.Content] = newId;
            if (token.Special)
            {
                _specialTokens.Add(token.Content);
            }

            _idToToken[newId] = token;
        }

        var normalizedPatterns = new List<(byte[] Pattern, int Id)>();
        var rawPatterns = new List<(byte[] Pattern, int Id)>();
        foreach ((int id, AddedToken token) in _idToToken)
        {
            if (token.Normalized)
            {
                string pattern = _normalizedCache.TryGetValue(id, out string? cached) ? cached : token.Content;
                normalizedPatterns.Add((Encoding.UTF8.GetBytes(pattern), id));
            }
            else
            {
                rawPatterns.Add((Encoding.UTF8.GetBytes(token.Content), id));
            }
        }

        _nonNormalizedTrie = rawPatterns.Count == 0 ? null : new ByteTrie(rawPatterns);
        _normalizedTrie = normalizedPatterns.Count == 0 ? null : new ByteTrie(normalizedPatterns);
    }

    public int Count => _tokenToId.Count;

    public IReadOnlyDictionary<string, int> TokenToIdMap => _tokenToId;

    public int? TokenToId(string token) => _tokenToId.TryGetValue(token, out int id) ? id : null;

    /// <summary>Upstream <c>simple_id_to_token</c>.</summary>
    public string? IdToToken(int id) =>
        _idToToken.TryGetValue(id, out AddedToken? token)
            ? (_normalizedCache.TryGetValue(id, out string? normed) ? normed : token.Content)
            : null;

    public bool IsSpecialToken(string token) => _specialTokens.Contains(token);

    /// <summary>Upstream <c>extract_and_normalize</c>.</summary>
    public PreTokenizedString ExtractAndNormalize(INormalizer? normalizer, byte[] sequence)
    {
        var pretokenized = new PreTokenizedString(sequence);
        pretokenized.SplitEach(s => SplitWithIndices(s, _nonNormalizedTrie));
        pretokenized.SplitEach(s =>
        {
            normalizer?.Normalize(s);
            return SplitWithIndices(s, _normalizedTrie);
        });
        return pretokenized;
    }

    private List<Split> SplitWithIndices(NormalizedString sentence, ByteTrie? trie)
    {
        var result = new List<Split>();
        foreach ((int? id, int start, int end) in FindMatches(sentence.Normalized, trie))
        {
            NormalizedString slice = sentence.SliceNormalized(start, end)
                ?? throw new InvalidOperationException("AddedVocabulary bad split");
            if (id is int tokenId)
            {
                string value = slice.NormalizedText;
                result.Add(new Split(slice, [new ModelToken(tokenId, value, 0, slice.Length)]));
            }
            else
            {
                result.Add(new Split(slice));
            }
        }

        return result;
    }

    private List<(int? Id, int Start, int End)> FindMatches(byte[] sentence, ByteTrie? trie)
    {
        if (sentence.Length == 0)
        {
            return [(null, 0, 0)];
        }

        if (trie is null)
        {
            return [(null, 0, sentence.Length)];
        }

        var splits = new List<(int? Id, int Start, int End)>();
        int startOffset = 0;
        int pos = 0;
        while (pos < sentence.Length)
        {
            if (trie.LongestMatch(sentence, pos) is not { } match)
            {
                pos++;
                continue;
            }

            int start = pos;
            int stop = pos + match.Length;
            AddedToken token = _idToToken[match.Id];

            if (token.Lstrip)
            {
                start = Math.Max(SpaceLeftmostAtEnd(sentence, start), startOffset);
            }

            if (token.Rstrip)
            {
                stop += SpaceRightmostAtStart(sentence, stop);
            }

            if (startOffset < start)
            {
                splits.Add((null, startOffset, start));
            }

            splits.Add((match.Id, start, stop));
            pos = stop;
            startOffset = stop;
        }

        if (startOffset != sentence.Length)
        {
            splits.Add((null, startOffset, sentence.Length));
        }

        return splits;
    }

    /// <summary>Upstream <c>space_leftmost_at_end</c>: index where the trailing whitespace run
    /// ending at <paramref name="end"/> starts (byte offset).</summary>
    private static int SpaceLeftmostAtEnd(byte[] sentence, int end)
    {
        int i = end;
        while (i > 0)
        {
            int start = i - 1;
            while (start > 0 && (sentence[start] & 0xC0) == 0x80)
            {
                start--;
            }

            if (!UnicodeData.IsWhiteSpace(Utf8Util.DecodeAt(sentence, start, out _)))
            {
                break;
            }

            i = start;
        }

        return i;
    }

    /// <summary>Upstream <c>space_rightmost_at_start</c>: length of the leading whitespace run
    /// at <paramref name="start"/> (byte count).</summary>
    private static int SpaceRightmostAtStart(byte[] sentence, int start)
    {
        int i = start;
        while (i < sentence.Length)
        {
            Rune r = Utf8Util.DecodeAt(sentence, i, out int len);
            if (!UnicodeData.IsWhiteSpace(r))
            {
                break;
            }

            i += len;
        }

        return i - start;
    }

    /// <summary>Byte trie giving the longest pattern that starts at a position.</summary>
    private sealed class ByteTrie
    {
        private readonly Node _root = new();

        public ByteTrie(IEnumerable<(byte[] Pattern, int Id)> patterns)
        {
            foreach ((byte[] pattern, int id) in patterns)
            {
                Node node = _root;
                foreach (byte b in pattern)
                {
                    node.Children ??= [];
                    if (!node.Children.TryGetValue(b, out Node? child))
                    {
                        child = new Node();
                        node.Children[b] = child;
                    }

                    node = child;
                }

                node.Id = id;
            }
        }

        public (int Id, int Length)? LongestMatch(byte[] text, int start)
        {
            Node node = _root;
            (int Id, int Length)? best = null;
            for (int i = start; i < text.Length; i++)
            {
                if (node.Children is null || !node.Children.TryGetValue(text[i], out Node? child))
                {
                    break;
                }

                node = child;
                if (node.Id is int id)
                {
                    best = (id, i + 1 - start);
                }
            }

            return best;
        }

        private sealed class Node
        {
            public Dictionary<byte, Node>? Children { get; set; }

            public int? Id { get; set; }
        }
    }
}
