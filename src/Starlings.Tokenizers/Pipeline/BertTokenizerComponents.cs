// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Portions ported from huggingface/tokenizers v0.23.2 (Apache-2.0): BERT normalizer and
// pre-tokenizer, WordPiece model, and WordPiece decoder. Modified by StarlingHq as a bounded
// C# implementation for the pinned encoder profiles. Unicode decomposition/alignment reuses the
// existing Unicode 9 tables in this assembly. See PROVENANCE.md and THIRD-PARTY-NOTICES.md.

using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>WordPiece vocabulary model with greedy longest-match-first segmentation.</summary>
internal sealed class WordPieceModel : IModel
{
    private readonly BpeVocabulary _vocabulary;
    private readonly int _unknownId;
    private readonly string _unknownToken;
    private readonly byte[] _prefix;
    private readonly int _maxInputCharsPerWord;

    public WordPieceModel(
        BpeVocabulary vocabulary,
        string unknownToken,
        string continuingSubwordPrefix,
        int maxInputCharsPerWord)
    {
        _vocabulary = vocabulary;
        _unknownToken = unknownToken;
        _unknownId = vocabulary.Find(unknownToken);
        if (_unknownId < 0)
        {
            throw new UnsupportedTokenizerComponentException(
                "model.unk_token", $"unknown token '{unknownToken}' is absent from the vocabulary.");
        }

        _prefix = Encoding.UTF8.GetBytes(continuingSubwordPrefix);
        _maxInputCharsPerWord = maxInputCharsPerWord;
    }

    public int VocabularySize => _vocabulary.Count;

    public int? TokenToId(string token)
    {
        int id = _vocabulary.Find(token);
        return id >= 0 ? id : null;
    }

    public string? IdToToken(int id) => _vocabulary.Token(id);

    public List<ModelToken> Tokenize(ReadOnlySpan<byte> sequence)
    {
        if (sequence.IsEmpty)
        {
            return [];
        }

        var runeStarts = new List<int>();
        int offset = 0;
        while (offset < sequence.Length)
        {
            runeStarts.Add(offset);
            offset += Utf8Util.SequenceLength(sequence, offset);
        }

        int charCount = runeStarts.Count;
        if (charCount > _maxInputCharsPerWord)
        {
            return [Unknown(sequence.Length)];
        }

        var tokens = new List<ModelToken>();
        int wordStart = 0;
        while (wordStart < charCount)
        {
            int start = wordStart;
            var wordTokens = new List<ModelToken>();
            while (start < charCount)
            {
                int matchEnd = -1;
                int matchId = -1;
                for (int end = charCount; end > start; end--)
                {
                    int byteStart = runeStarts[start];
                    int byteEnd = end == charCount ? sequence.Length : runeStarts[end];
                    ReadOnlySpan<byte> part = sequence[byteStart..byteEnd];
                    int id = start == wordStart ? _vocabulary.Find(part) : _vocabulary.FindConcatenation(_prefix, part);
                    if (id >= 0)
                    {
                        matchEnd = end;
                        matchId = id;
                        break;
                    }
                }

                if (matchEnd < 0)
                {
                    return [Unknown(sequence.Length)];
                }

                int tokenStart = runeStarts[start];
                int tokenEnd = matchEnd == charCount ? sequence.Length : runeStarts[matchEnd];
                wordTokens.Add(new ModelToken(
                    matchId,
                    _vocabulary.Token(matchId)!,
                    tokenStart,
                    tokenEnd));
                start = matchEnd;
            }

            tokens.AddRange(wordTokens);
            wordStart = charCount;
        }

        return tokens;
    }

    private ModelToken Unknown(int length) => new(_unknownId, _unknownToken, 0, length);
}

/// <summary>BERT BasicTokenizer normalizer with upstream-compatible ordering and offsets.</summary>
internal sealed class BertNormalizer(bool cleanText, bool handleChineseChars, bool? stripAccents, bool lowercase) : INormalizer
{
    public void Normalize(NormalizedString normalized)
    {
        if (cleanText)
        {
            normalized.Replace(BertCharacterPattern.CleanTextControls.FindMatches(normalized.Normalized), string.Empty);
            normalized.Replace(BertCharacterPattern.WhiteSpace.FindMatches(normalized.Normalized), " ");
        }

        if (handleChineseChars && !normalized.IsEmpty)
        {
            var changes = new List<CharChange>();
            foreach (Rune rune in Utf8Util.DecodeRunes(normalized.Normalized))
            {
                if (BertUnicode.IsChinese(rune))
                {
                    changes.Add(new CharChange(new Rune(' '), 0));
                    changes.Add(new CharChange(rune, 1));
                    changes.Add(new CharChange(new Rune(' '), 1));
                }
                else
                {
                    changes.Add(new CharChange(rune, 0));
                }
            }

            normalized.Transform(changes, 0);
        }

        if (stripAccents ?? lowercase)
        {
            normalized.Transform(Unicode9Normalization.Nfd(Utf8Util.DecodeRunes(normalized.Normalized)), 0);
            normalized.Replace(BertCharacterPattern.Marks.FindMatches(normalized.Normalized), string.Empty);
        }

        if (lowercase && !normalized.IsEmpty)
        {
            var changes = new List<CharChange>();
            foreach (Rune rune in Utf8Util.DecodeRunes(normalized.Normalized))
            {
                string lower = rune.ToString().ToLowerInvariant();
                bool first = true;
                foreach (Rune mapped in lower.EnumerateRunes())
                {
                    changes.Add(new CharChange(mapped, first ? 0 : 1));
                    first = false;
                }
            }

            normalized.Transform(changes, 0);
        }
    }
}

/// <summary>BERT pre-tokenizer: remove Unicode whitespace and isolate punctuation.</summary>
internal sealed class BertPreTokenizer : IPreTokenizer
{
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.SplitEach(normalized =>
            normalized.Split(BertCharacterPattern.WhiteSpace.FindMatches(normalized.Normalized), SplitDelimiterBehavior.Removed)
                .Select(part => new Split(part)));
        pretokenized.SplitEach(normalized =>
            normalized.Split(BertCharacterPattern.Punctuation.FindMatches(normalized.Normalized), SplitDelimiterBehavior.Isolated)
                .Select(part => new Split(part)));
    }
}

/// <summary>WordPiece decoder: remove continuation prefixes, join pieces, and apply reference cleanup.</summary>
internal sealed class WordPieceDecoder(string prefix, bool cleanup) : IDecoder
{
    private static readonly (string From, string To)[] _cleanup =
    [
        (" .", "."),
        (" ?", "?"),
        (" !", "!"),
        (" ,", ","),
        (" ' ", "'"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ];

    public List<string> DecodeChain(List<string> tokens)
    {
        if (tokens.Count == 0)
        {
            return [];
        }

        var builder = new StringBuilder();
        bool first = true;
        foreach (string token in tokens)
        {
            // Upstream uses starts_with(prefix), so an empty prefix matches every token and pieces are
            // concatenated without spaces (tokenizers 0.23.2: ["ab", "cd"] -> "abcd").
            bool continuing = !first && token.StartsWith(prefix, StringComparison.Ordinal);
            string fragment;
            if (first)
            {
                // Upstream leaves the first token untouched, even when it begins with the
                // continuation prefix.
                fragment = token;
            }
            else if (continuing)
            {
                fragment = token[prefix.Length..];
            }
            else
            {
                fragment = " " + token;
            }

            if (cleanup)
            {
                foreach ((string from, string to) in _cleanup)
                {
                    fragment = fragment.Replace(from, to, StringComparison.Ordinal);
                }
            }

            builder.Append(fragment);
            first = false;
        }

        return [builder.ToString()];
    }
}

/// <summary>Unicode predicates and match spans used by BERT normalizer and pre-tokenizer.</summary>
internal static class BertCharacterPattern
{
    public static readonly IPattern CleanTextControls = new PredicatePattern(BertUnicode.IsCleanTextRemoved);

    public static readonly IPattern WhiteSpace = new PredicatePattern(UnicodeData.IsWhiteSpace, isolateEachMatch: true);

    public static readonly IPattern Marks = new PredicatePattern(r => BertUnicode.IsMark(r));

    public static readonly IPattern Punctuation = new PredicatePattern(r => BertUnicode.IsPunctuation(r), isolateEachMatch: true);

    private sealed class PredicatePattern(Func<Rune, bool> predicate, bool isolateEachMatch = false) : IPattern
    {
        public List<PatternMatch> FindMatches(ReadOnlySpan<byte> inside)
        {
            if (inside.IsEmpty)
            {
                return [new PatternMatch(0, 0, false)];
            }

            var matches = new List<PatternMatch>();
            int runStart = 0;
            int offset = 0;
            bool? previous = null;
            while (offset < inside.Length)
            {
                Rune rune = Utf8Util.DecodeAt(inside, offset, out int length);
                bool selected = predicate(rune);
                if (selected && isolateEachMatch)
                {
                    if (previous == false)
                    {
                        matches.Add(new PatternMatch(runStart, offset, false));
                    }

                    matches.Add(new PatternMatch(offset, offset + length, true));
                    runStart = offset + length;
                    previous = false;
                    offset += length;
                    continue;
                }

                if (previous is bool old && old != selected)
                {
                    matches.Add(new PatternMatch(runStart, offset, old));
                    runStart = offset;
                }

                previous = selected;
                offset += length;
            }

            if (runStart < inside.Length)
            {
                matches.Add(new PatternMatch(runStart, inside.Length, previous == true));
            }
            return matches;
        }
    }
}

internal static class BertUnicode
{
    // Nonspacing marks (Mn) removed by BertNormalizer strip_accents in huggingface/tokenizers 0.23.2, which
    // classifies with the Rust unicode_categories tables. Those tables predate the .NET runtime's Unicode
    // data, so Rune.GetUnicodeCategory would also strip newer marks (for example U+0898, U+11F00, U+1CF00,
    // U+1E4EC) that the reference keeps, and would keep U+1734 and U+1171E, which the reference strips.
    // Ranges were enumerated exhaustively over U+0000..U+10FFFF from the 0.23.2 reference. The predicate
    // is applied only after canonical decomposition, so code points that decompose (for example U+0340,
    // U+0341, U+0343, U+0344) never reach it and are not listed.
    private static readonly (int First, int Last)[] _marks =
    [
        (0x300, 0x33F),
        (0x342, 0x342),
        (0x345, 0x36F),
        (0x483, 0x487),
        (0x591, 0x5BD),
        (0x5BF, 0x5BF),
        (0x5C1, 0x5C2),
        (0x5C4, 0x5C5),
        (0x5C7, 0x5C7),
        (0x610, 0x61A),
        (0x64B, 0x65F),
        (0x670, 0x670),
        (0x6D6, 0x6DC),
        (0x6DF, 0x6E4),
        (0x6E7, 0x6E8),
        (0x6EA, 0x6ED),
        (0x711, 0x711),
        (0x730, 0x74A),
        (0x7A6, 0x7B0),
        (0x7EB, 0x7F3),
        (0x816, 0x819),
        (0x81B, 0x823),
        (0x825, 0x827),
        (0x829, 0x82D),
        (0x859, 0x85B),
        (0x8E3, 0x902),
        (0x93A, 0x93A),
        (0x93C, 0x93C),
        (0x941, 0x948),
        (0x94D, 0x94D),
        (0x951, 0x957),
        (0x962, 0x963),
        (0x981, 0x981),
        (0x9BC, 0x9BC),
        (0x9C1, 0x9C4),
        (0x9CD, 0x9CD),
        (0x9E2, 0x9E3),
        (0xA01, 0xA02),
        (0xA3C, 0xA3C),
        (0xA41, 0xA42),
        (0xA47, 0xA48),
        (0xA4B, 0xA4D),
        (0xA51, 0xA51),
        (0xA70, 0xA71),
        (0xA75, 0xA75),
        (0xA81, 0xA82),
        (0xABC, 0xABC),
        (0xAC1, 0xAC5),
        (0xAC7, 0xAC8),
        (0xACD, 0xACD),
        (0xAE2, 0xAE3),
        (0xB01, 0xB01),
        (0xB3C, 0xB3C),
        (0xB3F, 0xB3F),
        (0xB41, 0xB44),
        (0xB4D, 0xB4D),
        (0xB56, 0xB56),
        (0xB62, 0xB63),
        (0xB82, 0xB82),
        (0xBC0, 0xBC0),
        (0xBCD, 0xBCD),
        (0xC00, 0xC00),
        (0xC3E, 0xC40),
        (0xC46, 0xC47),
        (0xC4A, 0xC4D),
        (0xC55, 0xC56),
        (0xC62, 0xC63),
        (0xC81, 0xC81),
        (0xCBC, 0xCBC),
        (0xCBF, 0xCBF),
        (0xCC6, 0xCC6),
        (0xCCC, 0xCCD),
        (0xCE2, 0xCE3),
        (0xD01, 0xD01),
        (0xD41, 0xD44),
        (0xD4D, 0xD4D),
        (0xD62, 0xD63),
        (0xDCA, 0xDCA),
        (0xDD2, 0xDD4),
        (0xDD6, 0xDD6),
        (0xE31, 0xE31),
        (0xE34, 0xE3A),
        (0xE47, 0xE4E),
        (0xEB1, 0xEB1),
        (0xEB4, 0xEB9),
        (0xEBB, 0xEBC),
        (0xEC8, 0xECD),
        (0xF18, 0xF19),
        (0xF35, 0xF35),
        (0xF37, 0xF37),
        (0xF39, 0xF39),
        (0xF71, 0xF72),
        (0xF74, 0xF74),
        (0xF77, 0xF77),
        (0xF79, 0xF7E),
        (0xF80, 0xF80),
        (0xF82, 0xF84),
        (0xF86, 0xF87),
        (0xF8D, 0xF92),
        (0xF94, 0xF97),
        (0xF99, 0xF9C),
        (0xF9E, 0xFA1),
        (0xFA3, 0xFA6),
        (0xFA8, 0xFAB),
        (0xFAD, 0xFB8),
        (0xFBA, 0xFBC),
        (0xFC6, 0xFC6),
        (0x102D, 0x1030),
        (0x1032, 0x1037),
        (0x1039, 0x103A),
        (0x103D, 0x103E),
        (0x1058, 0x1059),
        (0x105E, 0x1060),
        (0x1071, 0x1074),
        (0x1082, 0x1082),
        (0x1085, 0x1086),
        (0x108D, 0x108D),
        (0x109D, 0x109D),
        (0x135D, 0x135F),
        (0x1712, 0x1714),
        (0x1732, 0x1734),
        (0x1752, 0x1753),
        (0x1772, 0x1773),
        (0x17B4, 0x17B5),
        (0x17B7, 0x17BD),
        (0x17C6, 0x17C6),
        (0x17C9, 0x17D3),
        (0x17DD, 0x17DD),
        (0x180B, 0x180D),
        (0x18A9, 0x18A9),
        (0x1920, 0x1922),
        (0x1927, 0x1928),
        (0x1932, 0x1932),
        (0x1939, 0x193B),
        (0x1A17, 0x1A18),
        (0x1A1B, 0x1A1B),
        (0x1A56, 0x1A56),
        (0x1A58, 0x1A5E),
        (0x1A60, 0x1A60),
        (0x1A62, 0x1A62),
        (0x1A65, 0x1A6C),
        (0x1A73, 0x1A7C),
        (0x1A7F, 0x1A7F),
        (0x1AB0, 0x1ABD),
        (0x1B00, 0x1B03),
        (0x1B34, 0x1B34),
        (0x1B36, 0x1B3A),
        (0x1B3C, 0x1B3C),
        (0x1B42, 0x1B42),
        (0x1B6B, 0x1B73),
        (0x1B80, 0x1B81),
        (0x1BA2, 0x1BA5),
        (0x1BA8, 0x1BA9),
        (0x1BAB, 0x1BAD),
        (0x1BE6, 0x1BE6),
        (0x1BE8, 0x1BE9),
        (0x1BED, 0x1BED),
        (0x1BEF, 0x1BF1),
        (0x1C2C, 0x1C33),
        (0x1C36, 0x1C37),
        (0x1CD0, 0x1CD2),
        (0x1CD4, 0x1CE0),
        (0x1CE2, 0x1CE8),
        (0x1CED, 0x1CED),
        (0x1CF4, 0x1CF4),
        (0x1CF8, 0x1CF9),
        (0x1DC0, 0x1DF5),
        (0x1DFC, 0x1DFF),
        (0x20D0, 0x20DC),
        (0x20E1, 0x20E1),
        (0x20E5, 0x20F0),
        (0x2CEF, 0x2CF1),
        (0x2D7F, 0x2D7F),
        (0x2DE0, 0x2DFF),
        (0x302A, 0x302D),
        (0x3099, 0x309A),
        (0xA66F, 0xA66F),
        (0xA674, 0xA67D),
        (0xA69E, 0xA69F),
        (0xA6F0, 0xA6F1),
        (0xA802, 0xA802),
        (0xA806, 0xA806),
        (0xA80B, 0xA80B),
        (0xA825, 0xA826),
        (0xA8C4, 0xA8C4),
        (0xA8E0, 0xA8F1),
        (0xA926, 0xA92D),
        (0xA947, 0xA951),
        (0xA980, 0xA982),
        (0xA9B3, 0xA9B3),
        (0xA9B6, 0xA9B9),
        (0xA9BC, 0xA9BC),
        (0xA9E5, 0xA9E5),
        (0xAA29, 0xAA2E),
        (0xAA31, 0xAA32),
        (0xAA35, 0xAA36),
        (0xAA43, 0xAA43),
        (0xAA4C, 0xAA4C),
        (0xAA7C, 0xAA7C),
        (0xAAB0, 0xAAB0),
        (0xAAB2, 0xAAB4),
        (0xAAB7, 0xAAB8),
        (0xAABE, 0xAABF),
        (0xAAC1, 0xAAC1),
        (0xAAEC, 0xAAED),
        (0xAAF6, 0xAAF6),
        (0xABE5, 0xABE5),
        (0xABE8, 0xABE8),
        (0xABED, 0xABED),
        (0xFB1E, 0xFB1E),
        (0xFE00, 0xFE0F),
        (0xFE20, 0xFE2F),
        (0x101FD, 0x101FD),
        (0x102E0, 0x102E0),
        (0x10376, 0x1037A),
        (0x10A01, 0x10A03),
        (0x10A05, 0x10A06),
        (0x10A0C, 0x10A0F),
        (0x10A38, 0x10A3A),
        (0x10A3F, 0x10A3F),
        (0x10AE5, 0x10AE6),
        (0x11001, 0x11001),
        (0x11038, 0x11046),
        (0x1107F, 0x11081),
        (0x110B3, 0x110B6),
        (0x110B9, 0x110BA),
        (0x11100, 0x11102),
        (0x11127, 0x1112B),
        (0x1112D, 0x1112D),
        (0x11130, 0x11134),
        (0x11173, 0x11173),
        (0x11180, 0x11181),
        (0x111B6, 0x111BE),
        (0x111CA, 0x111CC),
        (0x1122F, 0x11231),
        (0x11234, 0x11234),
        (0x11236, 0x11237),
        (0x112DF, 0x112DF),
        (0x112E3, 0x112EA),
        (0x11300, 0x11301),
        (0x1133C, 0x1133C),
        (0x11340, 0x11340),
        (0x11366, 0x1136C),
        (0x11370, 0x11374),
        (0x114B3, 0x114B8),
        (0x114BA, 0x114BA),
        (0x114BF, 0x114C0),
        (0x114C2, 0x114C3),
        (0x115B2, 0x115B5),
        (0x115BC, 0x115BD),
        (0x115BF, 0x115C0),
        (0x115DC, 0x115DD),
        (0x11633, 0x1163A),
        (0x1163D, 0x1163D),
        (0x1163F, 0x11640),
        (0x116AB, 0x116AB),
        (0x116AD, 0x116AD),
        (0x116B0, 0x116B5),
        (0x116B7, 0x116B7),
        (0x1171D, 0x1171F),
        (0x11722, 0x11725),
        (0x11727, 0x1172B),
        (0x16AF0, 0x16AF4),
        (0x16B30, 0x16B36),
        (0x16F8F, 0x16F92),
        (0x1BC9D, 0x1BC9E),
        (0x1D167, 0x1D169),
        (0x1D17B, 0x1D182),
        (0x1D185, 0x1D18B),
        (0x1D1AA, 0x1D1AD),
        (0x1D242, 0x1D244),
        (0x1DA00, 0x1DA36),
        (0x1DA3B, 0x1DA6C),
        (0x1DA75, 0x1DA75),
        (0x1DA84, 0x1DA84),
        (0x1DA9B, 0x1DA9F),
        (0x1DAA1, 0x1DAAF),
        (0x1E8D0, 0x1E8D6),
        (0xE0100, 0xE01EF),
    ];

    // Non-ASCII punctuation (Pc/Pd/Ps/Pe/Pi/Pf/Po) isolated by BertPreTokenizer in huggingface/tokenizers
    // 0.23.2 (upstream is_punctuation: ASCII punctuation or the Rust unicode_categories punctuation
    // tables). The .NET runtime's newer data would also isolate later additions (for example U+061D,
    // U+2E43..U+2E5D) and would not isolate U+166D or U+111C9, which the reference does. Ranges were
    // enumerated exhaustively over U+0000..U+10FFFF from the 0.23.2 reference, including ASCII.
    private static readonly (int First, int Last)[] _punctuation =
    [
        (0x21, 0x2F),
        (0x3A, 0x40),
        (0x5B, 0x60),
        (0x7B, 0x7E),
        (0xA1, 0xA1),
        (0xA7, 0xA7),
        (0xAB, 0xAB),
        (0xB6, 0xB7),
        (0xBB, 0xBB),
        (0xBF, 0xBF),
        (0x37E, 0x37E),
        (0x387, 0x387),
        (0x55A, 0x55F),
        (0x589, 0x58A),
        (0x5BE, 0x5BE),
        (0x5C0, 0x5C0),
        (0x5C3, 0x5C3),
        (0x5C6, 0x5C6),
        (0x5F3, 0x5F4),
        (0x609, 0x60A),
        (0x60C, 0x60D),
        (0x61B, 0x61B),
        (0x61E, 0x61F),
        (0x66A, 0x66D),
        (0x6D4, 0x6D4),
        (0x700, 0x70D),
        (0x7F7, 0x7F9),
        (0x830, 0x83E),
        (0x85E, 0x85E),
        (0x964, 0x965),
        (0x970, 0x970),
        (0xAF0, 0xAF0),
        (0xDF4, 0xDF4),
        (0xE4F, 0xE4F),
        (0xE5A, 0xE5B),
        (0xF04, 0xF12),
        (0xF14, 0xF14),
        (0xF3A, 0xF3D),
        (0xF85, 0xF85),
        (0xFD0, 0xFD4),
        (0xFD9, 0xFDA),
        (0x104A, 0x104F),
        (0x10FB, 0x10FB),
        (0x1360, 0x1368),
        (0x1400, 0x1400),
        (0x166D, 0x166E),
        (0x169B, 0x169C),
        (0x16EB, 0x16ED),
        (0x1735, 0x1736),
        (0x17D4, 0x17D6),
        (0x17D8, 0x17DA),
        (0x1800, 0x180A),
        (0x1944, 0x1945),
        (0x1A1E, 0x1A1F),
        (0x1AA0, 0x1AA6),
        (0x1AA8, 0x1AAD),
        (0x1B5A, 0x1B60),
        (0x1BFC, 0x1BFF),
        (0x1C3B, 0x1C3F),
        (0x1C7E, 0x1C7F),
        (0x1CC0, 0x1CC7),
        (0x1CD3, 0x1CD3),
        (0x2010, 0x2027),
        (0x2030, 0x2043),
        (0x2045, 0x2051),
        (0x2053, 0x205E),
        (0x207D, 0x207E),
        (0x208D, 0x208E),
        (0x2308, 0x230B),
        (0x2329, 0x232A),
        (0x2768, 0x2775),
        (0x27C5, 0x27C6),
        (0x27E6, 0x27EF),
        (0x2983, 0x2998),
        (0x29D8, 0x29DB),
        (0x29FC, 0x29FD),
        (0x2CF9, 0x2CFC),
        (0x2CFE, 0x2CFF),
        (0x2D70, 0x2D70),
        (0x2E00, 0x2E2E),
        (0x2E30, 0x2E42),
        (0x3001, 0x3003),
        (0x3008, 0x3011),
        (0x3014, 0x301F),
        (0x3030, 0x3030),
        (0x303D, 0x303D),
        (0x30A0, 0x30A0),
        (0x30FB, 0x30FB),
        (0xA4FE, 0xA4FF),
        (0xA60D, 0xA60F),
        (0xA673, 0xA673),
        (0xA67E, 0xA67E),
        (0xA6F2, 0xA6F7),
        (0xA874, 0xA877),
        (0xA8CE, 0xA8CF),
        (0xA8F8, 0xA8FA),
        (0xA8FC, 0xA8FC),
        (0xA92E, 0xA92F),
        (0xA95F, 0xA95F),
        (0xA9C1, 0xA9CD),
        (0xA9DE, 0xA9DF),
        (0xAA5C, 0xAA5F),
        (0xAADE, 0xAADF),
        (0xAAF0, 0xAAF1),
        (0xABEB, 0xABEB),
        (0xFD3E, 0xFD3F),
        (0xFE10, 0xFE19),
        (0xFE30, 0xFE52),
        (0xFE54, 0xFE61),
        (0xFE63, 0xFE63),
        (0xFE68, 0xFE68),
        (0xFE6A, 0xFE6B),
        (0xFF01, 0xFF03),
        (0xFF05, 0xFF0A),
        (0xFF0C, 0xFF0F),
        (0xFF1A, 0xFF1B),
        (0xFF1F, 0xFF20),
        (0xFF3B, 0xFF3D),
        (0xFF3F, 0xFF3F),
        (0xFF5B, 0xFF5B),
        (0xFF5D, 0xFF5D),
        (0xFF5F, 0xFF65),
        (0x10100, 0x10102),
        (0x1039F, 0x1039F),
        (0x103D0, 0x103D0),
        (0x1056F, 0x1056F),
        (0x10857, 0x10857),
        (0x1091F, 0x1091F),
        (0x1093F, 0x1093F),
        (0x10A50, 0x10A58),
        (0x10A7F, 0x10A7F),
        (0x10AF0, 0x10AF6),
        (0x10B39, 0x10B3F),
        (0x10B99, 0x10B9C),
        (0x11047, 0x1104D),
        (0x110BB, 0x110BC),
        (0x110BE, 0x110C1),
        (0x11140, 0x11143),
        (0x11174, 0x11175),
        (0x111C5, 0x111C9),
        (0x111CD, 0x111CD),
        (0x111DB, 0x111DB),
        (0x111DD, 0x111DF),
        (0x11238, 0x1123D),
        (0x112A9, 0x112A9),
        (0x114C6, 0x114C6),
        (0x115C1, 0x115D7),
        (0x11641, 0x11643),
        (0x1173C, 0x1173E),
        (0x12470, 0x12474),
        (0x16A6E, 0x16A6F),
        (0x16AF5, 0x16AF5),
        (0x16B37, 0x16B3B),
        (0x16B44, 0x16B44),
        (0x1BC9F, 0x1BC9F),
        (0x1DA87, 0x1DA8B),
    ];

    public static bool IsMark(Rune rune) => InRanges(_marks, rune.Value);

    public static bool IsPunctuation(Rune rune) => InRanges(_punctuation, rune.Value);

    // Code points removed by BertNormalizer clean_text in huggingface/tokenizers 0.23.2: NUL, U+FFFD, and
    // upstream is_control (Cc/Cf/Co from the Rust unicode_categories tables, excluding \t \n \r). Those
    // tables predate the .NET runtime's Unicode data, so Rune.GetUnicodeCategory would also remove newer
    // Cf characters (for example U+0890, U+110CD, U+13430) that the reference keeps; Cn and noncharacters
    // are kept. Ranges were enumerated exhaustively over U+0000..U+10FFFF from the 0.23.2 reference.
    private static readonly (int First, int Last)[] _cleanTextRemoved =
    [
        (0x0, 0x8),
        (0xB, 0xC),
        (0xE, 0x1F),
        (0x7F, 0x9F),
        (0xAD, 0xAD),
        (0x600, 0x605),
        (0x61C, 0x61C),
        (0x6DD, 0x6DD),
        (0x70F, 0x70F),
        (0x180E, 0x180E),
        (0x200B, 0x200F),
        (0x202A, 0x202E),
        (0x2060, 0x2064),
        (0x2066, 0x206F),
        (0xE000, 0xF8FF),
        (0xFEFF, 0xFEFF),
        (0xFFF9, 0xFFFB),
        (0xFFFD, 0xFFFD),
        (0x110BD, 0x110BD),
        (0x1BCA0, 0x1BCA3),
        (0x1D173, 0x1D17A),
        (0xE0001, 0xE0001),
        (0xE0020, 0xE007F),
        (0xF0000, 0xFFFFD),
        (0x100000, 0x10FFFD),
    ];

    public static bool IsCleanTextRemoved(Rune rune) => InRanges(_cleanTextRemoved, rune.Value);

    private static bool InRanges((int First, int Last)[] ranges, int c)
    {
        int lo = 0;
        int hi = ranges.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            (int first, int last) = ranges[mid];
            if (c < first)
            {
                hi = mid - 1;
            }
            else if (c > last)
            {
                lo = mid + 1;
            }
            else
            {
                return true;
            }
        }

        return false;
    }

    // CJK ranges treated as Chinese by tokenizers 0.23.2 (verified exhaustively): note the reference skips
    // U+2B820..U+2B91F and starts its Extension E range at U+2B920.
    public static bool IsChinese(Rune rune)
    {
        int c = rune.Value;
        return c is >= 0x4E00 and <= 0x9FFF
            or >= 0x3400 and <= 0x4DBF
            or >= 0x20000 and <= 0x2A6DF
            or >= 0x2A700 and <= 0x2B73F
            or >= 0x2B740 and <= 0x2B81F
            or >= 0x2B920 and <= 0x2CEAF
            or >= 0xF900 and <= 0xFAFF
            or >= 0x2F800 and <= 0x2FA1F;
    }
}
