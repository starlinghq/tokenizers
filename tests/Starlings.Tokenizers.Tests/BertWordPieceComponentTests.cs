using System.Text;
using System.Text.Json;
using Starlings.Tokenizers.Pipeline;

namespace Starlings.Tokenizers.Tests;

/// <summary>Reference cases from huggingface/tokenizers 0.23.2 on a small BERT-profile vocabulary.</summary>
public sealed class BertWordPieceComponentTests
{
    private static readonly string _json = JsonSerializer.Serialize(new
    {
        version = "1.0",
        added_tokens = new[]
        {
            Added("[PAD]", 1), Added("[UNK]", 0), Added("[CLS]", 2), Added("[SEP]", 3), Added("[MASK]", 23),
        },
        normalizer = new
        {
            type = "BertNormalizer",
            clean_text = true,
            handle_chinese_chars = true,
            strip_accents = (bool?)null,
            lowercase = true,
        },
        pre_tokenizer = new { type = "BertPreTokenizer" },
        post_processor = new
        {
            type = "TemplateProcessing",
            single = new object[] { Special("[CLS]", 0), Sequence("A", 0), Special("[SEP]", 0) },
            pair = new object[] { Special("[CLS]", 0), Sequence("A", 0), Special("[SEP]", 0), Sequence("B", 1), Special("[SEP]", 1) },
            special_tokens = new Dictionary<string, object>
            {
                ["[CLS]"] = SpecialEntry("[CLS]", 2),
                ["[SEP]"] = SpecialEntry("[SEP]", 3),
            },
        },
        decoder = new { type = "WordPiece", prefix = "##", cleanup = true },
        model = new
        {
            type = "WordPiece",
            unk_token = "[UNK]",
            continuing_subword_prefix = "##",
            max_input_chars_per_word = 100,
            vocab = new Dictionary<string, int>
            {
                ["[UNK]"] = 0, ["[PAD]"] = 1, ["[CLS]"] = 2, ["[SEP]"] = 3,
                ["hello"] = 4, [","] = 5, ["world"] = 6, ["!"] = 7,
                ["un"] = 8, ["##aff"] = 9, ["##able"] = 10, ["cafe"] = 11,
                ["中"] = 12, ["文"] = 13, ["i"] = 14, ["'"] = 15, ["s"] = 16,
                ["n't"] = 17, ["."] = 18, ["?"] = 19, ["we"] = 20,
                ["##'"] = 21, ["##s"] = 22, ["[MASK]"] = 23,
                ["a"] = 24, ["b"] = 25, ["𠀀"] = 26, ["a⃝ा"] = 27, ["ab"] = 28,
            },
        },
    });

    [Fact]
    public void MatchesReferenceForNormalizationWordPieceOffsetsAndDecode()
    {
        Tokenizer tokenizer = Load(_json);

        TokenEncoding basic = tokenizer.Encode("Hello, world!");
        Assert.Equal([2, 4, 5, 6, 7, 3], basic.Ids);
        Assert.Equal(["[CLS]", "hello", ",", "world", "!", "[SEP]"], basic.Tokens);
        Assert.Equal([new(0, 0), new(0, 5), new(5, 6), new(7, 12), new(12, 13), new(0, 0)], basic.ScalarOffsets);
        Assert.Equal("[CLS] hello, world! [SEP]", tokenizer.Decode(basic.Ids.AsSpan(), skipSpecialTokens: false));

        TokenEncoding accent = tokenizer.Encode("Café");
        Assert.Equal([2, 11, 3], accent.Ids);
        Assert.Equal(new TokenOffset(0, 4), accent.ScalarOffsets[1]);

        TokenEncoding decomposed = tokenizer.Encode("Cafe\u0301");
        Assert.Equal([2, 11, 3], decomposed.Ids);
        Assert.Equal(new TokenOffset(0, 4), decomposed.ScalarOffsets[1]);

        TokenEncoding chinese = tokenizer.Encode("中文");
        Assert.Equal([2, 12, 13, 3], chinese.Ids);
        Assert.Equal([new(0, 0), new(0, 1), new(1, 2), new(0, 0)], chinese.ScalarOffsets);
    }

    [Fact]
    public void MatchesReferenceForPairsSequenceAndTypeIds()
    {
        TokenEncoding pair = Load(_json).Encode("hello", "world!");
        Assert.Equal([2, 4, 3, 6, 7, 3], pair.Ids);
        Assert.Equal([0, 0, 0, 1, 1, 1], pair.TypeIds);
        Assert.Equal<int?>([null, 0, null, 1, 1, null], pair.SequenceIds);
        Assert.Equal([new(0, 0), new(0, 5), new(0, 0), new(0, 5), new(5, 6), new(0, 0)], pair.ScalarOffsets);
    }

    [Fact]
    public void CleansControlsAndSplitsAdjacentPunctuationIndividually()
    {
        TokenEncoding encoding = Load(_json).Encode("Hello\0,!?\tworld");
        Assert.Equal([2, 4, 5, 7, 19, 6, 3], encoding.Ids);
        Assert.Equal([new(0, 0), new(0, 5), new(6, 7), new(7, 8), new(8, 9), new(10, 15), new(0, 0)], encoding.ScalarOffsets);
    }

    [Fact]
    public void UsesGreedySubwordsAndWholeWordUnknownFallback()
    {
        Tokenizer tokenizer = Load(_json);
        TokenEncoding segmented = tokenizer.Encode("unaffable", addSpecialTokens: false);
        Assert.Equal([8, 9, 10], segmented.Ids);
        Assert.Equal([new(0, 2), new(2, 5), new(5, 9)], segmented.ScalarOffsets);

        Tokenizer bounded = Load(_json.Replace(
            "\"max_input_chars_per_word\":100", "\"max_input_chars_per_word\":4", StringComparison.Ordinal));
        TokenEncoding tooLong = bounded.Encode("hello", addSpecialTokens: false);
        Assert.Equal([0], tooLong.Ids);
        Assert.Equal([new TokenOffset(0, 5)], tooLong.ScalarOffsets);
    }

    [Fact]
    public void ValidatesBertComponentOptionsAndWordPieceVocabulary()
    {
        Assert.Throws<TokenizerFormatException>(() => Load(_json.Replace(
            "\"lowercase\":true", "\"lowercase\":1", StringComparison.Ordinal)));
        Assert.Throws<TokenizerFormatException>(() => Load(_json.Replace(
            "\"cleanup\":true", "\"cleanup\":1", StringComparison.Ordinal)));
        Assert.Throws<TokenizerFormatException>(() => Load(_json.Replace(
            "\"clean_text\":true", "\"clean_text\":null", StringComparison.Ordinal)));
        Assert.Throws<TokenizerFormatException>(() => Load(_json.Replace(
            "\"max_input_chars_per_word\":100", "\"max_input_chars_per_word\":null", StringComparison.Ordinal)));
        AssertUnsupported(_json.Replace(
            "\"unk_token\":\"[UNK]\"", "\"unk_token\":\"[MISSING]\"", StringComparison.Ordinal), "model.unk_token");
        AssertUnsupported(_json.Replace(
            "\"clean_text\":true", "\"clean_text\":false", StringComparison.Ordinal), "normalizer.type=BertNormalizer");
        AssertUnsupported(_json.Replace(
            "\"handle_chinese_chars\":true", "\"handle_chinese_chars\":false", StringComparison.Ordinal), "normalizer.type=BertNormalizer");
        AssertUnsupported(_json.Replace(
            "\"strip_accents\":null", "\"strip_accents\":true", StringComparison.Ordinal), "normalizer.type=BertNormalizer");
        AssertUnsupported(_json.Replace(
            "\"lowercase\":true", "\"lowercase\":false", StringComparison.Ordinal), "normalizer.type=BertNormalizer");
    }

    [Fact]
    public void MatchesBertNormalizerControlMarkAndSupplementaryCjkBehavior()
    {
        Tokenizer tokenizer = Load(_json);

        foreach (string format in new[] { "\u200B", "\u200D", "\u200E", "\u00AD", "\uFEFF" })
        {
            TokenEncoding cleaned = tokenizer.Encode("A" + format + "B", addSpecialTokens: false);
            Assert.Equal([28], cleaned.Ids);
            Assert.Equal([new TokenOffset(0, 3)], cleaned.ScalarOffsets);
        }

        // tokenizers 0.23.2 removes Mn only; the enclosing and spacing marks survive.
        TokenEncoding marks = tokenizer.Encode("a\u034F\u20DD\u093E", addSpecialTokens: false);
        Assert.Equal([27], marks.Ids);
        Assert.Equal([new TokenOffset(0, 4)], marks.ScalarOffsets);

        // U+20000 is within the BERT CJK supplementary range and becomes its own token.
        TokenEncoding supplementary = tokenizer.Encode("a𠀀b", addSpecialTokens: false);
        Assert.Equal([24, 26, 25], supplementary.Ids);
        Assert.Equal([new(0, 1), new(1, 2), new(2, 3)], supplementary.ScalarOffsets);
    }

    [Theory]
    // Expected values from tokenizers 0.23.2 (.models/venv-reference, python -I):
    // BertNormalizer(clean_text=True, handle_chinese_chars=False).normalize_str("a" + chr(cp) + "b")
    // returns "ab" (removed) for these code points.
    [InlineData(0x0000)] // Cc NUL
    [InlineData(0x001F)] // Cc
    [InlineData(0x007F)] // Cc
    [InlineData(0x009F)] // Cc
    [InlineData(0x00AD)] // Cf
    [InlineData(0x0600)] // Cf
    [InlineData(0x200B)] // Cf
    [InlineData(0x110BD)] // Cf
    [InlineData(0x1BCA0)] // Cf
    [InlineData(0xE0001)] // Cf
    [InlineData(0xE0020)] // Cf
    [InlineData(0xE007F)] // Cf
    [InlineData(0xE000)] // Co (BMP private use)
    [InlineData(0xF8FF)] // Co
    [InlineData(0xF0000)] // Co (plane 15)
    [InlineData(0xFFFFD)] // Co
    [InlineData(0x100000)] // Co (plane 16)
    [InlineData(0x10FFFD)] // Co
    [InlineData(0xFFFD)] // replacement character
    public void CleanTextRemovesReferenceControlFormatAndPrivateUse(int codePoint)
    {
        Assert.True(BertUnicode.IsCleanTextRemoved(new Rune(codePoint)));
    }

    [Theory]
    // Expected values from tokenizers 0.23.2 (.models/venv-reference, python -I): normalize_str("a" + chr(cp) + "b")
    // keeps the code point. U+0890, U+08E2, U+110CD and U+13430 are Cf in newer Unicode data but not in the
    // reference's tables; Cn (U+0378, U+E0002) and noncharacters (U+FFFE, U+FFFFE, U+10FFFF) are kept.
    [InlineData(0x0378)]
    [InlineData(0x0890)]
    [InlineData(0x08E2)]
    [InlineData(0x110CD)]
    [InlineData(0x13430)]
    [InlineData(0xE0002)]
    [InlineData(0xFFFE)]
    [InlineData(0xFFFFE)]
    [InlineData(0x10FFFF)]
    [InlineData(0x0041)]
    public void CleanTextKeepsCodePointsTheReferenceKeeps(int codePoint)
    {
        Assert.False(BertUnicode.IsCleanTextRemoved(new Rune(codePoint)));
    }

    [Theory]
    // Expected values from tokenizers 0.23.2 (.models/venv-reference, python -I):
    // BertNormalizer(clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=False)
    // .normalize_str(chr(cp)) == "" for code points that NFD leaves unchanged. U+0898, U+11F00, U+1CF00
    // and U+1E4EC are Mn in newer Unicode data but kept by the reference; U+1734 and U+1171E are Mc in
    // newer data but stripped by the reference.
    [InlineData(0x0300, true)]
    [InlineData(0x1734, true)]
    [InlineData(0x1171E, true)]
    [InlineData(0x0898, false)]
    [InlineData(0x11F00, false)]
    [InlineData(0x1CF00, false)]
    [InlineData(0x1E4EC, false)]
    [InlineData(0x113B8, false)]
    [InlineData(0x0041, false)]
    public void StripAccentsUsesReferenceMarkTable(int codePoint, bool expected)
    {
        Assert.Equal(expected, BertUnicode.IsMark(new Rune(codePoint)));
    }

    [Theory]
    // Expected values from tokenizers 0.23.2: BertPreTokenizer().pre_tokenize_str("a" + chr(cp) + "a")
    // yields ["a", chr(cp), "a"]. U+061D and U+2E43..U+2E5D are punctuation in newer Unicode data but not
    // in the reference's tables; U+166D and U+111C9 were reclassified but the reference still isolates them.
    [InlineData(0x0021, true)]
    [InlineData(0x007E, true)]
    [InlineData(0x2010, true)]
    [InlineData(0x2E42, true)]
    [InlineData(0x166D, true)]
    [InlineData(0x111C9, true)]
    [InlineData(0x061D, false)]
    [InlineData(0x2E43, false)]
    [InlineData(0x2E5D, false)]
    [InlineData(0x0041, false)]
    public void PreTokenizerUsesReferencePunctuationTable(int codePoint, bool expected)
    {
        Assert.Equal(expected, BertUnicode.IsPunctuation(new Rune(codePoint)));
    }

    [Fact]
    public void StripAccentsKeepsNewerMarksLikeReference()
    {
        // tokenizers 0.23.2 with this profile's vocabulary (add_special_tokens=False): "A\U00011F00"
        // stays one word "a\U00011F00" and encodes to [UNK].
        Tokenizer tokenizer = Load(_json);
        TokenEncoding encoding = tokenizer.Encode("A\U00011F00", addSpecialTokens: false);
        Assert.Equal([0], encoding.Ids);
    }

    [Fact]
    public void CleanTextRemovesPrivateUseLikeReference()
    {
        // tokenizers 0.23.2: BertNormalizer(...).normalize_str("hello\uE000world") == "helloworld" and
        // "a\U000F0000b" == "ab"; "a\u0378b" is unchanged. Encoding with this profile's vocabulary
        // (add_special_tokens=False): "hello\uE000world" -> ids [0] ([UNK], one word) offsets [(0, 11)];
        // "a\U000F0000b" -> ids [28] offsets [(0, 3)].
        Tokenizer tokenizer = Load(_json);
        TokenEncoding privateUse = tokenizer.Encode("hello\uE000world", addSpecialTokens: false);
        Assert.Equal([0], privateUse.Ids);
        Assert.Equal([new TokenOffset(0, 11)], privateUse.ScalarOffsets);

        TokenEncoding supplementaryPrivateUse = tokenizer.Encode("a\U000F0000b", addSpecialTokens: false);
        Assert.Equal([28], supplementaryPrivateUse.Ids);
        Assert.Equal([new TokenOffset(0, 3)], supplementaryPrivateUse.ScalarOffsets);
    }

    [Theory]
    // tokenizers 0.23.2 (.models/venv-reference, python -I), BertNormalizer(handle_chinese_chars=True):
    // padded (Chinese): U+2B740, U+2B81F, U+2B920, U+2CEAF; unpadded: U+2B820, U+2B91F, U+2CEB0.
    // Exhaustive reference ranges: 3400-4DBF 4E00-9FFF F900-FAFF 20000-2A6DF 2A700-2B81F 2B920-2CEAF 2F800-2FA1F.
    [InlineData(0x2B73F, true)]
    [InlineData(0x2B740, true)]
    [InlineData(0x2B81F, true)]
    [InlineData(0x2B820, false)]
    [InlineData(0x2B91F, false)]
    [InlineData(0x2B920, true)]
    [InlineData(0x2CEAF, true)]
    [InlineData(0x2CEB0, false)]
    [InlineData(0x2A6DF, true)]
    [InlineData(0x2A6E0, false)]
    [InlineData(0x2A700, true)]
    [InlineData(0x2F800, true)]
    [InlineData(0x2FA1F, true)]
    [InlineData(0x2FA20, false)]
    [InlineData(0x30000, false)]
    public void ChineseRangesMatchReferenceBoundaries(int codePoint, bool expected)
    {
        Assert.Equal(expected, BertUnicode.IsChinese(new Rune(codePoint)));
    }

    [Fact]
    public void WordPieceDecoderWithEmptyPrefixConcatenatesLikeReference()
    {
        // tokenizers 0.23.2: decoders.WordPiece(prefix="", cleanup=False).decode(["ab", "cd"]) == "abcd" and
        // decoders.WordPiece(prefix="", cleanup=True).decode(["ab", "cd", "."]) == "abcd.".
        Assert.Equal("abcd", new WordPieceDecoder(string.Empty, cleanup: false).DecodeChain(["ab", "cd"])[0]);
        Assert.Equal("abcd.", new WordPieceDecoder(string.Empty, cleanup: true).DecodeChain(["ab", "cd", "."])[0]);
    }

    [Fact]
    public void MatchesReferenceWordPieceDecoderFragmentCleanup()
    {
        var decoder = new WordPieceDecoder("##", cleanup: true);

        Assert.Equal("##ing", decoder.DecodeChain(["##ing"])[0]);
        Assert.Equal("can ' t", decoder.DecodeChain(["can", "'", "t"])[0]);
        Assert.Equal("can't", decoder.DecodeChain(["can", "##'", "##t"])[0]);
        Assert.Equal("hello. world", decoder.DecodeChain(["hello", ".", "world"])[0]);
        Assert.Equal("##ab c", decoder.DecodeChain(["##a", "##b", "c"])[0]);
    }

    [Fact]
    public void MatchesPinnedEncoderAssetsAgainstTokenizers0232Cases()
    {
        // Expected values were generated with tokenizers==0.23.2 from the SHA-pinned tokenizer.json
        // assets recorded by TestAssets. These test tokenizer IDs/offsets only, not model outputs.
        foreach (PinnedAsset asset in new[]
        {
            TestAssets.MiniLmTokenizerJson,
            TestAssets.BgeSmallTokenizerJson,
            TestAssets.MsMarcoTokenizerJson,
        })
        {
            Tokenizer tokenizer = TestAssets.LoadTokenizer(asset)
                .WithPadding(null)
                .WithTruncation(null);

            TokenEncoding basic = tokenizer.Encode("Hello, world!");
            Assert.Equal([101, 7592, 1010, 2088, 999, 102], basic.Ids);
            Assert.Equal([new(0, 0), new(0, 5), new(5, 6), new(7, 12), new(12, 13), new(0, 0)], basic.ScalarOffsets);
            Assert.Equal("[CLS] hello, world! [SEP]", tokenizer.Decode(basic.Ids.AsSpan(), skipSpecialTokens: false));

            TokenEncoding dotless = tokenizer.Encode("Iİ");
            Assert.Equal([101, 2462, 102], dotless.Ids);
            Assert.Equal(new TokenOffset(0, 2), dotless.ScalarOffsets[1]);

            TokenEncoding capitalSharpS = tokenizer.Encode("ẞ");
            Assert.Equal([101, 1096, 102], capitalSharpS.Ids);

            TokenEncoding newMark = tokenizer.Encode("A\u1AB0");
            Assert.Equal([101, 1037, 102], newMark.Ids);
            Assert.Equal(new TokenOffset(0, 1), newMark.ScalarOffsets[1]);

            TokenEncoding supplementaryCjk = tokenizer.Encode("中文");
            Assert.Equal([101, 1746, 1861, 102], supplementaryCjk.Ids);
            Assert.Equal([new(0, 0), new(0, 1), new(1, 2), new(0, 0)], supplementaryCjk.ScalarOffsets);

            TokenEncoding controls = tokenizer.Encode("Hello\0,! ?\tworld");
            Assert.Equal([101, 7592, 1010, 999, 1029, 2088, 102], controls.Ids);
            Assert.Equal([new(0, 0), new(0, 5), new(6, 7), new(7, 8), new(9, 10), new(11, 16), new(0, 0)], controls.ScalarOffsets);

            TokenEncoding pair = tokenizer.Encode("Hello, world!", "Café");
            Assert.Equal([101, 7592, 1010, 2088, 999, 102, 7668, 102], pair.Ids);
            Assert.Equal([0, 0, 0, 0, 0, 0, 1, 1], pair.TypeIds);
            Assert.Equal([new(0, 0), new(0, 5), new(5, 6), new(7, 12), new(12, 13), new(0, 0), new(0, 4), new(0, 0)], pair.ScalarOffsets);

            TokenEncoding atLimit = tokenizer.Encode(new string('x', 100), addSpecialTokens: false);
            Assert.Equal(50, atLimit.Ids.Length);
            Assert.DoesNotContain(100, atLimit.Ids);
            TokenEncoding overLimit = tokenizer.Encode(new string('x', 101), addSpecialTokens: false);
            Assert.Equal([100], overLimit.Ids);
            Assert.Equal([new TokenOffset(0, 101)], overLimit.ScalarOffsets);
        }
    }

    private static Tokenizer Load(string json) => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json));

    private static void AssertUnsupported(string json, string expectedComponent)
    {
        UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(() => Load(json));
        Assert.Equal(expectedComponent, e.Component);
    }

    private static object Added(string content, int id) => new
    {
        id,
        content,
        single_word = false,
        lstrip = false,
        rstrip = false,
        normalized = false,
        special = true,
    };

    private static object Special(string id, int typeId) => new { SpecialToken = new { id, type_id = typeId } };

    private static object Sequence(string id, int typeId) => new { Sequence = new { id, type_id = typeId } };

    private static object SpecialEntry(string id, int tokenId) => new { id, ids = new[] { tokenId }, tokens = new[] { id } };
}
