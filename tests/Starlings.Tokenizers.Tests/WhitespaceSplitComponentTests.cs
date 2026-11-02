using System.Text;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// <c>WhitespaceSplit</c> pre-tokenizer against huggingface/tokenizers 0.23.3: two small Unigram
/// tokenizers, <c>WhitespaceSplit</c> alone and <c>Sequence[WhitespaceSplit, Metaspace]</c> (the
/// pipeline of sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Expected ids and
/// Unicode-scalar offsets are the output of <c>reference/components/whitespace_split_cases.py</c>
/// (<c>reference/tokenizer-json/requirements.txt</c>, <c>python -I</c>), which builds the same
/// tokenizers. The inputs cover every White_Space character class Rust's <c>char::is_whitespace</c>
/// accepts (U+2000 to U+200A each), runs, leading and trailing whitespace, whitespace-only and empty
/// input, and characters that are not whitespace (U+200B, U+180E, U+FEFF).
/// </summary>
public sealed class WhitespaceSplitComponentTests
{
    private const string Vocab = """
        [["<unk>", -1.0], ["a", -1.1], ["b", -1.2], ["c", -1.3], ["ab", -1.4], ["\u00e9", -1.5], ["\u4e2d", -1.6], ["\u6587", -1.7000000000000002], ["\u200b", -1.8], ["\u2581", -1.9], ["\u2581a", -2.0], ["\u2581b", -2.1], ["\u2581ab", -2.2], ["\u2581c", -2.3]]
        """;

    private const string WhitespaceSplit = """{"type":"WhitespaceSplit"}""";

    private const string WhitespaceSplitMetaspace =
        """{"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"\u2581","add_prefix_space":true}]}""";

    public static TheoryData<string, string, int[], int[]> ReferenceCases() => new()
    {
        { "whitespace_split", "", [], [] },
        { "whitespace_split", "a", [1], [0, 1] },
        { "whitespace_split", "a b", [1, 2], [0, 1, 2, 3] },
        { "whitespace_split", "   ", [], [] },
        { "whitespace_split", "  ab\u0009c\u000A", [4, 3], [2, 4, 5, 6] },
        { "whitespace_split", "a\u000D\u000Ab", [1, 2], [0, 1, 3, 4] },
        { "whitespace_split", "a\u000Bb\u000Cc", [1, 2, 3], [0, 1, 2, 3, 4, 5] },
        { "whitespace_split", "a\u0085b", [1, 2], [0, 1, 2, 3] },
        { "whitespace_split", "a\u00A0b", [1, 2], [0, 1, 2, 3] },
        { "whitespace_split", "\u1680a", [1], [1, 2] },
        { "whitespace_split", "a\u2000\u200Ab", [1, 2], [0, 1, 3, 4] },
        { "whitespace_split", "a\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009b", [1, 2], [0, 1, 10, 11] },
        { "whitespace_split", "a\u2028b\u2029c", [1, 2, 3], [0, 1, 2, 3, 4, 5] },
        { "whitespace_split", "a\u202Fb\u205Fc", [1, 2, 3], [0, 1, 2, 3, 4, 5] },
        { "whitespace_split", "a\u3000b", [1, 2], [0, 1, 2, 3] },
        { "whitespace_split", "a\u200Bb", [1, 8, 2], [0, 1, 1, 2, 2, 3] },
        { "whitespace_split", "a\u180Eb", [1, 0, 2], [0, 1, 1, 2, 2, 3] },
        { "whitespace_split", "a\uFEFFb", [1, 0, 2], [0, 1, 1, 2, 2, 3] },
        { "whitespace_split", "\u00E9 \u4E2D\u6587", [5, 6, 7], [0, 1, 2, 3, 3, 4] },
        { "whitespace_split", "\U0001F600 a", [0, 1], [0, 1, 2, 3] },
        { "whitespace_split_metaspace", "", [], [] },
        { "whitespace_split_metaspace", "a", [10], [0, 1] },
        { "whitespace_split_metaspace", "a b", [10, 11], [0, 1, 2, 3] },
        { "whitespace_split_metaspace", "   ", [], [] },
        { "whitespace_split_metaspace", "  ab\u0009c\u000A", [12, 13], [2, 4, 5, 6] },
        { "whitespace_split_metaspace", "a\u000D\u000Ab", [10, 11], [0, 1, 3, 4] },
        { "whitespace_split_metaspace", "a\u000Bb\u000Cc", [10, 11, 13], [0, 1, 2, 3, 4, 5] },
        { "whitespace_split_metaspace", "a\u0085b", [10, 11], [0, 1, 2, 3] },
        { "whitespace_split_metaspace", "a\u00A0b", [10, 11], [0, 1, 2, 3] },
        { "whitespace_split_metaspace", "\u1680a", [10], [1, 2] },
        { "whitespace_split_metaspace", "a\u2000\u200Ab", [10, 11], [0, 1, 3, 4] },
        { "whitespace_split_metaspace", "a\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009b", [10, 11], [0, 1, 10, 11] },
        { "whitespace_split_metaspace", "a\u2028b\u2029c", [10, 11, 13], [0, 1, 2, 3, 4, 5] },
        { "whitespace_split_metaspace", "a\u202Fb\u205Fc", [10, 11, 13], [0, 1, 2, 3, 4, 5] },
        { "whitespace_split_metaspace", "a\u3000b", [10, 11], [0, 1, 2, 3] },
        { "whitespace_split_metaspace", "a\u200Bb", [10, 8, 2], [0, 1, 1, 2, 2, 3] },
        { "whitespace_split_metaspace", "a\u180Eb", [10, 0, 2], [0, 1, 1, 2, 2, 3] },
        { "whitespace_split_metaspace", "a\uFEFFb", [10, 0, 2], [0, 1, 1, 2, 2, 3] },
        { "whitespace_split_metaspace", "\u00E9 \u4E2D\u6587", [9, 5, 9, 6, 7], [0, 1, 0, 1, 2, 3, 2, 3, 3, 4] },
        { "whitespace_split_metaspace", "\U0001F600 a", [9, 0, 10], [0, 1, 0, 1, 2, 3] },
    };

    [Theory]
    [MemberData(nameof(ReferenceCases))]
    public void MatchesReference(string variant, string text, int[] ids, int[] scalarOffsets)
    {
        Tokenizer tokenizer = Load(variant == "whitespace_split" ? WhitespaceSplit : WhitespaceSplitMetaspace);
        TokenEncoding encoding = tokenizer.Encode(text);
        Assert.Equal(ids, encoding.Ids);
        Assert.Equal(
            Enumerable.Range(0, scalarOffsets.Length / 2).Select(i => new TokenOffset(scalarOffsets[2 * i], scalarOffsets[(2 * i) + 1])),
            encoding.ScalarOffsets);
    }

    [Fact]
    public void LoaderAcceptsOnlyTheTypeField()
    {
        Assert.Contains("pre_tokenizer:WhitespaceSplit", Load(WhitespaceSplit).Components);
        UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(
            () => Load("""{"type":"WhitespaceSplit","split":true}"""));
        Assert.Equal("pre_tokenizer.type=WhitespaceSplit.split", e.Component);
    }

    private static Tokenizer Load(string preTokenizer) => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(
        """{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":"""
        + preTokenizer
        + ""","post_processor":null,"decoder":null,"model":{"type":"Unigram","unk_id":0,"vocab":"""
        + Vocab.Trim()
        + "}}"));
}
