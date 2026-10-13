using System.Text;
using System.Text.Json;
using Starlings.Tokenizers.Pipeline;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// The Qwen Split matcher's character classes against the reference engine, over every Unicode
/// scalar value (<c>TestData/qwen-bpe/unicode-classes.json</c>, probed through tokenizers 0.23.2 by
/// run_qwen_bpe_tokenizer.py --unicode-classes), and its alternatives on hand-picked inputs.
/// </summary>
public class QwenSplitPatternTests
{
    private static readonly Lazy<JsonDocument> _classes = new(() =>
        JsonDocument.Parse(File.ReadAllText(TestAssets.TestDataPath("qwen-bpe", "unicode-classes.json"))));

    [Theory]
    [InlineData("L", RegexUnicodeClasses.Letter)]
    [InlineData("N", RegexUnicodeClasses.Number)]
    [InlineData("M", RegexUnicodeClasses.Mark)]
    public void ClassesMatchTheReferenceForEveryScalar(string name, int bit)
    {
        HashSet<int> expected = Members(name);
        var wrong = new List<string>();
        for (int c = 0; c <= 0x10FFFF; c++)
        {
            if (c is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            bool actual = (RegexUnicodeClasses.Classify(c) & bit) != 0;
            if (actual != expected.Contains(c))
            {
                wrong.Add($"U+{c:X4}");
            }
        }

        Assert.True(wrong.Count == 0, $"{wrong.Count} code points differ: {string.Join(' ', wrong.Take(20))}");
    }

    [Fact]
    public void WhiteSpaceMatchesTheReference()
    {
        HashSet<int> expected = Members("s");
        Assert.Equal(25, expected.Count);
        for (int c = 0; c <= 0x10FFFF; c++)
        {
            if (c is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            Assert.True(expected.Contains(c) == UnicodeData.IsWhiteSpace(new Rune(c)), $"U+{c:X4}");
        }
    }

    [Fact]
    public void ContractionCaseFoldingMatchesTheReference()
    {
        JsonElement folds = _classes.Value.RootElement.GetProperty("contraction_folds");
        static int[] Get(JsonElement f, string t) => [.. f.GetProperty(t).EnumerateArray().Select(e => e.GetInt32())];
        Assert.Equal("DMSTdmst\u017F".Select(c => (int)c), Get(folds, "'{}"));
        Assert.Equal("RVrv".Select(c => (int)c), Get(folds, "'{}e"));
        Assert.Equal("Ee".Select(c => (int)c), Get(folds, "'r{}"));
        Assert.Equal("Ee".Select(c => (int)c), Get(folds, "'v{}"));
        Assert.Equal("Ll".Select(c => (int)c), Get(folds, "'{}l"));
        Assert.Equal("Ll".Select(c => (int)c), Get(folds, "'l{}"));
    }

    [Theory]
    [InlineData(false, "Hello world", "Hello| world")]
    [InlineData(false, "it's I'M we'Ll he'ſ", "it|'s| I|'M| we|'Ll| he|'ſ")]
    [InlineData(false, "a  b", "a| | b")]
    [InlineData(false, "a \n\n b", "a| \n\n| b")]
    [InlineData(false, "x  ", "x|  ")]
    [InlineData(false, " 123", " |1|2|3")]
    [InlineData(false, " ...\n\nx", " ...\n\n|x")]
    [InlineData(false, "x\U0001D400y", "x\U0001D400y")]
    [InlineData(false, "e\u0301x", "e|\u0301x")]
    [InlineData(true, "e\u0301x", "e\u0301x")]
    [InlineData(true, " \u0301!", " \u0301|!")]
    [InlineData(false, " \u0301!", " \u0301!")]
    public void SplitsLikeTheRegex(bool qwen35, string text, string expected)
    {
        QwenSplitPattern pattern = QwenSplitPattern.TryCreate(qwen35 ? QwenSplitPattern.Qwen35 : QwenSplitPattern.Qwen2)!;
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        List<PatternMatch> matches = pattern.FindMatches(utf8);
        Assert.All(matches, m => Assert.True(m.IsMatch));
        Assert.Equal(expected, string.Join('|', matches.Select(m => Encoding.UTF8.GetString(utf8, m.Start, m.End - m.Start))));
    }

    [Fact]
    public void OtherRegexesAreRejected()
    {
        Assert.Null(QwenSplitPattern.TryCreate(@"\s+"));
        Assert.Null(QwenSplitPattern.TryCreate(QwenSplitPattern.Qwen2 + " "));
    }

    private static HashSet<int> Members(string name)
    {
        var set = new HashSet<int>();
        foreach (JsonElement r in _classes.Value.RootElement.GetProperty("classes").GetProperty(name).EnumerateArray())
        {
            for (int c = r[0].GetInt32(); c <= r[1].GetInt32(); c++)
            {
                set.Add(c);
            }
        }

        return set;
    }
}
