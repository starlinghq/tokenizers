using System.IO.Compression;
using System.Text;
using System.Text.Json;
using Starlings.Tokenizers.Pipeline;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// The o200k Split matcher against the reference: its letter-category tables over every Unicode scalar
/// value (<c>TestData/o200k-bpe/unicode-classes.json</c>), and its pieces on every one- to three-character
/// string over one representative per class plus 4,000 seeded random strings
/// (<c>TestData/o200k-bpe/split.jsonl.gz</c>). Both written by reference/o200k-bpe/run_o200k_reference.py
/// with tokenizers 0.23.3 (<c>Split(Regex(o200k), "removed", invert=True)</c>; the runner checks that
/// <c>"isolated"</c> without invert gives the same pieces).
/// </summary>
public class O200kSplitPatternTests
{
    private static readonly Lazy<JsonDocument> _classes = new(() =>
        JsonDocument.Parse(File.ReadAllText(TestAssets.TestDataPath("o200k-bpe", "unicode-classes.json"))));

    [Theory]
    [InlineData("Lu", RegexUnicodeClasses.UppercaseLetter)]
    [InlineData("Ll", RegexUnicodeClasses.LowercaseLetter)]
    [InlineData("Lt", RegexUnicodeClasses.TitlecaseLetter)]
    [InlineData("Lm", RegexUnicodeClasses.ModifierLetter)]
    [InlineData("Lo", RegexUnicodeClasses.OtherLetter)]
    [InlineData("L", RegexUnicodeClasses.Letter)]
    [InlineData("N", RegexUnicodeClasses.Number)]
    [InlineData("M", RegexUnicodeClasses.Mark)]
    public void ClassesMatchTheReferenceForEveryScalar(string name, int bit)
    {
        var expected = new HashSet<int>();
        foreach (JsonElement r in _classes.Value.RootElement.GetProperty("classes").GetProperty(name).EnumerateArray())
        {
            for (int c = r[0].GetInt32(); c <= r[1].GetInt32(); c++)
            {
                expected.Add(c);
            }
        }

        var wrong = new List<string>();
        for (int c = 0; c <= 0x10FFFF; c++)
        {
            if (c is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            if (((RegexUnicodeClasses.Classify(c) & bit) != 0) != expected.Contains(c))
            {
                wrong.Add($"U+{c:X4}");
            }
        }

        Assert.True(wrong.Count == 0, $"{wrong.Count} code points differ: {string.Join(' ', wrong.Take(20))}");
    }

    [Fact]
    public void SplitsLikeTheReferenceOnTheDump()
    {
        O200kSplitPattern pattern = O200kSplitPattern.TryCreate(O200kSplitPattern.O200k)!;
        int rows = 0;
        var failures = new List<string>();
        using var stream = new GZipStream(File.OpenRead(TestAssets.TestDataPath("o200k-bpe", "split.jsonl.gz")), CompressionMode.Decompress);
        using var reader = new StreamReader(stream, new UTF8Encoding(false, true));
        while (reader.ReadLine() is { } line)
        {
            if (line.Length == 0)
            {
                continue;
            }

            using JsonDocument doc = JsonDocument.Parse(line);
            string text = doc.RootElement.GetProperty("t").GetString()!;
            int[] byteAt = ScalarToUtf8(text);
            string expected = string.Join(' ', doc.RootElement.GetProperty("s").EnumerateArray()
                .Select(o => $"{byteAt[o[0].GetInt32()]}-{byteAt[o[1].GetInt32()]}"));
            List<PatternMatch> matches = pattern.FindMatches(Encoding.UTF8.GetBytes(text));
            string actual = string.Join(' ', matches.Select(m => $"{m.Start}-{m.End}"));
            if (!matches.TrueForAll(m => m.IsMatch) || expected != actual)
            {
                failures.Add($"{JsonSerializer.Serialize(text)}: expected {expected} actual {actual}");
            }

            rows++;
        }

        Assert.True(rows >= 12000, $"{rows} rows");
        Assert.True(failures.Count == 0, $"{failures.Count} of {rows} differ:\n{string.Join('\n', failures.Take(20))}");
    }

    // Hand-picked spot checks; expected pieces copied by hand from tokenizers 0.23.3
    // (Split(Regex(o200k), "removed", invert=True).pre_tokenize_str). They are not
    // fixtures: the reference-written evidence is TestData/o200k-bpe (run_o200k_reference.py).
    [Theory]
    [InlineData("Hello world", "Hello| world")]
    [InlineData("it's IT'S we'Ll heſ", "it's| IT'S| we'Ll| heſ")]
    [InlineData("camelCase XMLHttpRequest", "camel|Case| XMLHttp|Request")]
    [InlineData("日A", "日|A")]
    [InlineData("A日", "A日")]
    [InlineData("1234567 89", "123|456|7| |89")]
    [InlineData("a/b//c", "a|/b|//|c")]
    [InlineData("!\n/x", "!\n/|x")]
    [InlineData(" ́!", " ́|!")]
    [InlineData("x  ", "x|  ")]
    [InlineData("a \n\n b", "a| \n\n| b")]
    public void SplitsLikeTheRegex(string text, string expected)
    {
        O200kSplitPattern pattern = O200kSplitPattern.TryCreate(O200kSplitPattern.O200k)!;
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        List<PatternMatch> matches = pattern.FindMatches(utf8);
        Assert.All(matches, m => Assert.True(m.IsMatch));
        Assert.Equal(expected, string.Join('|', matches.Select(m => Encoding.UTF8.GetString(utf8, m.Start, m.End - m.Start))));
    }

    [Fact]
    public void OtherRegexesAreRejected()
    {
        Assert.Null(O200kSplitPattern.TryCreate(QwenSplitPattern.Llama3));
        Assert.Null(O200kSplitPattern.TryCreate(O200kSplitPattern.O200k + " "));
    }

    [Theory]
    [InlineData("Isolated", false, null)]
    [InlineData("Removed", true, null)]
    [InlineData("Removed", false, "invert")]
    [InlineData("Isolated", true, "invert")]
    [InlineData("MergedWithPrevious", false, "behavior")]
    [InlineData("MergedWithPrevious", true, "behavior")]
    public void LoaderAcceptsOnlyTheGaplessForms(string behavior, bool invert, string? rejectedField)
    {
        string json = MinimalTokenizerJson(behavior, invert);
        if (rejectedField is null)
        {
            Tokenizer t = Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json));
            Assert.Equal(["ab", " c"], t.Encode("ab c").Tokens);
            Assert.Empty(t.Encode(string.Empty).Ids);
        }
        else
        {
            UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(
                () => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json)));
            Assert.Equal($"pre_tokenizer.type=Split.{rejectedField}", e.Component);
        }
    }

    private static string MinimalTokenizerJson(string behavior, bool invert)
    {
        var doc = new
        {
            version = "1.0",
            truncation = (object?)null,
            padding = (object?)null,
            added_tokens = Array.Empty<object>(),
            normalizer = (object?)null,
            pre_tokenizer = new
            {
                type = "Split",
                pattern = new Dictionary<string, string> { ["Regex"] = O200kSplitPattern.O200k },
                behavior,
                invert,
            },
            post_processor = (object?)null,
            decoder = (object?)null,
            model = new
            {
                type = "WordLevelPlaceholder",
            },
        };
        string json = JsonSerializer.Serialize(doc);
        const string model = "{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{\"a\":0,\"b\":1,\" \":2,\"c\":3,\"ab\":4,\" c\":5},\"merges\":[[\"a\",\"b\"],[\" \",\"c\"]]}";
        return json.Replace("{\"type\":\"WordLevelPlaceholder\"}", model, StringComparison.Ordinal);
    }

    private static int[] ScalarToUtf8(string text)
    {
        var at = new List<int> { 0 };
        foreach (Rune r in text.EnumerateRunes())
        {
            at.Add(at[^1] + r.Utf8SequenceLength);
        }

        return [.. at];
    }
}
