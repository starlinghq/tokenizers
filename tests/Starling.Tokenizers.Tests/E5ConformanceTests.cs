using System.Text.Json;

namespace Starling.Tokenizers.Tests;

/// <summary>
/// Profile <c>xlmr-unigram-e5</c>: every case of the protected fixtures in
/// <c>conformance/xlmr-unigram-e5/</c> (reference: the pinned
/// <c>tokenizer.json</c> read by tokenizers 0.23.2) must match exactly.
/// </summary>
public class E5ConformanceTests
{
    private const int _expectedCaseCount = 49;
    private static readonly string _fixtureDir = TestAssets.ConformancePath("xlmr-unigram-e5");
    private static readonly Lazy<Dictionary<string, JsonElement>> _cases = new(() =>
        TestAssets.ReadJsonLines(Path.Combine(_fixtureDir, "tokenizer.jsonl"))
            .ToDictionary(r => r.GetProperty("id").GetString()!));

    public static TheoryData<string> CaseIds()
    {
        var data = new TheoryData<string>();
        foreach (string id in _cases.Value.Keys)
        {
            data.Add(id);
        }

        return data;
    }

    [Fact]
    public void FixtureHasEveryCase()
    {
        Assert.Equal(_expectedCaseCount, _cases.Value.Count);
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void EncodeMatchesReference(string caseId)
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.E5TokenizerJson);
        JsonElement row = _cases.Value[caseId];
        TokenEncoding actual = Encode(tokenizer, row);
        FixtureAssert.EncodingMatches(row.GetProperty("encoding"), actual, caseId);
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void DecodeMatchesReference(string caseId)
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.E5TokenizerJson);
        JsonElement row = _cases.Value[caseId];
        int[] ids = FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids"));
        JsonElement decode = row.GetProperty("decode");
        Assert.Equal(decode.GetProperty("skip_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: true));
        Assert.Equal(decode.GetProperty("keep_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: false));
    }

    public static TheoryData<string> TruncatedCaseIds()
    {
        var data = new TheoryData<string>();
        foreach ((string id, JsonElement row) in _cases.Value)
        {
            if (row.TryGetProperty("truncated", out _))
            {
                data.Add(id);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(TruncatedCaseIds))]
    public void TruncatedEncodeMatchesReference(string caseId)
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.E5TokenizerJson);
        JsonElement row = _cases.Value[caseId];
        JsonElement truncated = row.GetProperty("truncated");
        Assert.Equal("longest_first", truncated.GetProperty("strategy").GetString());
        Assert.Equal("right", truncated.GetProperty("direction").GetString());
        Tokenizer truncating = tokenizer.WithTruncation(new TruncationOptions(truncated.GetProperty("max_length").GetInt32()));
        FixtureAssert.EncodingMatches(truncated.GetProperty("encoding"), Encode(truncating, row), caseId + "/truncated");
    }

    [Fact]
    public void TruncationCasesArePresent()
    {
        Assert.Equal(["long-english", "long-mixed"], _cases.Value.Where(kv => kv.Value.TryGetProperty("truncated", out _)).Select(kv => kv.Key).Order());
    }

    [Fact]
    public void BatchMatchesReference()
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.E5TokenizerJson);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(_fixtureDir, "tokenizer-batch.json")));
        JsonElement padding = doc.RootElement.GetProperty("padding");
        Assert.Equal("right", padding.GetProperty("side").GetString());
        Assert.Equal("longest", padding.GetProperty("strategy").GetString());
        int padId = padding.GetProperty("pad_id").GetInt32();
        Assert.Equal("<pad>", tokenizer.IdToToken(padId));

        Tokenizer padded = tokenizer.WithPadding(new PaddingOptions(padId, "<pad>"));
        JsonElement[] rows = [.. doc.RootElement.GetProperty("rows").EnumerateArray()];
        string[] texts = [.. rows.Select(r => _cases.Value[r.GetProperty("id").GetString()!].GetProperty("text").GetString()!)];
        IReadOnlyList<TokenEncoding> batch = padded.EncodeBatch(texts);

        Assert.Equal(rows.Length, batch.Count);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("ids")), batch[i].Ids);
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("attention_mask")), batch[i].AttentionMask);

            // Batch equivalence: the padded row starts with the unpadded single encode.
            TokenEncoding single = tokenizer.Encode(texts[i]);
            Assert.Equal(single.Ids, batch[i].Ids.Take(single.Count));
            Assert.All(batch[i].Ids.Skip(single.Count), id => Assert.Equal(padId, id));
        }
    }

    private static TokenEncoding Encode(Tokenizer tokenizer, JsonElement row)
    {
        string text = row.GetProperty("text").GetString()!;
        return row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!)
            : tokenizer.Encode(text);
    }
}

internal static class FixtureAssert
{
    public static int[] Ints(JsonElement array) => [.. array.EnumerateArray().Select(e => e.GetInt32())];

    public static void EncodingMatches(JsonElement expected, TokenEncoding actual, string caseId)
    {
        Assert.Equal(Ints(expected.GetProperty("ids")), actual.Ids);
        Assert.Equal(expected.GetProperty("tokens").EnumerateArray().Select(e => e.GetString()!), actual.Tokens);
        Assert.Equal(Ints(expected.GetProperty("type_ids")), actual.TypeIds);
        Assert.Equal(Ints(expected.GetProperty("attention_mask")), actual.AttentionMask);
        Assert.Equal(Ints(expected.GetProperty("special_tokens_mask")), actual.SpecialTokensMask);

        JsonElement offsets = expected.GetProperty("offsets");
        Assert.Equal(Offsets(offsets.GetProperty("utf16-code-units")), actual.Offsets);
        Assert.Equal(Offsets(offsets.GetProperty("unicode-scalars")), actual.ScalarOffsets);

        if (expected.TryGetProperty("sequence_ids", out JsonElement sequenceIds))
        {
            int?[] expectedSeq = [.. sequenceIds.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? (int?)null : e.GetInt32())];
            Assert.Equal(expectedSeq, actual.SequenceIds);
        }

        Assert.True(actual.Count == actual.Ids.Length, caseId);
    }

    private static TokenOffset[] Offsets(JsonElement array) =>
        [.. array.EnumerateArray().Select(p => new TokenOffset(p[0].GetInt32(), p[1].GetInt32()))];
}
