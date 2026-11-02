using System.Text.Json;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Profile <c>xlmr-unigram-paraphrase-multilingual</c>: the checks <see cref="ProfileConformanceTests"/>
/// cannot make. The pinned file declares truncation (128, longest first, right) and BatchLongest padding
/// with <c>&lt;pad&gt;</c> (1). The fixtures were written with both cleared
/// (<c>--clear-asset-truncation-padding</c>) and <c>--truncate 128</c>, so the tokenizer exactly as
/// loaded must give the <c>truncated</c> record where there is one and the plain encoding otherwise, and
/// its own padding must give the reference's padded batches.
/// </summary>
public sealed class ParaphraseMultilingualConformanceTests
{
    private const string Profile = "xlmr-unigram-paraphrase-multilingual";

    private static readonly Lazy<Dictionary<string, JsonElement>> _cases = new(() =>
        TestAssets.ReadJsonLines(TestAssets.ConformancePath(Profile, "tokenizer.jsonl"))
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
    public void LoadsTheDeclaredPipelineTruncationAndPadding()
    {
        Tokenizer tokenizer = Load();
        Assert.Equal(
            [
                "normalizer:Precompiled", "pre_tokenizer:WhitespaceSplit", "pre_tokenizer:Metaspace", "pre_tokenizer:Sequence",
                "model:Unigram", "added_tokens", "post_processor:TemplateProcessing", "decoder:Metaspace",
            ],
            tokenizer.Components);
        Assert.Equal(new TruncationOptions(128), tokenizer.Truncation);
        Assert.Equal(new PaddingOptions(1, "<pad>"), tokenizer.Padding);

        using JsonDocument env = JsonDocument.Parse(File.ReadAllText(TestAssets.ConformancePath(Profile, "environment.json")));
        Assert.Equal(128, env.RootElement.GetProperty("truncate").GetInt32());
        JsonElement cleared = env.RootElement.GetProperty("options").GetProperty("cleared_asset_settings");
        Assert.Equal(128, cleared.GetProperty("truncation").GetProperty("max_length").GetInt32());
        Assert.Equal(1, cleared.GetProperty("padding").GetProperty("pad_id").GetInt32());
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void AsLoadedEncodeMatchesTheReferenceAtTheDeclaredMaxLength(string caseId)
    {
        JsonElement row = _cases.Value[caseId];
        Tokenizer tokenizer = Load();
        AssertAsLoaded(tokenizer, row, row, addSpecialTokens: true, caseId);
        AssertAsLoaded(tokenizer, row, row.GetProperty("without_special_tokens"), addSpecialTokens: false, $"{caseId}/without_special_tokens");
    }

    [Fact]
    public void DeclaredPaddingMatchesTheReferenceBatches()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(TestAssets.ConformancePath(Profile, "tokenizer-batch.json")));
        JsonElement padding = doc.RootElement.GetProperty("padding");
        Assert.Equal("right", padding.GetProperty("side").GetString());
        Assert.Equal("longest", padding.GetProperty("strategy").GetString());
        Assert.Equal(1, padding.GetProperty("pad_id").GetInt32());
        Tokenizer tokenizer = Load();

        JsonElement[] rows = [.. doc.RootElement.GetProperty("rows").EnumerateArray()];
        string[] texts = [.. rows.Select(r => _cases.Value[r.GetProperty("id").GetString()!].GetProperty("text").GetString()!)];
        AssertBatch(rows, tokenizer.EncodeBatch(texts));

        JsonElement[] pairRows = [.. doc.RootElement.GetProperty("pairs").GetProperty("rows").EnumerateArray()];
        (string, string)[] pairs = [.. pairRows.Select(r =>
        {
            JsonElement c = _cases.Value[r.GetProperty("id").GetString()!];
            return (c.GetProperty("text").GetString()!, c.GetProperty("pair").GetString()!);
        })];
        AssertBatch(pairRows, tokenizer.EncodeBatch(pairs));

        Assert.True(doc.RootElement.GetProperty("equivalent").GetBoolean());
        Assert.True(doc.RootElement.GetProperty("pairs").GetProperty("equivalent").GetBoolean());
    }

    private static Tokenizer Load() => TestAssets.LoadTokenizer(TestAssets.ParaphraseMultilingualTokenizerJson);

    private static void AssertAsLoaded(Tokenizer tokenizer, JsonElement row, JsonElement record, bool addSpecialTokens, string label)
    {
        JsonElement expected = record.TryGetProperty("truncated", out JsonElement truncated)
            ? truncated.GetProperty("encoding")
            : record.GetProperty("encoding");
        string text = row.GetProperty("text").GetString()!;
        TokenEncoding actual = row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!, addSpecialTokens)
            : tokenizer.Encode(text, addSpecialTokens);
        FixtureAssert.EncodingMatches(expected, actual, label);
    }

    private static void AssertBatch(JsonElement[] rows, IReadOnlyList<TokenEncoding> batch)
    {
        Assert.NotEmpty(rows);
        Assert.Equal(rows.Length, batch.Count);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("ids")), batch[i].Ids);
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("attention_mask")), batch[i].AttentionMask);
        }
    }
}
