using System.Text.Json;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Profile <c>modernbert-bpe-gte-reranker</c>: the checks <see cref="ProfileConformanceTests"/>
/// cannot make. The pinned file declares truncation (8000, longest first, right) and Fixed 8000
/// padding with <c>[PAD]</c> (50283). The fixtures were written with both cleared
/// (<c>--clear-asset-truncation-padding</c>), so this class checks that the file loads with exactly
/// those settings, that the runner recorded what it cleared, and that padding with <c>[PAD]</c> gives
/// the reference's padded batches.
/// </summary>
public sealed class ModernBertConformanceTests
{
    private const string Profile = "modernbert-bpe-gte-reranker";
    private const int PadId = 50283;

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
            ["normalizer:NFC", "pre_tokenizer:ByteLevel", "model:BPE", "added_tokens", "post_processor:TemplateProcessing", "decoder:ByteLevel"],
            tokenizer.Components);
        Assert.Equal(new TruncationOptions(8000), tokenizer.Truncation);
        Assert.Equal(new PaddingOptions(PadId, "[PAD]", FixedLength: 8000), tokenizer.Padding);

        using JsonDocument env = JsonDocument.Parse(File.ReadAllText(TestAssets.ConformancePath(Profile, "environment.json")));
        JsonElement cleared = env.RootElement.GetProperty("options").GetProperty("cleared_asset_settings");
        Assert.Equal(8000, cleared.GetProperty("truncation").GetProperty("max_length").GetInt32());
        Assert.Equal(8000, cleared.GetProperty("padding").GetProperty("length").GetInt32());
        Assert.Equal(PadId, cleared.GetProperty("padding").GetProperty("pad_id").GetInt32());
    }

    /// <summary>
    /// As loaded, every encoding is the reference's unpadded encoding (no corpus case reaches 8000
    /// tokens) followed by <c>[PAD]</c> up to 8000, with attention mask 0 on the padding.
    /// </summary>
    [Theory]
    [MemberData(nameof(CaseIds))]
    public void AsLoadedEncodeIsTheReferenceEncodingPaddedTo8000(string caseId)
    {
        JsonElement row = _cases.Value[caseId];
        int[] expected = FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids"));
        Assert.True(expected.Length < 8000);

        string text = row.GetProperty("text").GetString()!;
        TokenEncoding actual = row.TryGetProperty("pair", out JsonElement pair)
            ? Load().Encode(text, pair.GetString()!)
            : Load().Encode(text);

        Assert.Equal(8000, actual.Count);
        Assert.Equal(expected, actual.Ids.Take(expected.Length).ToArray());
        Assert.All(actual.Ids.Skip(expected.Length), id => Assert.Equal(PadId, id));
        Assert.Equal([.. Enumerable.Repeat(1, expected.Length), .. Enumerable.Repeat(0, 8000 - expected.Length)], actual.AttentionMask.ToArray());
    }

    [Fact]
    public void PaddingWithPadMatchesTheReferenceBatches()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(TestAssets.ConformancePath(Profile, "tokenizer-batch.json")));
        JsonElement padding = doc.RootElement.GetProperty("padding");
        Assert.Equal("right", padding.GetProperty("side").GetString());
        Assert.Equal("longest", padding.GetProperty("strategy").GetString());
        Assert.Equal(PadId, padding.GetProperty("pad_id").GetInt32());
        Tokenizer tokenizer = Load().WithTruncation(null).WithPadding(new PaddingOptions(PadId, "[PAD]"));

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

    private static Tokenizer Load() => TestAssets.LoadTokenizer(TestAssets.ModernBertGteRerankerTokenizerJson);

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
