using System.Text.Json;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Profiles <c>qwen3-bpe-embedding</c> (Qwen/Qwen3-Embedding-0.6B) and <c>qwen3-bpe-chat</c>
/// (Qwen/Qwen3-Reranker-0.6B, the same file as Qwen/Qwen3-0.6B): the checks the data-driven
/// <see cref="ProfileConformanceTests"/> cannot make. These are the declared pipelines, the appended
/// <c>&lt;|endoftext|&gt;</c> on every embedding encode and the truncation budget it reserves, the padded
/// batches recorded by the reference runner, and the relation between the two profiles.
/// </summary>
public sealed class Qwen3BpeConformanceTests
{
    private const int _endOfText = 151643;
    private const int _expectedCaseCount = 137;

    private static readonly Lazy<Dictionary<string, JsonElement>> _embeddingCases = new(() => ReadCases("qwen3-bpe-embedding"));
    private static readonly Lazy<Dictionary<string, JsonElement>> _chatCases = new(() => ReadCases("qwen3-bpe-chat"));

    private static Tokenizer Embedding() => TestAssets.LoadTokenizer(TestAssets.Qwen3EmbeddingTokenizerJson);

    private static Tokenizer Chat() => TestAssets.LoadTokenizer(TestAssets.Qwen3ChatTokenizerJson);

    public static TheoryData<string> EmbeddingCases() => [.. _embeddingCases.Value.Keys];

    [Fact]
    public void FixturesHaveEveryCase()
    {
        Assert.Equal(_expectedCaseCount, _embeddingCases.Value.Count);
        Assert.Equal(_expectedCaseCount, _chatCases.Value.Count);
    }

    [Fact]
    public void DeclaresTheEmbeddingPipeline()
    {
        Assert.Equal(
            ["normalizer:NFC", "pre_tokenizer:Split", "pre_tokenizer:ByteLevel", "pre_tokenizer:Sequence", "model:BPE", "added_tokens",
             "post_processor:ByteLevel", "post_processor:TemplateProcessing", "post_processor:Sequence", "decoder:ByteLevel"],
            Embedding().Components);
    }

    [Fact]
    public void DeclaresTheChatPipeline()
    {
        Assert.Equal(
            ["normalizer:NFC", "pre_tokenizer:Split", "pre_tokenizer:ByteLevel", "pre_tokenizer:Sequence", "model:BPE", "added_tokens",
             "post_processor:ByteLevel", "decoder:ByteLevel"],
            Chat().Components);
    }

    [Fact]
    public void VocabularySizes()
    {
        foreach (Tokenizer tokenizer in new[] { Embedding(), Chat() })
        {
            Assert.Equal(151643, tokenizer.GetVocabularySize(includeAddedTokens: false));
            Assert.Equal(151669, tokenizer.GetVocabularySize(includeAddedTokens: true));
            Assert.Equal("<|endoftext|>", tokenizer.IdToToken(_endOfText));
        }
    }

    [Theory]
    [MemberData(nameof(EmbeddingCases))]
    public void EmbeddingEncodeEndsWithEndOfText(string caseId)
    {
        JsonElement row = _embeddingCases.Value[caseId];
        TokenEncoding e = Encode(Embedding(), row);

        // The reference fixture and this library agree that the last token is the appended special token.
        Assert.Equal(_endOfText, FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids"))[^1]);
        Assert.Equal(_endOfText, e.Ids[^1]);
        Assert.Equal(1, e.SpecialTokensMask[^1]);
        Assert.Null(e.SequenceIds[^1]);

        // add_special_tokens false appends nothing, and adds no other token.
        TokenEncoding plain = Encode(Embedding(), row, addSpecialTokens: false);
        Assert.Equal(e.Ids.Take(e.Count - 1), plain.Ids);
    }

    [Theory]
    [MemberData(nameof(EmbeddingCases))]
    public void ChatEncodeIsTheEmbeddingEncodeWithoutEndOfText(string caseId)
    {
        // Same vocabulary, merges, normalizer and pre-tokenizer: the files differ only in the
        // post-processor (ByteLevel alone vs Sequence[ByteLevel, TemplateProcessing]).
        JsonElement row = _embeddingCases.Value[caseId];
        TokenEncoding chat = Encode(Chat(), row);
        TokenEncoding embedding = Encode(Embedding(), row);
        Assert.Equal(embedding.Ids.Take(embedding.Count - 1), chat.Ids);
        Assert.Equal(embedding.Offsets.Take(embedding.Count - 1), chat.Offsets);
        Assert.Equal(chat.Ids, Encode(Chat(), row, addSpecialTokens: false).Ids);
    }

    [Fact]
    public void TruncatedEmbeddingRowsKeepEndOfTextWithinMaxLength()
    {
        int truncated = 0;
        int pairs = 0;
        foreach ((string id, JsonElement row) in _embeddingCases.Value)
        {
            if (!row.TryGetProperty("truncated", out JsonElement t))
            {
                continue;
            }

            int maxLength = t.GetProperty("max_length").GetInt32();
            TokenEncoding e = Encode(Embedding().WithTruncation(new TruncationOptions(maxLength)), row);
            Assert.True(maxLength == e.Count, $"{id}: {e.Count} tokens, max_length {maxLength}");
            Assert.Equal(_endOfText, e.Ids[^1]);
            Assert.Equal(FixtureAssert.Ints(t.GetProperty("encoding").GetProperty("ids")), e.Ids);
            truncated++;
            pairs += row.TryGetProperty("pair", out _) ? 1 : 0;
        }

        // The fixtures were written with --truncate 16: long single texts and at least one pair truncate.
        Assert.True(truncated >= 40, $"{truncated} truncated rows");
        Assert.True(pairs >= 1, $"{pairs} truncated pairs");
    }

    [Theory]
    [InlineData("qwen3-bpe-embedding")]
    [InlineData("qwen3-bpe-chat")]
    public void BatchMatchesReference(string profile)
    {
        Tokenizer tokenizer = profile == "qwen3-bpe-embedding" ? Embedding() : Chat();
        Dictionary<string, JsonElement> cases = profile == "qwen3-bpe-embedding" ? _embeddingCases.Value : _chatCases.Value;
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(TestAssets.ConformancePath(profile, "tokenizer-batch.json")));
        JsonElement padding = doc.RootElement.GetProperty("padding");
        Assert.Equal("right", padding.GetProperty("side").GetString());
        Assert.Equal("longest", padding.GetProperty("strategy").GetString());
        int padId = padding.GetProperty("pad_id").GetInt32();
        Assert.Equal(_endOfText, padId);
        Tokenizer padded = tokenizer.WithPadding(new PaddingOptions(padId, "<|endoftext|>"));

        JsonElement[] rows = [.. doc.RootElement.GetProperty("rows").EnumerateArray()];
        string[] texts = [.. rows.Select(r => cases[r.GetProperty("id").GetString()!].GetProperty("text").GetString()!)];
        AssertBatch(rows, padded.EncodeBatch(texts));

        JsonElement[] pairRows = [.. doc.RootElement.GetProperty("pairs").GetProperty("rows").EnumerateArray()];
        (string, string)[] pairs = [.. pairRows.Select(r =>
        {
            JsonElement c = cases[r.GetProperty("id").GetString()!];
            return (c.GetProperty("text").GetString()!, c.GetProperty("pair").GetString()!);
        })];
        AssertBatch(pairRows, padded.EncodeBatch(pairs));

        Assert.True(doc.RootElement.GetProperty("equivalent").GetBoolean());
        Assert.True(doc.RootElement.GetProperty("pairs").GetProperty("equivalent").GetBoolean());
    }

    private static void AssertBatch(JsonElement[] rows, IReadOnlyList<TokenEncoding> batch)
    {
        Assert.NotEmpty(rows);
        Assert.Equal(rows.Length, batch.Count);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("ids")), batch[i].Ids);
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("attention_mask")), batch[i].AttentionMask);
            Assert.True(rows[i].GetProperty("matches_single").GetBoolean());
        }
    }

    private static TokenEncoding Encode(Tokenizer tokenizer, JsonElement row, bool addSpecialTokens = true)
    {
        string text = row.GetProperty("text").GetString()!;
        return row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!, addSpecialTokens)
            : tokenizer.Encode(text, addSpecialTokens);
    }

    private static Dictionary<string, JsonElement> ReadCases(string profile) =>
        TestAssets.ReadJsonLines(TestAssets.ConformancePath(profile, "tokenizer.jsonl")).ToDictionary(r => r.GetProperty("id").GetString()!);
}
