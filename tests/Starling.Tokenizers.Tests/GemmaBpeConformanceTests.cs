using System.Text;
using System.Text.Json;

namespace Starling.Tokenizers.Tests;

/// <summary>
/// Profile <c>gemma-bpe-embeddinggemma2</c>: every case of the accepted protected fixtures in
/// <c>conformance/gemma-bpe-embeddinggemma2/</c> (reference: the pinned
/// <c>tokenizer.json</c> read by tokenizers 0.23.3) must match exactly: ids, tokens, type ids, masks,
/// sequence ids, offsets in all three recorded units, decode with and without special tokens, and
/// the padded batch.
/// </summary>
public class GemmaBpeConformanceTests
{
    private const int _expectedCaseCount = 49;
    private static readonly string _fixtureDir = TestAssets.ConformancePath("gemma-bpe-embeddinggemma2");
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

    private static Tokenizer Load() => TestAssets.LoadTokenizer(TestAssets.EmbeddingGemma2TokenizerJson);

    [Fact]
    public void FixtureHasEveryCase()
    {
        Assert.Equal(_expectedCaseCount, _cases.Value.Count);
        Assert.Equal(3, _cases.Value.Values.Count(r => r.TryGetProperty("pair", out _)));
    }

    [Fact]
    public void AssetLoadsTheExpectedPipeline()
    {
        Tokenizer tokenizer = Load();
        Assert.Equal(
            [
                "normalizer:Replace", "pre_tokenizer:Split", "model:BPE", "added_tokens", "post_processor:TemplateProcessing",
                "decoder:Replace", "decoder:ByteFallback", "decoder:Fuse", "decoder:Sequence",
            ],
            tokenizer.Components);
        Assert.Equal(262144, tokenizer.GetVocabularySize(includeAddedTokens: false));
        Assert.Equal(262144, tokenizer.GetVocabularySize(includeAddedTokens: true));
        for (int b = 0; b < 256; b++)
        {
            Assert.NotNull(tokenizer.TokenToId($"<0x{b:X2}>"));
        }
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void EncodeMatchesReference(string caseId)
    {
        JsonElement row = _cases.Value[caseId];
        TokenEncoding actual = Encode(Load(), row);
        JsonElement expected = row.GetProperty("encoding");
        FixtureAssert.EncodingMatches(expected, actual, caseId);

        // The third recorded unit: UTF-8 byte offsets, derived from the scalar offsets the library
        // returns (special tokens (0, 0)).
        string text = row.GetProperty("text").GetString()!;
        string? pair = row.TryGetProperty("pair", out JsonElement p) ? p.GetString() : null;
        JsonElement[] bytes = [.. expected.GetProperty("offsets").GetProperty("utf8-bytes").EnumerateArray()];
        Assert.Equal(actual.Count, bytes.Length);
        for (int i = 0; i < actual.Count; i++)
        {
            string source = actual.SequenceIds[i] == 1 ? pair! : text;
            TokenOffset expectedBytes = new(bytes[i][0].GetInt32(), bytes[i][1].GetInt32());
            Assert.Equal(expectedBytes, ScalarToUtf8(source, actual.ScalarOffsets[i]));
        }
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void EncodeWithoutSpecialTokensDropsOnlyTheTemplateTokens(string caseId)
    {
        Tokenizer tokenizer = Load();
        JsonElement row = _cases.Value[caseId];
        string text = row.GetProperty("text").GetString()!;
        string? pair = row.TryGetProperty("pair", out JsonElement p) ? p.GetString() : null;
        TokenEncoding with = pair is null ? tokenizer.Encode(text) : tokenizer.Encode(text, pair);
        TokenEncoding without = pair is null ? tokenizer.Encode(text, addSpecialTokens: false) : tokenizer.Encode(text, pair, addSpecialTokens: false);
        int[] inner = [.. with.Ids.Where((_, i) => !(with.SpecialTokensMask[i] == 1 && (with.Ids[i] == 2 || with.Ids[i] == 1) && with.ScalarOffsets[i] == default))];
        Assert.Equal(inner, without.Ids);
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void DecodeMatchesReference(string caseId)
    {
        Tokenizer tokenizer = Load();
        JsonElement row = _cases.Value[caseId];
        int[] ids = FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids"));
        JsonElement decode = row.GetProperty("decode");
        Assert.Equal(decode.GetProperty("skip_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: true));
        Assert.Equal(decode.GetProperty("keep_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: false));
    }

    [Fact]
    public void ByteFallbackIsExercisedByTheFixtures()
    {
        // The spike lists 9 cases with byte-fallback tokens; the conformance run must cover them.
        string[] withByteTokens = [.. _cases.Value
            .Where(kv => kv.Value.GetProperty("encoding").GetProperty("tokens").EnumerateArray().Any(t => t.GetString()!.StartsWith("<0x", StringComparison.Ordinal)))
            .Select(kv => kv.Key)
            .Order(StringComparer.Ordinal)];
        Assert.Equal(
            ["cjk-ext-b", "combining-stack", "control-chars", "german-turkish-case", "math-alnum-astral", "nfkc-compat", "unknown-char", "ws-tab-newline", "ws-unicode-spaces"],
            withByteTokens);
    }

    [Fact]
    public void BatchMatchesReference()
    {
        Tokenizer tokenizer = Load();
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

        Assert.Equal(6, rows.Length);
        Assert.Equal(rows.Length, batch.Count);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("ids")), batch[i].Ids);
            Assert.Equal(FixtureAssert.Ints(rows[i].GetProperty("attention_mask")), batch[i].AttentionMask);

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

    private static TokenOffset ScalarToUtf8(string text, TokenOffset scalar)
    {
        int ToUtf8(int s)
        {
            int bytes = 0;
            int count = 0;
            foreach (Rune r in text.EnumerateRunes())
            {
                if (count == s)
                {
                    break;
                }

                bytes += r.Utf8SequenceLength;
                count++;
            }

            return bytes;
        }

        return new TokenOffset(ToUtf8(scalar.Start), ToUtf8(scalar.End));
    }
}
