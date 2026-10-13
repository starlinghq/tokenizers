using System.Text.Json;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>
/// Stretch check of the second Unigram asset, GLiNER2's DeBERTa-v3 <c>tokenizer.json</c>, against
/// the proposed fixtures in <c>conformance/gliner2-base-v1-unigram/</c>
/// (reference: tokenizers 0.22.2 via transformers 4.57.6 fast). The profile stays untested until
/// a full whole-text fixture exists; these tests cover what the fixtures record.
/// </summary>
public class Gliner2ConformanceTests
{
    private const int _sepTextId = 128002;
    private const int _unknownId = 3;
    private static readonly string _fixtureDir = TestAssets.ConformancePath("gliner2-base-v1-unigram");
    private static readonly Lazy<Dictionary<string, JsonElement>> _rows = new(() =>
        TestAssets.ReadJsonLines(Path.Combine(_fixtureDir, "gliner2-tokenizer.jsonl"))
            .ToDictionary(r => r.GetProperty("id").GetString()!));

    public static TheoryData<string> CaseIds()
    {
        var data = new TheoryData<string>();
        foreach (string id in _rows.Value.Keys)
        {
            data.Add(id);
        }

        return data;
    }

    [Fact]
    public void FixtureHasEveryCase()
    {
        Assert.Equal(46, _rows.Value.Count);
    }

    /// <summary>GLiNER2 tokenizes each lower-cased word alone, without special tokens.</summary>
    [Theory]
    [MemberData(nameof(CaseIds))]
    public void PerWordSubwordsMatchReference(string caseId)
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.Gliner2TokenizerJson);
        JsonElement row = _rows.Value[caseId];
        foreach (string splitter in new[] { "whitespace_splitter", "char_splitter" })
        {
            if (!row.TryGetProperty(splitter, out JsonElement split))
            {
                continue;
            }

            var textIds = new List<int>();
            foreach (JsonElement word in split.GetProperty("words").EnumerateArray())
            {
                string text = word.GetProperty("word").GetString()!;
                string[] expected = [.. word.GetProperty("subwords").EnumerateArray().Select(e => e.GetString()!)];
                TokenEncoding actual = tokenizer.Encode(text, addSpecialTokens: false);
                Assert.True(
                    expected.SequenceEqual(actual.Tokens),
                    $"{caseId}/{splitter}: word {JsonSerializer.Serialize(text)} expected [{string.Join(", ", expected)}] got [{string.Join(", ", actual.Tokens)}]");

                // GLiNER2 maps subword strings with convert_tokens_to_ids: unknown strings become [UNK].
                textIds.AddRange(expected.Select(t => tokenizer.TokenToId(t) ?? _unknownId));
            }

            int[] inputIds = [.. split.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32())];
            int sep = Array.IndexOf(inputIds, _sepTextId);
            Assert.True(sep >= 0, $"{caseId}/{splitter}: no [SEP_TEXT] in input_ids");
            Assert.Equal(inputIds[(sep + 1)..], textIds);
        }
    }

    /// <summary>Whole-text encodes recorded in the implementation comparison (reference ids).</summary>
    [Fact]
    public void WholeTextReferenceIdsMatch()
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.Gliner2TokenizerJson);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(_fixtureDir, "tokenizer-implementations.transformers-4.json")));
        var corpus = TestAssets.ReadJsonLines(TestAssets.ConformancePath("corpus", "tokenizer-corpus-v1.jsonl"))
            .ToDictionary(r => r.GetProperty("id").GetString()!, r => r.GetProperty("text").GetString()!);

        int checkedCases = 0;
        var seen = new HashSet<string>();
        foreach (JsonProperty implementation in doc.RootElement.GetProperty("differences").EnumerateObject())
        {
            foreach (JsonElement diff in implementation.Value.EnumerateArray())
            {
                string id = diff.GetProperty("id").GetString()!;
                if (!seen.Add(id))
                {
                    continue;
                }

                TokenEncoding actual = tokenizer.Encode(corpus[id]);
                Assert.Equal(FixtureAssert.Ints(diff.GetProperty("reference_ids")), actual.Ids);
                Assert.Equal(diff.GetProperty("reference_tokens").EnumerateArray().Select(e => e.GetString()!), actual.Tokens);
                checkedCases++;
            }
        }

        Assert.Equal(10, checkedCases);
    }
}
