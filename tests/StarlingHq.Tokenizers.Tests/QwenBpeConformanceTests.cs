using System.Text.Json;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>A declared Qwen byte-level BPE profile and where its fixtures are.</summary>
internal sealed record QwenProfile(string Name, string Key, PinnedAsset Asset)
{
    public static readonly QwenProfile Kev = new("qwen2-bpe-kev", "kev", TestAssets.QwenKevTokenizerJson);
    public static readonly QwenProfile Chat = new("qwen35-bpe-chat", "chat", TestAssets.QwenChatTokenizerJson);

    public static QwenProfile Get(string key) => key switch
    {
        "kev" => Kev,
        "chat" => Chat,
        _ => throw new ArgumentOutOfRangeException(nameof(key)),
    };

    /// <summary>The profile's fixtures, <c>conformance/&lt;profile&gt;/</c>, written by
    /// reference/qwen-bpe/run_qwen_bpe_tokenizer.py.</summary>
    public string FixtureDir => TestAssets.ConformancePath(Name);

    public Tokenizer Load() => TestAssets.LoadTokenizer(Asset);
}

/// <summary>
/// Profiles <c>qwen2-bpe-kev</c> and <c>qwen35-bpe-chat</c>: every case of the fixtures (the M0
/// tokenizer corpus plus the Qwen BPE extension; reference: tokenizers 0.23.2 on the profile asset,
/// which the runner checked equals the live transformers pipeline) must match exactly.
/// </summary>
public class QwenBpeConformanceTests
{
    private const int _expectedCaseCount = 137;

    private static readonly Dictionary<string, Lazy<Dictionary<string, JsonElement>>> _cases = new()
    {
        ["kev"] = new(() => ReadCases(QwenProfile.Kev)),
        ["chat"] = new(() => ReadCases(QwenProfile.Chat)),
    };

    public static TheoryData<string, string> Cases()
    {
        var data = new TheoryData<string, string>();
        foreach ((string profile, Lazy<Dictionary<string, JsonElement>> cases) in _cases)
        {
            foreach (string id in cases.Value.Keys)
            {
                data.Add(profile, id);
            }
        }

        return data;
    }

    [Theory]
    [InlineData("kev")]
    [InlineData("chat")]
    public void FixtureHasEveryCase(string profile)
    {
        Assert.Equal(_expectedCaseCount, _cases[profile].Value.Count);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EncodeMatchesReference(string profile, string caseId)
    {
        Tokenizer tokenizer = QwenProfile.Get(profile).Load();
        JsonElement row = _cases[profile].Value[caseId];
        string text = row.GetProperty("text").GetString()!;
        TokenEncoding actual = row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!)
            : tokenizer.Encode(text);
        FixtureAssert.EncodingMatches(row.GetProperty("encoding"), actual, caseId);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EncodeWithoutSpecialTokensIsTheSame(string profile, string caseId)
    {
        // The ByteLevel post-processor adds no tokens; Kev encodes with add_special_tokens=False.
        Tokenizer tokenizer = QwenProfile.Get(profile).Load();
        JsonElement row = _cases[profile].Value[caseId];
        string text = row.GetProperty("text").GetString()!;
        TokenEncoding actual = row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!, addSpecialTokens: false)
            : tokenizer.Encode(text, addSpecialTokens: false);
        Assert.Equal(FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids")), actual.Ids);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DecodeMatchesReference(string profile, string caseId)
    {
        Tokenizer tokenizer = QwenProfile.Get(profile).Load();
        JsonElement row = _cases[profile].Value[caseId];
        int[] ids = FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids"));
        JsonElement decode = row.GetProperty("decode");
        Assert.Equal(decode.GetProperty("skip_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: true));
        Assert.Equal(decode.GetProperty("keep_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: false));
    }

    [Theory]
    [InlineData("kev")]
    [InlineData("chat")]
    public void PipelineIsTheDeclaredOne(string profile)
    {
        Tokenizer tokenizer = QwenProfile.Get(profile).Load();
        Assert.Equal(
            ["normalizer:NFC", "pre_tokenizer:Split", "pre_tokenizer:ByteLevel", "pre_tokenizer:Sequence", "model:BPE", "added_tokens", "post_processor:ByteLevel", "decoder:ByteLevel"],
            tokenizer.Components);
        Assert.Equal(248044, tokenizer.GetVocabularySize(includeAddedTokens: false));
        Assert.Equal(248077, tokenizer.GetVocabularySize(includeAddedTokens: true));
        Assert.Equal(248044, tokenizer.TokenToId("<|endoftext|>"));
        Assert.Equal(248068, tokenizer.TokenToId("<think>"));
        Assert.Equal("<|audio_pad|>", tokenizer.IdToToken(248076));
    }

    [Fact]
    public void ProfilesDifferOnlyOnCombiningMarks()
    {
        // The \p{M} regex difference shows up on these corpus cases and nowhere else.
        string[] differing = [.. _cases["kev"].Value.Keys.Where(id =>
            !FixtureAssert.Ints(_cases["kev"].Value[id].GetProperty("encoding").GetProperty("ids"))
                .SequenceEqual(FixtureAssert.Ints(_cases["chat"].Value[id].GetProperty("encoding").GetProperty("ids"))))];
        Assert.Equal(["devanagari", "thai", "bpe-devanagari-marks", "bpe-thai-marks", "bpe-arabic-harakat", "bpe-tamil-composing"], differing);
    }

    private static Dictionary<string, JsonElement> ReadCases(QwenProfile profile) =>
        TestAssets.ReadJsonLines(Path.Combine(profile.FixtureDir, "tokenizer.jsonl"))
            .ToDictionary(r => r.GetProperty("id").GetString()!);
}
