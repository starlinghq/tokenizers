using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Data-driven conformance for every profile in <c>profiles/</c> whose <c>conformance.fixtures</c>
/// directory holds a fixture-format-1 <c>tokenizer.jsonl</c> (as written by
/// <c>reference/tokenizer-json/run_tokenizer_json.py</c>): every case must match the reference exactly
/// (ids, tokens, type ids, masks, sequence ids, offsets in all recorded units, decode with and without
/// special tokens, and the truncated encoding where recorded; with <c>--without-special-tokens</c>, the
/// same with <c>add_special_tokens</c> false). A new profile needs no test code.
/// </summary>
/// <remarks>
/// Profile-specific test classes keep the checks a generic test cannot make (declared pipeline,
/// vocabulary sizes, padded batches, cross-profile differences). The GLiNER2 per-word fixtures use
/// another format and are covered by <see cref="Gliner2ConformanceTests"/>.
/// </remarks>
public sealed partial class ProfileConformanceTests
{
    private static readonly Lazy<List<Profile>> _profiles = new(LoadProfiles);

    public static TheoryData<string, string> Cases()
    {
        var data = new TheoryData<string, string>();
        foreach (Profile profile in _profiles.Value.Where(p => p.HasCaseFixtures))
        {
            foreach (string id in profile.Cases.Value.Keys)
            {
                data.Add(profile.Name, id);
            }
        }

        return data;
    }

    public static TheoryData<string, string> TruncatedCases()
    {
        var data = new TheoryData<string, string>();
        foreach (Profile profile in _profiles.Value.Where(p => p.HasCaseFixtures))
        {
            foreach ((string id, JsonElement row) in profile.Cases.Value)
            {
                if (row.TryGetProperty("truncated", out _))
                {
                    data.Add(profile.Name, id);
                }
            }
        }

        return data;
    }

    public static TheoryData<string, string> WithoutSpecialTokensCases()
    {
        var data = new TheoryData<string, string>();
        foreach (Profile profile in _profiles.Value.Where(p => p.HasCaseFixtures))
        {
            foreach ((string id, JsonElement row) in profile.Cases.Value)
            {
                if (row.TryGetProperty("without_special_tokens", out _))
                {
                    data.Add(profile.Name, id);
                }
            }
        }

        return data;
    }

    /// <summary>Profiles whose fixtures were written by the generic runner, which records the asset
    /// truncation and padding it cleared in <c>environment.json</c> (<c>options.cleared_asset_settings</c>).</summary>
    public static TheoryData<string> GenericRunnerProfiles()
    {
        var data = new TheoryData<string>();
        foreach (Profile profile in _profiles.Value.Where(p => p.HasCaseFixtures && p.Environment.Value is { } env
            && env.GetProperty("runner").GetString() == "reference/tokenizer-json/run_tokenizer_json.py"))
        {
            data.Add(profile.Name);
        }

        return data;
    }

    [Fact]
    public void EveryProfileDeclaresExistingFixtures()
    {
        Assert.NotEmpty(_profiles.Value);
        foreach (Profile profile in _profiles.Value)
        {
            Assert.True(Directory.Exists(profile.FixtureDir), $"{profile.Name}: conformance.fixtures {profile.FixtureDir} does not exist.");
            Assert.True(
                Directory.EnumerateFiles(profile.FixtureDir, "*.jsonl").Any(),
                $"{profile.Name}: {profile.FixtureDir} has no fixture file.");
        }
    }

    [Fact]
    public void CaseFixturesCoverTheProfilesThatHaveThem()
    {
        // Guards against a format change silently dropping a profile from the data-driven run.
        Assert.Superset(
            new HashSet<string> { "gemma-bpe-embeddinggemma2", "lfm3-bpe-chat", "phi4-bpe-chat", "qwen2-bpe-kev", "qwen35-bpe-chat", "qwen3-bpe-chat", "qwen3-bpe-embedding", "xlmr-unigram-e5", "xlmr-unigram-paraphrase-multilingual", "modernbert-bpe-gte-reranker" },
            _profiles.Value.Where(p => p.HasCaseFixtures).Select(p => p.Name).ToHashSet());
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EncodeMatchesReference(string profile, string caseId)
    {
        Profile p = Get(profile);
        JsonElement row = p.Cases.Value[caseId];
        FixtureAssert.EncodingMatches(row.GetProperty("encoding"), Encode(p.Load(), row), $"{profile}/{caseId}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DecodeMatchesReference(string profile, string caseId)
    {
        Profile p = Get(profile);
        Tokenizer tokenizer = p.Load();
        JsonElement row = p.Cases.Value[caseId];
        int[] ids = FixtureAssert.Ints(row.GetProperty("encoding").GetProperty("ids"));
        JsonElement decode = row.GetProperty("decode");
        Assert.Equal(decode.GetProperty("skip_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: true));
        Assert.Equal(decode.GetProperty("keep_special_tokens").GetString(), tokenizer.Decode(ids, skipSpecialTokens: false));
    }

    [Theory]
    [MemberData(nameof(TruncatedCases))]
    public void TruncatedEncodeMatchesReference(string profile, string caseId)
    {
        Profile p = Get(profile);
        JsonElement row = p.Cases.Value[caseId];
        JsonElement truncated = row.GetProperty("truncated");

        Assert.Equal("longest_first", truncated.GetProperty("strategy").GetString());
        Assert.Equal("right", truncated.GetProperty("direction").GetString());
        Tokenizer truncating = p.Load().WithTruncation(new TruncationOptions(truncated.GetProperty("max_length").GetInt32()));
        FixtureAssert.EncodingMatches(truncated.GetProperty("encoding"), Encode(truncating, row), $"{profile}/{caseId}/truncated");
    }

    [Theory]
    [MemberData(nameof(WithoutSpecialTokensCases))]
    public void EncodeWithoutSpecialTokensMatchesReference(string profile, string caseId)
    {
        Profile p = Get(profile);
        JsonElement row = p.Cases.Value[caseId];
        JsonElement plain = row.GetProperty("without_special_tokens");
        FixtureAssert.EncodingMatches(plain.GetProperty("encoding"), Encode(p.Load(), row, addSpecialTokens: false), $"{profile}/{caseId}/without_special_tokens");

        if (plain.TryGetProperty("truncated", out JsonElement truncated))
        {
            Assert.Equal("longest_first", truncated.GetProperty("strategy").GetString());
            Assert.Equal("right", truncated.GetProperty("direction").GetString());
            Tokenizer truncating = p.Load().WithTruncation(new TruncationOptions(truncated.GetProperty("max_length").GetInt32()));
            FixtureAssert.EncodingMatches(
                truncated.GetProperty("encoding"), Encode(truncating, row, addSpecialTokens: false), $"{profile}/{caseId}/without_special_tokens/truncated");
        }
    }

    /// <summary><see cref="Profile.Load"/> clears the asset's truncation and padding for every profile. The
    /// settings cleared must be exactly those the runner cleared and recorded: none when
    /// <c>cleared_asset_settings</c> is absent (the runner refuses such an asset without the flag).</summary>
    [Theory]
    [MemberData(nameof(GenericRunnerProfiles))]
    public void ClearedAssetSettingsMatchTheRunner(string profile)
    {
        Profile p = Get(profile);
        Tokenizer asLoaded = TestAssets.LoadTokenizer(p.Asset);
        JsonElement cleared = default;
        if (!p.Environment.Value!.Value.TryGetProperty("options", out JsonElement options)
            || !options.TryGetProperty("cleared_asset_settings", out cleared))
        {
            Assert.Null(asLoaded.Truncation);
            Assert.Null(asLoaded.Padding);
            return;
        }

        Assert.Equal(ExpectedTruncation(cleared.GetProperty("truncation")), asLoaded.Truncation);
        Assert.Equal(ExpectedPadding(cleared.GetProperty("padding")), asLoaded.Padding);
    }

    /// <summary>Maps the reference's <c>Tokenizer.truncation</c> dictionary to <see cref="TruncationOptions"/>.</summary>
    private static TruncationOptions? ExpectedTruncation(JsonElement t)
    {
        if (t.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        Assert.Equal(0, t.GetProperty("stride").GetInt32());
        return new TruncationOptions(
            t.GetProperty("max_length").GetInt32(),
            t.GetProperty("strategy").GetString() switch
            {
                "longest_first" => TruncationStrategy.LongestFirst,
                "only_first" => TruncationStrategy.OnlyFirst,
                "only_second" => TruncationStrategy.OnlySecond,
                string other => throw new InvalidOperationException($"Unknown truncation strategy {other}."),
                null => throw new InvalidOperationException("Truncation strategy missing."),
            },
            t.GetProperty("direction").GetString() == "left" ? TruncationDirection.Left : TruncationDirection.Right);
    }

    /// <summary>Maps the reference's <c>Tokenizer.padding</c> dictionary (<c>length</c> null means
    /// BatchLongest) to <see cref="PaddingOptions"/>.</summary>
    private static PaddingOptions? ExpectedPadding(JsonElement pad)
    {
        if (pad.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        static int? NullableInt(JsonElement e) => e.ValueKind == JsonValueKind.Null ? null : e.GetInt32();
        return new PaddingOptions(
            pad.GetProperty("pad_id").GetInt32(),
            pad.GetProperty("pad_token").GetString()!,
            NullableInt(pad.GetProperty("length")),
            pad.GetProperty("direction").GetString() == "left" ? PaddingDirection.Left : PaddingDirection.Right,
            pad.GetProperty("pad_type_id").GetInt32(),
            NullableInt(pad.GetProperty("pad_to_multiple_of")));
    }

    private static Profile Get(string name) => _profiles.Value.Single(p => p.Name == name);

    private static TokenEncoding Encode(Tokenizer tokenizer, JsonElement row, bool addSpecialTokens = true)
    {
        string text = row.GetProperty("text").GetString()!;
        return row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!, addSpecialTokens)
            : tokenizer.Encode(text, addSpecialTokens);
    }

    private static List<Profile> LoadProfiles()
    {
        var profiles = new List<Profile>();
        foreach (string file in Directory.GetFiles(Path.Combine(TestAssets.RepoRoot, "profiles"), "*.yaml").Order(StringComparer.Ordinal))
        {
            string yaml = File.ReadAllText(file);
            string name = TopLevel(yaml, "profile", file);
            var asset = new PinnedAsset(
                Nested(yaml, "asset", "repo", file),
                Nested(yaml, "asset", "revision", file),
                Nested(yaml, "asset", "file", file),
                Nested(yaml, "asset", "sha256", file));
            string fixtures = Nested(yaml, "conformance", "fixtures", file);
            profiles.Add(new Profile(name, asset, Path.Combine(TestAssets.RepoRoot, fixtures.TrimEnd('/'))));
        }

        return profiles;
    }

    private static string TopLevel(string yaml, string key, string file)
    {
        Match m = Regex.Match(yaml, $@"^{key}:\s*(\S+)", RegexOptions.Multiline);
        Assert.True(m.Success, $"{Path.GetFileName(file)}: {key} missing.");
        return m.Groups[1].Value;
    }

    /// <summary>Reads <c>key: value</c> from a top-level block (<c>block:</c> followed by indented lines).</summary>
    private static string Nested(string yaml, string block, string key, string file)
    {
        Match b = Regex.Match(yaml, $@"^{block}:[^\n]*\n((?:[ \t]+[^\n]*\n?|[ \t]*\n)+)", RegexOptions.Multiline);
        Assert.True(b.Success, $"{Path.GetFileName(file)}: {block}: block missing.");
        Match m = Regex.Match(b.Groups[1].Value, $@"^[ \t]+{key}:\s*(\S+)", RegexOptions.Multiline);
        Assert.True(m.Success, $"{Path.GetFileName(file)}: {block}.{key} missing.");
        return m.Groups[1].Value;
    }

    private sealed record Profile(string Name, PinnedAsset Asset, string FixtureDir)
    {
        public string CaseFile => Path.Combine(FixtureDir, "tokenizer.jsonl");

        /// <summary>True if the fixtures are per-case rows of format 1 (id, text, encoding, decode).</summary>
        public bool HasCaseFixtures => File.Exists(CaseFile);

        public Lazy<Dictionary<string, JsonElement>> Cases { get; } = new(() =>
            TestAssets.ReadJsonLines(Path.Combine(FixtureDir, "tokenizer.jsonl")).ToDictionary(r => r.GetProperty("id").GetString()!));

        /// <summary>The fixtures' <c>environment.json</c>, or null if there is none.</summary>
        public Lazy<JsonElement?> Environment { get; } = new(() =>
        {
            string path = Path.Combine(FixtureDir, "environment.json");
            if (!File.Exists(path))
            {
                return null;
            }

            using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
            return doc.RootElement.Clone();
        });

        /// <summary>The profile's tokenizer without the asset's own truncation and padding: the rows are
        /// untruncated, unpadded encodings, and the runner clears both the same way
        /// (<c>--clear-asset-truncation-padding</c>). A no-op for assets that declare neither.
        /// <see cref="ClearedAssetSettingsMatchTheRunner"/> checks what is cleared against the runner's record.</summary>
        public Tokenizer Load() => TestAssets.LoadTokenizer(Asset).WithTruncation(null).WithPadding(null);
    }
}
