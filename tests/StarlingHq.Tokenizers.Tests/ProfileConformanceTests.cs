using System.Text.Json;
using System.Text.RegularExpressions;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>
/// Data-driven conformance for every profile in <c>profiles/</c> whose <c>conformance.fixtures</c>
/// directory holds a fixture-format-1 <c>tokenizer.jsonl</c> (as written by
/// <c>reference/tokenizer-json/run_tokenizer_json.py</c>): every case must match the reference exactly
/// (ids, tokens, type ids, masks, sequence ids, offsets in all recorded units, decode with and without
/// special tokens, and the truncated encoding where recorded). A new profile needs no test code.
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
            new HashSet<string> { "gemma-bpe-embeddinggemma2", "qwen2-bpe-kev", "qwen35-bpe-chat", "xlmr-unigram-e5" },
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

    private static Profile Get(string name) => _profiles.Value.Single(p => p.Name == name);

    private static TokenEncoding Encode(Tokenizer tokenizer, JsonElement row)
    {
        string text = row.GetProperty("text").GetString()!;
        return row.TryGetProperty("pair", out JsonElement pair)
            ? tokenizer.Encode(text, pair.GetString()!)
            : tokenizer.Encode(text);
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

        public Tokenizer Load() => TestAssets.LoadTokenizer(Asset);
    }
}
