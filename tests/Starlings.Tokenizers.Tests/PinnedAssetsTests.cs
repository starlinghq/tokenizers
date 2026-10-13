using System.Reflection;
using System.Text.RegularExpressions;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Keeps the three places that name a pinned asset in step: the <see cref="TestAssets"/> pins, the
/// download list in <c>tools/pinned-assets.txt</c> (derived assets: <c>tools/fetch-assets.sh</c>), and the
/// <c>asset:</c> block of every profile under <c>profiles/</c>. Needs no asset.
/// </summary>
public sealed partial class PinnedAssetsTests
{
    private static readonly PinnedAsset[] _pins = [.. typeof(TestAssets)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Where(f => f.FieldType == typeof(PinnedAsset))
        .Select(f => (PinnedAsset)f.GetValue(null)!)];

    [Fact]
    public void EveryTestPinIsFetched()
    {
        string list = File.ReadAllText(Path.Combine(TestAssets.RepoRoot, "tools", "pinned-assets.txt"));
        string script = File.ReadAllText(Path.Combine(TestAssets.RepoRoot, "tools", "fetch-assets.sh"));
        foreach (PinnedAsset pin in _pins)
        {
            string line = $"{pin.Repo}|{pin.Commit}|{pin.Path}|{pin.Sha256}";
            Assert.True(
                list.Split('\n').Any(l => l.Trim() == line) || script.Contains(line, StringComparison.Ordinal),
                $"{line} is pinned in TestAssets.cs but not in tools/pinned-assets.txt or tools/fetch-assets.sh.");
        }
    }

    [Fact]
    public void EveryFetchedAssetIsPinnedForTests()
    {
        string[] lines = [.. File.ReadAllLines(Path.Combine(TestAssets.RepoRoot, "tools", "pinned-assets.txt"))
            .Select(l => l.Trim())
            .Where(l => l.Length > 0 && !l.StartsWith('#'))];
        Assert.NotEmpty(lines);
        foreach (string line in lines)
        {
            string[] parts = line.Split('|');
            Assert.True(parts.Length == 4, $"tools/pinned-assets.txt: '{line}' is not repo|commit|path|sha256.");
            Assert.Matches(FullSha1(), parts[1]);
            Assert.Matches(Sha256(), parts[3]);
            Assert.Contains(new PinnedAsset(parts[0], parts[1], parts[2], parts[3]), _pins);
        }
    }

    [Fact]
    public void EveryProfileAssetIsPinnedForTests()
    {
        string[] profiles = Directory.GetFiles(Path.Combine(TestAssets.RepoRoot, "profiles"), "*.yaml");
        Assert.NotEmpty(profiles);
        foreach (string file in profiles)
        {
            string text = File.ReadAllText(file);
            string repo = Field(text, "repo", file);
            string revision = Field(text, "revision", file);
            string path = Field(text, "file", file);
            string sha256 = Field(text, "sha256", file);
            Assert.True(
                _pins.Contains(new PinnedAsset(repo, revision, path, sha256)),
                $"{Path.GetFileName(file)}: asset {repo}@{revision}/{path} ({sha256}) has no pin in TestAssets.cs.");
        }
    }

    /// <summary>Reads <c>key: value</c> from the profile's top-level <c>asset:</c> block.</summary>
    private static string Field(string yaml, string key, string file)
    {
        Match block = AssetBlock().Match(yaml);
        Assert.True(block.Success, $"{Path.GetFileName(file)} has no asset: block.");
        Match field = Regex.Match(block.Groups[1].Value, $@"^\s+{key}:\s*(\S+)", RegexOptions.Multiline);
        Assert.True(field.Success, $"{Path.GetFileName(file)}: asset.{key} missing.");
        return field.Groups[1].Value;
    }

    [GeneratedRegex(@"^asset:\s*(?:#.*)?\n((?:[ \t]+.*\n?)+)", RegexOptions.Multiline)]
    private static partial Regex AssetBlock();

    [GeneratedRegex("^[0-9a-f]{40}$")]
    private static partial Regex FullSha1();

    [GeneratedRegex("^[0-9a-f]{64}$")]
    private static partial Regex Sha256();
}
