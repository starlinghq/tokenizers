using System.Text.RegularExpressions;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Keeps the GitHub Actions under <c>.github/workflows/</c> pinned the way <c>.github/dependabot.yml</c> expects:
/// every <c>uses:</c> names a full 40-character commit with its release in a trailing comment, and an action
/// has one pin across all workflows, so a Dependabot bump cannot leave a job on the old commit. Needs no asset.
/// </summary>
public sealed partial class WorkflowPinsTests
{
    private static readonly (string File, int Line, string Text)[] _uses = [.. Directory
        .GetFiles(Path.Combine(TestAssets.RepoRoot, ".github", "workflows"), "*.yml")
        .Order(StringComparer.Ordinal)
        .SelectMany(file => File.ReadAllLines(file)
            .Select((text, index) => (File: Path.GetFileName(file), Line: index + 1, Text: text.Trim())))
        .Where(l => UsesLine().IsMatch(l.Text))];

    [Fact]
    public void EveryActionIsPinnedByFullCommitSha()
    {
        Assert.NotEmpty(_uses);
        foreach ((string file, int line, string text) in _uses)
        {
            Assert.True(
                PinnedUses().IsMatch(text),
                $"{file}:{line}: '{text}' is not 'owner/repo@<40-hex commit> # vX.Y.Z'.");
        }
    }

    [Fact]
    public void EachActionHasOnePinAcrossWorkflows()
    {
        Assert.NotEmpty(_uses);
        foreach (IGrouping<string, Match> action in _uses
            .Select(l => UsesLine().Match(l.Text))
            .GroupBy(m => m.Groups["action"].Value, StringComparer.Ordinal))
        {
            string[] pins = [.. action
                .Select(m => m.Groups["pin"].Value.Trim())
                .Distinct(StringComparer.Ordinal)];
            Assert.True(
                pins.Length == 1,
                $"{action.Key} is pinned {pins.Length} ways across .github/workflows: {string.Join(", ", pins)}.");
        }
    }

    [GeneratedRegex(@"^(?:-\s+)?uses:\s*(?<action>[^@\s]+)@(?<pin>.*)$")]
    private static partial Regex UsesLine();

    [GeneratedRegex(@"^(?:-\s+)?uses:\s*(?<action>[\w.-]+/[\w./-]+)@(?<commit>[0-9a-f]{40})\s+#\s*(?<version>v\d+\.\d+\.\d+)\s*$")]
    private static partial Regex PinnedUses();
}
