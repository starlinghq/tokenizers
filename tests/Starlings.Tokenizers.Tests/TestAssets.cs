using System.Security.Cryptography;
using System.Text.Json;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Locates conformance fixtures (in the repository) and pinned model assets (outside it).
/// </summary>
/// <remarks>
/// Assets live under <c>$STARLING_MODELS/&lt;repo&gt;/&lt;commit&gt;/&lt;file&gt;</c>, defaulting to
/// <c>&lt;repo root&gt;/.models</c>; <c>tools/fetch-assets.sh</c> downloads and verifies
/// them. A missing asset skips the test with the reason, unless <c>STARLING_REQUIRE_ASSETS=1</c>,
/// which turns it into a failure (CI sets it). A present asset with the wrong SHA-256 always fails.
/// </remarks>
internal static class TestAssets
{
    public static readonly PinnedAsset E5TokenizerJson = new(
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "tokenizer.json",
        "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39");

    public static readonly PinnedAsset Gliner2TokenizerJson = new(
        "fastino/gliner2-base-v1",
        "f9634218e53580c56edf0de97ca1a7d3f1c2354e",
        "tokenizer.json",
        "1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16");

    /// <summary>Profile qwen2-bpe-kev: the Kev-0.8B repository tokenizer.json (Kev's runtime pipeline).</summary>
    public static readonly PinnedAsset QwenKevTokenizerJson = new(
        "jaredpalmer/kev-0.8b",
        "9a45d25eb2ab761841196625383fa1dff0e56c1e",
        "tokenizer.json",
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523");

    /// <summary>Profile qwen35-bpe-chat: the pipeline transformers 5.18.0 builds for Qwen/Qwen3.5-0.8B
    /// (derived by tools/fetch-assets.sh, digest pinned).</summary>
    public static readonly PinnedAsset QwenChatTokenizerJson = new(
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
        "transformers-5.18.0/tokenizer.json",
        "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f");

    /// <summary>Profile gemma-bpe-embeddinggemma2: the google/embeddinggemma-2 repository tokenizer.json.</summary>
    public static readonly PinnedAsset EmbeddingGemma2TokenizerJson = new(
        "google/embeddinggemma-2",
        "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "tokenizer.json",
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4");

    public static readonly PinnedAsset MiniLmTokenizerJson = new(
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "tokenizer.json",
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037");

    public static readonly PinnedAsset BgeSmallTokenizerJson = new(
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66");

    public static readonly PinnedAsset MsMarcoTokenizerJson = new(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66");

    /// <summary>Profile whisper-bpe: the openai/whisper-large-v3-turbo GPT-2 byte-level BPE tokenizer.json.</summary>
    public static readonly PinnedAsset WhisperTokenizerJson = new(
        "openai/whisper-large-v3-turbo",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "tokenizer.json",
        "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd");

    /// <summary>Profile qwen3-bpe-embedding: the Qwen/Qwen3-Embedding-0.6B tokenizer.json
    /// (post-processor Sequence[ByteLevel, TemplateProcessing], appends &lt;|endoftext|&gt;).</summary>
    public static readonly PinnedAsset Qwen3EmbeddingTokenizerJson = new(
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "tokenizer.json",
        "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a");

    /// <summary>Profile qwen3-bpe-chat: the Qwen/Qwen3-Reranker-0.6B tokenizer.json (byte-identical to
    /// the Qwen/Qwen3-0.6B file; post-processor ByteLevel, adds no tokens).</summary>
    public static readonly PinnedAsset Qwen3ChatTokenizerJson = new(
        "Qwen/Qwen3-Reranker-0.6B",
        "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "tokenizer.json",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4");

    /// <summary>Profile lfm3-bpe-chat: the LiquidAI/LFM3-1.7B tokenizer.json (Llama 3 split regex,
    /// BPE with ignore_merges; post-processor ByteLevel, adds no tokens).</summary>
    public static readonly PinnedAsset Lfm3ChatTokenizerJson = new(
        "LiquidAI/LFM3-1.7B",
        "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "tokenizer.json",
        "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701");

    /// <summary>Profile clover-bpe-chat: the Diffuspace/Clover-35B-A3B-Preview tokenizer.json (Qwen2
    /// split regex; vocabulary id 151669 renamed to the special added token &lt;|mask|&gt;).</summary>
    public static readonly PinnedAsset CloverChatTokenizerJson = new(
        "Diffuspace/Clover-35B-A3B-Preview",
        "4b0402b65e523d0183a14c431955438714cb16ff",
        "tokenizer.json",
        "7c62ba7d8bb13291c94459896cd27537858beb58a88a037b09a150e213420988");

    private static readonly Lazy<string> _lazyRepoRoot = new(FindRepoRoot);
    private static readonly Dictionary<PinnedAsset, Tokenizer> _loaded = [];
    private static readonly Lock _loadLock = new();

    public static string RepoRoot => _lazyRepoRoot.Value;

    /// <summary>A profile's conformance fixtures: <c>conformance/&lt;profile&gt;/</c> (protected, byte-exact).</summary>
    public static string ConformancePath(params string[] parts) =>
        Path.Combine([RepoRoot, "conformance", .. parts]);

    /// <summary>Loads (once) the tokenizer for a pinned asset, or skips the test.</summary>
    public static Tokenizer LoadTokenizer(PinnedAsset asset)
    {
        lock (_loadLock)
        {
            if (!_loaded.TryGetValue(asset, out Tokenizer? tokenizer))
            {
                tokenizer = Tokenizer.FromFile(RequireAsset(asset));
                _loaded[asset] = tokenizer;
            }

            return tokenizer;
        }
    }

    public static string TestDataPath(params string[] parts) =>
        Path.Combine([RepoRoot, "tests", "Starlings.Tokenizers.Tests", "TestData", .. parts]);

    public static string RequireAsset(PinnedAsset asset)
    {
        string modelsRoot = Environment.GetEnvironmentVariable("STARLING_MODELS") is { Length: > 0 } env
            ? env
            : Path.Combine(RepoRoot, ".models");
        string path = Path.Combine(modelsRoot, asset.Repo, asset.Commit, asset.Path);
        if (!File.Exists(path))
        {
            string reason = $"Pinned asset {asset.Repo}@{asset.Commit}/{asset.Path} not found at {path}. "
                + "Run tools/fetch-assets.sh or set STARLING_MODELS.";
            if (Environment.GetEnvironmentVariable("STARLING_REQUIRE_ASSETS") == "1")
            {
                Assert.Fail(reason);
            }

            Assert.Skip(reason);
        }

        using FileStream stream = File.OpenRead(path);
        string actual = Convert.ToHexStringLower(SHA256.HashData(stream));
        Assert.True(actual == asset.Sha256, $"{path} has SHA-256 {actual}, expected {asset.Sha256}.");
        return path;
    }

    /// <summary>Reads JSON Lines, splitting on LF only (corpus text contains U+0085 and U+2028).</summary>
    public static List<JsonElement> ReadJsonLines(string path)
    {
        string text = File.ReadAllText(path);
        var rows = new List<JsonElement>();
        foreach (string line in text.Split('\n'))
        {
            if (line.Length > 0)
            {
                rows.Add(JsonDocument.Parse(line).RootElement.Clone());
            }
        }

        return rows;
    }

    private static string FindRepoRoot()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir, "Tokenizers.slnx"))
                && Directory.Exists(Path.Combine(dir, "conformance")))
            {
                return dir;
            }

            dir = Path.GetDirectoryName(dir);
        }

        throw new InvalidOperationException("Repository root (with Tokenizers.slnx and conformance/) not found above " + AppContext.BaseDirectory);
    }
}

internal sealed record PinnedAsset(string Repo, string Commit, string Path, string Sha256);
