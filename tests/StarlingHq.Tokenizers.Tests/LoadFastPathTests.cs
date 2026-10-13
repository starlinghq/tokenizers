using System.Text;
using System.Text.Json;
using StarlingHq.Tokenizers.Pipeline;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>
/// The load fast path (vocabulary read from the bytes, double-array trie built from sorted keys
/// without the DAWG pass) must give the same tokenizer as the document-model path, and must leave
/// every error unchanged. The document path is the pre-0.1.0-preview.2 loader; it is forced with
/// <c>useVocabularyFastPath: false</c>.
/// </summary>
public class LoadFastPathTests
{
    private const string _prefix = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": null,
          "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
          "post_processor": null,
          "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
        """;

    private const string _vocab = """[["<unk>", 0.0], ["▁a", -1.0], ["b", -2.0], ["▁", -3.0], ["▁b", -9.0], ["▁ab", -2.5]]""";

    public static TheoryData<string, string, Mode> Variants() => new()
    {
        { "plain", Doc(_vocab), Mode.Fast },
        { "whitespace", Doc("[ \n\t[ \"<unk>\" ,\r\n0.0 ] ,[\"▁a\",-1.0]\t,\n[ \"b\" , -2 ] ]"), Mode.Fast },
        { "empty-vocab", Doc("[]", unk: "null"), Mode.Fast },
        { "empty-vocab-spaced", Doc("[ \n ]", unk: "null"), Mode.Fast },
        { "vocab-first", _prefix + "\"model\": {\"vocab\": " + _vocab + ", \"type\": \"Unigram\", \"unk_id\": 0}}", Mode.Fast },
        { "model-first", "{\"model\": {\"type\": \"Unigram\", \"unk_id\": 0, \"vocab\": " + _vocab + "}, " + _prefix.TrimStart()[1..].TrimEnd().TrimEnd(',') + "}", Mode.Fast },
        { "escaped-pieces", Doc("""[["<unk>", 0.0], ["▁a", -1.0], ["\"q\"", -2.0], ["a\\b", -3.0], ["\/", -4.0], ["😀", -5.0], ["\t", -6.0]]"""), Mode.Fast },
        { "unicode-escape", Doc("[[\"<unk>\", 0.0], [\"\\u2581\\u00e9\", -1.0], [\"\\ud83d\\ude00\", -2.0]]"), Mode.Fast },
        { "escaped-nul", Doc("""[["<unk>", 0.0], ["a\u0000", -1.0]]"""), Mode.Any },
        { "empty-piece", Doc("""[["<unk>", 0.0], ["", -1.0]]"""), Mode.Any },
        { "duplicate-pieces", Doc("""[["<unk>", 0.0], ["x", -1.0], ["y", -2.0], ["x", -3.0], ["x", -4.0], ["xy", -5.0]]"""), Mode.Fast },
        { "prefix-pieces", Doc("""[["<unk>", 0.0], ["abcdefghij", -1.0], ["abcdefgh", -2.0], ["abcdefghi", -3.0], ["abcdefghé", -4.0], ["a", -5.0], ["ab", -6.0]]"""), Mode.Fast },
        { "numbers", Doc("[[\"<unk>\", -0], [\"a\", 1e5], [\"b\", 1E-5], [\"c\", -1.5e+3], [\"d\", 0.1000000000000000055511151231257827], [\"e\", 12345678901234567890123], [\"f\", -0.0], [\"g\", 5e-324], [\"h\", 2.2250738585072011e-308], [\"i\", 1e-400]]"), Mode.Fast },
        { "number-out-of-range", Doc("[[\"<unk>\", 0], [\"a\", 1e400]]"), Mode.Decline },
        { "number-leading-zero", Doc("[[\"<unk>\", 0], [\"a\", 01]]"), Mode.Decline },
        { "number-trailing-dot", Doc("[[\"<unk>\", 0], [\"a\", 1.]]"), Mode.Decline },
        { "number-leading-dot", Doc("[[\"<unk>\", 0], [\"a\", .5]]"), Mode.Decline },
        { "number-plus", Doc("[[\"<unk>\", 0], [\"a\", +1]]"), Mode.Decline },
        { "number-nan", Doc("[[\"<unk>\", 0], [\"a\", NaN]]"), Mode.Decline },
        { "number-suffix", Doc("[[\"<unk>\", 0], [\"a\", 1.5x]]"), Mode.Decline },
        { "number-as-string", Doc("[[\"<unk>\", 0], [\"a\", \"1\"]]"), Mode.Decline },
        { "number-null", Doc("[[\"<unk>\", 0], [\"a\", null]]"), Mode.Decline },
        { "entry-numbers", Doc("[[\"<unk>\", 0], [1, 2]]"), Mode.Decline },
        { "entry-short", Doc("[[\"<unk>\", 0], [\"a\"]]"), Mode.Decline },
        { "entry-long", Doc("[[\"<unk>\", 0], [\"a\", 1, 2]]"), Mode.Decline },
        { "entry-empty", Doc("[[\"<unk>\", 0], []]"), Mode.Decline },
        { "entry-string", Doc("[[\"<unk>\", 0], \"a\"]"), Mode.Decline },
        { "entry-object", Doc("[[\"<unk>\", 0], {\"a\": 1}]"), Mode.Decline },
        { "trailing-comma", Doc("[[\"<unk>\", 0], [\"a\", 1],]"), Mode.Decline },
        { "entry-trailing-comma", Doc("[[\"<unk>\", 0], [\"a\", 1,]]"), Mode.Decline },
        { "missing-comma", Doc("[[\"<unk>\", 0] [\"a\", 1]]"), Mode.Decline },
        { "lone-surrogate", Doc("""[["<unk>", 0.0], ["\ud800", -1.0]]"""), Mode.Decline },
        { "bad-escape", Doc("""[["<unk>", 0.0], ["\x", -1.0]]"""), Mode.Decline },
        { "raw-control", Doc("[[\"<unk>\", 0.0], [\"a\tb\", -1.0]]"), Mode.Decline },
        { "truncated", Doc(_vocab)[..(_prefix.Length + 60)], Mode.Decline },
        { "unk-outside", Doc(_vocab, unk: "6"), Mode.Any },
        { "unknown-model-key", Doc(_vocab, extra: ", \"foo\": 1"), Mode.Any },
        { "byte-fallback-true", Doc(_vocab, extra: ", \"byte_fallback\": true"), Mode.Any },
        { "bpe-with-array-vocab", Doc(_vocab).Replace("\"Unigram\"", "\"BPE\"", StringComparison.Ordinal), Mode.Any },
        { "vocab-object", Doc("{\"a\": 0}"), Mode.Decline },
        { "vocab-null", Doc("null"), Mode.Decline },
        { "duplicate-vocab", Doc(_vocab, extra: ", \"vocab\": [[\"<unk>\", 0.0]]"), Mode.Decline },
        { "duplicate-model", Doc(_vocab)[..^1] + ", \"model\": {\"type\": \"Unigram\", \"unk_id\": 0, \"vocab\": [[\"<unk>\", 0.0]]}}", Mode.Decline },
        { "model-not-object", "{\"model\": 1, " + _prefix.TrimStart()[1..].TrimEnd().TrimEnd(',') + "}", Mode.Decline },
        { "no-vocab", Doc(_vocab).Replace("\"vocab\"", "\"vocabulary\"", StringComparison.Ordinal), Mode.Decline },
        { "trailing-garbage", Doc(_vocab) + " x", Mode.Any },
        { "trailing-document", Doc(_vocab) + " {}", Mode.Any },
        { "comment-after-vocab", Doc(_vocab).Replace("]]}", "]] /* c */ }", StringComparison.Ordinal), Mode.Any },
        { "invalid-json-after-vocab", Doc(_vocab).Replace("]]}", "]], }", StringComparison.Ordinal), Mode.Any },
        { "unknown-root-key", Doc(_vocab)[..^1] + ", \"extra\": true}", Mode.Any },
        { "root-array", "[" + Doc(_vocab) + "]", Mode.Decline },
    };

    [Theory]
    [MemberData(nameof(Variants))]
    public void FastPathGivesTheSameResultAsTheDocumentPath(string name, string json, Mode mode)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        (string fast, bool usedFastPath) = Outcome(bytes, useFastPath: true);
        (string document, _) = Outcome(bytes, useFastPath: false);
        Assert.Equal(document, fast);
        if (mode != Mode.Any)
        {
            Assert.True(usedFastPath == (mode == Mode.Fast), $"{name}: fast path used = {usedFastPath}; outcome: {fast}");
        }
    }

    /// <summary>What the fast path must do with a variant (an exception counts as not taking it).</summary>
    public enum Mode
    {
        /// <summary>Loads successfully through the fast path.</summary>
        Fast,

        /// <summary>The fast path declines; the document path decides.</summary>
        Decline,

        /// <summary>Either; only the outcome is compared.</summary>
        Any,
    }

    [Fact]
    public void FastPathTakesThePlainDocument()
    {
        Assert.True(TokenizerJsonLoader.Load(Encoding.UTF8.GetBytes(Doc(_vocab))).VocabularyFastPath);
    }

    [Fact]
    public void InvalidUtf8InPieceGivesTheSameResult()
    {
        byte[] bytes = Encoding.UTF8.GetBytes(Doc("[[\"<unk>\", 0.0], [\"a?b\", -1.0]]"));
        int at = Array.IndexOf(bytes, (byte)'?');
        foreach (byte bad in new byte[] { 0xFF, 0xC0, 0x80, 0xED })
        {
            bytes[at] = bad;
            Assert.Equal(Outcome(bytes, useFastPath: false).Text, Outcome(bytes, useFastPath: true).Text);
            Assert.False(Outcome(bytes, useFastPath: true).UsedFastPath);
        }
    }

    [Fact]
    public void ByteOrderMarkGivesTheSameResult()
    {
        byte[] bytes = [0xEF, 0xBB, 0xBF, .. Encoding.UTF8.GetBytes(Doc(_vocab))];
        Assert.Equal(Outcome(bytes, useFastPath: false).Text, Outcome(bytes, useFastPath: true).Text);
    }

    [Theory]
    [InlineData("e5")]
    [InlineData("gliner2")]
    public void PinnedAssetLoadsIdenticallyOnBothPaths(string assetName)
    {
        byte[] json = File.ReadAllBytes(TestAssets.RequireAsset(Asset(assetName)));
        TokenizerPipeline fast = TokenizerJsonLoader.Load(json, useVocabularyFastPath: true);
        TokenizerPipeline document = TokenizerJsonLoader.Load(json, useVocabularyFastPath: false);
        Assert.True(fast.VocabularyFastPath);
        Assert.False(document.VocabularyFastPath);
        Assert.Equal(document.Components, fast.Components);
        Assert.Equal(document.Model.VocabularySize, fast.Model.VocabularySize);
        for (int id = 0; id < fast.Model.VocabularySize; id++)
        {
            string piece = document.Model.IdToToken(id)!;
            Assert.Equal(piece, fast.Model.IdToToken(id));
            Assert.Equal(BitConverter.DoubleToInt64Bits(document.Unigram!.Score(id)), BitConverter.DoubleToInt64Bits(fast.Unigram!.Score(id)));
            Assert.Equal(document.Model.TokenToId(piece), fast.Model.TokenToId(piece));
        }
    }

    /// <summary>
    /// The vocabulary trie against an independent oracle built from the file with
    /// <see cref="JsonDocument"/>: walking every piece byte by byte as the encoder does, each step
    /// returns the id of the last occurrence of that prefix as a piece, -1 for a prefix of a piece
    /// that is not a piece, and -2 for a byte string that prefixes no piece.
    /// </summary>
    [Theory]
    [InlineData("e5")]
    [InlineData("gliner2")]
    public void VocabularyTrieMatchesTheOracle(string assetName) =>
        AssertTrieMatchesOracle(File.ReadAllBytes(TestAssets.RequireAsset(Asset(assetName))), probeEvery: 16);

    /// <summary>
    /// The same oracle check on a synthetic vocabulary built to stress the sort and the builder:
    /// a three-letter alphabet plus multi-byte characters gives long shared prefixes (beyond the
    /// eight radix bytes), prefixes that are pieces, and many duplicated pieces.
    /// </summary>
    [Fact]
    public void VocabularyTrieMatchesTheOracleOnSyntheticVocabulary()
    {
        var random = new Random(20261009);
        string[] alphabet = ["a", "b", "c", "é", "▁", "😀"];
        var entries = new List<string> { "[\"<unk>\", 0.0]" };
        for (int i = 0; i < 20000; i++)
        {
            var piece = new StringBuilder(i % 7 == 0 ? "aaaaaaaaab" : string.Empty);
            int length = 1 + random.Next(random.Next(2) == 0 ? 4 : 16);
            for (int k = 0; k < length; k++)
            {
                piece.Append(alphabet[random.Next(random.Next(4) == 0 ? alphabet.Length : 3)]);
            }

            entries.Add("[" + JsonSerializer.Serialize(piece.ToString()) + ", " + (-random.NextDouble() * 20).ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "]");
        }

        byte[] json = Encoding.UTF8.GetBytes(Doc("[" + string.Join(", ", entries) + "]"));
        Assert.True(TokenizerJsonLoader.Load(json).VocabularyFastPath);
        AssertTrieMatchesOracle(json, probeEvery: 1);
    }

    private static void AssertTrieMatchesOracle(byte[] json, int probeEvery)
    {
        var lastId = new Dictionary<string, int>(StringComparer.Ordinal);
        using (JsonDocument doc = JsonDocument.Parse(json))
        {
            int id = 0;
            foreach (JsonElement entry in doc.RootElement.GetProperty("model").GetProperty("vocab").EnumerateArray())
            {
                lastId[Encoding.Latin1.GetString(Encoding.UTF8.GetBytes(entry[0].GetString()!))] = id++;
            }
        }

        var prefixes = new HashSet<string>(StringComparer.Ordinal);
        foreach (string key in lastId.Keys)
        {
            for (int n = 1; n < key.Length; n++)
            {
                prefixes.Add(key[..n]);
            }
        }

        int Expected(string latin1) =>
            lastId.TryGetValue(latin1, out int found) ? found : prefixes.Contains(latin1) ? -1 : -2;

        TokenizerPipeline pipeline = TokenizerJsonLoader.Load(json);
        ThirdParty.MLNet.DoubleArrayTrie trie = pipeline.Unigram!.Trie;
        byte[] probes = [0x01, (byte)' ', (byte)'a', (byte)'z', 0x7F, 0x80, 0xBF, 0xC3, 0xE2, 0xF0, 0xFF];
        long checkedSteps = 0;
        int keyIndex = 0;
        foreach (string key in lastId.Keys)
        {
            byte[] bytes = Encoding.Latin1.GetBytes(key);
            bool probeThisKey = keyIndex++ % probeEvery == 0;
            int nodePos = 0;
            for (int keyPos = 0; keyPos < bytes.Length;)
            {
                // Probe one-byte extensions of the current prefix from the current node.
                foreach (byte probe in probeThisKey ? probes : [])
                {
                    byte[] extended = [.. bytes.AsSpan(0, keyPos), probe];
                    int probeNode = nodePos;
                    int probePos = keyPos;
                    int probeResult = trie.Traverse(extended, ref probeNode, ref probePos, keyPos + 1);
                    Assert.Equal(Expected(Encoding.Latin1.GetString(extended)), probeResult);
                }

                int result = trie.Traverse(bytes, ref nodePos, ref keyPos, keyPos + 1);
                Assert.Equal(Expected(key[..keyPos]), result);
                checkedSteps++;
            }

            Assert.Equal(lastId[key], pipeline.Model.TokenToId(Encoding.UTF8.GetString(bytes)));
        }

        Assert.True(checkedSteps > lastId.Count);
        Assert.Null(pipeline.Model.TokenToId(string.Empty));
        Assert.Null(pipeline.Model.TokenToId("\ud800"));
    }

    private static PinnedAsset Asset(string name) => name switch
    {
        "e5" => TestAssets.E5TokenizerJson,
        "gliner2" => TestAssets.Gliner2TokenizerJson,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    private static string Doc(string vocab, string unk = "0", string extra = "") =>
        _prefix + "\"model\": {\"type\": \"Unigram\", \"unk_id\": " + unk + extra + ", \"vocab\": " + vocab + "}}";

    /// <summary>Everything observable about a load: the model's contents, or the exception chain.</summary>
    private static (string Text, bool UsedFastPath) Outcome(byte[] json, bool useFastPath)
    {
        try
        {
            TokenizerPipeline p = TokenizerJsonLoader.Load(json, useFastPath);
            var text = new StringBuilder();
            text.AppendLine(string.Join(",", p.Components));
            for (int id = 0; id < p.Model.VocabularySize; id++)
            {
                string piece = p.Model.IdToToken(id)!;
                text.Append(id).Append('\t').Append(Convert.ToHexString(Encoding.UTF8.GetBytes(piece)))
                    .Append('\t').Append(BitConverter.DoubleToInt64Bits(p.Unigram!.Score(id)))
                    .Append('\t').Append(p.Model.TokenToId(piece)).AppendLine();
            }

            text.Append("added=").Append(p.AddedVocabulary.Count);
            return (text.ToString(), p.VocabularyFastPath);
        }
        catch (Exception e)
        {
            var text = new StringBuilder();
            for (Exception? x = e; x is not null; x = x.InnerException)
            {
                text.Append(x.GetType().FullName).Append(": ").Append(x.Message).Append(" | ");
            }

            return (text.ToString(), false);
        }
    }
}
