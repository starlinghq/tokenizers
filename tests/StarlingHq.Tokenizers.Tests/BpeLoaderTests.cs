using System.Text;
using System.Text.Json.Nodes;
using StarlingHq.Tokenizers.Pipeline;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>
/// The BPE loader: the byte-scanning fast path against the document path on the pinned Qwen assets
/// and on synthetic documents, the supported options, and explicit rejection of the rest.
/// Expected encodings of the synthetic tokenizer are derived by hand (regression tests); conformance
/// evidence is in <see cref="QwenBpeConformanceTests"/> and <see cref="QwenBpeDifferentialTests"/>.
/// </summary>
public class BpeLoaderTests
{
    private const string _minimal = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {"id": 7, "content": "<|x|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": {"type": "NFC"},
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
          ]},
          "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
          "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": {"a": 0, "b": 1, "c": 2, "\u0120": 3, "ab": 4, "abc": 5, "\u0120a": 6},
            "merges": [["a", "b"], ["ab", "c"], ["\u0120", "a"]]
          }
        }
        """;

    private static Tokenizer Load(string json) => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json));

    private static string Mutate(Action<JsonObject> change)
    {
        JsonObject root = JsonNode.Parse(_minimal)!.AsObject();
        change(root);
        return root.ToJsonString();
    }

    [Theory]
    [InlineData("kev")]
    [InlineData("chat")]
    public void PinnedAssetLoadsIdenticallyOnBothPaths(string profile)
    {
        byte[] json = File.ReadAllBytes(TestAssets.RequireAsset(QwenProfile.Get(profile).Asset));
        TokenizerPipeline fast = TokenizerJsonLoader.Load(json, useVocabularyFastPath: true);
        TokenizerPipeline document = TokenizerJsonLoader.Load(json, useVocabularyFastPath: false);
        Assert.True(fast.VocabularyFastPath);
        Assert.False(document.VocabularyFastPath);
        AssertSameModel(document, fast);
        Assert.Equal(248044, fast.Model.VocabularySize);
        Assert.Equal(247587, fast.Bpe!.Merges.Count);
    }

    [Fact]
    public void GemmaAssetLoadsIdenticallyOnBothPaths()
    {
        byte[] json = File.ReadAllBytes(TestAssets.RequireAsset(TestAssets.EmbeddingGemma2TokenizerJson));
        TokenizerPipeline fast = TokenizerJsonLoader.Load(json, useVocabularyFastPath: true);
        TokenizerPipeline document = TokenizerJsonLoader.Load(json, useVocabularyFastPath: false);
        Assert.True(fast.VocabularyFastPath);
        Assert.False(document.VocabularyFastPath);
        AssertSameModel(document, fast);
        Assert.Equal(262144, fast.Model.VocabularySize);
        Assert.Equal(514906, fast.Bpe!.Merges.Count);
        Assert.True(fast.Bpe.ByteFallback);
        Assert.True(fast.Bpe.FuseUnk);
        Assert.Equal(3, fast.Bpe.UnkId);
    }

    [Fact]
    public void EncodesByLowestRankFirst()
    {
        Tokenizer tokenizer = Load(_minimal);
        TokenEncoding e = tokenizer.Encode("abc ab");

        // "abc" -> ab+c -> abc; "Ġab": (a,b) rank 0 before (Ġ,a) rank 2, then (Ġ,ab) is no merge.
        Assert.Equal([5, 3, 4], e.Ids);
        Assert.Equal(["abc", "\u0120", "ab"], e.Tokens);
        Assert.Equal([new TokenOffset(0, 3), new TokenOffset(3, 4), new TokenOffset(4, 6)], e.Offsets);
        Assert.Equal("abc ab", tokenizer.Decode(e.Ids.AsSpan(), skipSpecialTokens: false));
    }

    [Fact]
    public void DropsCharactersOutsideTheVocabularyWithoutUnknownToken()
    {
        // Upstream merge_word adds no symbol for such a character; later offsets do not count it.
        Tokenizer tokenizer = Load(_minimal);
        TokenEncoding e = tokenizer.Encode("dab");
        Assert.Equal([4], e.Ids);
        Assert.Equal([new TokenOffset(0, 2)], e.Offsets);
    }

    [Fact]
    public void IgnoreMergesTakesWholeVocabularyWords()
    {
        // "Ġa" is in the vocabulary: with ignore_merges it is one token even without merging.
        Tokenizer plain = Load(_minimal);
        Tokenizer ignoring = Load(Mutate(r => r["model"]!["ignore_merges"] = true));
        Assert.Equal([6], plain.Encode(" a").Ids);
        Assert.Equal([6], ignoring.Encode(" a").Ids);
        Assert.Equal([5], ignoring.Encode("abc").Ids);
        Assert.Equal([4, 2], Load(Mutate(r => r["model"]!["merges"] = new JsonArray(new JsonArray("a", "b")))).Encode("abc").Ids);
    }

    [Fact]
    public void LegacyMergeStringsMatchPairs()
    {
        Tokenizer legacy = Load(Mutate(r => r["model"]!["merges"] = new JsonArray("#version: 0.2", "a b", "ab c", "\u0120 a")));
        Tokenizer pairs = Load(_minimal);
        foreach (string text in new[] { "abc ab", " a", "cab" })
        {
            Assert.Equal(pairs.Encode(text).Ids, legacy.Encode(text).Ids);
        }
    }

    public static TheoryData<string> FastPathDocuments() =>
    [
        _minimal,
        Mutate(r => r["model"]!["merges"] = new JsonArray("#version: 0.2", "a b", "ab c", "\u0120 a")),
        Mutate(_ => { }),                                                   // compact JSON
        _minimal.Replace("\"ab\": 4", "\"a\\u0062\": 4", StringComparison.Ordinal),       // escaped key
        _minimal.Replace("[\"a\", \"b\"]", "[ \"a\" ,\n \"\\u0062\" ]", StringComparison.Ordinal), // escaped merge, whitespace
        Mutate(r => { r["model"]!["vocab"] = new JsonObject(); r["model"]!["merges"] = new JsonArray(); r["added_tokens"] = new JsonArray(); }),
    ];

    [Theory]
    [MemberData(nameof(FastPathDocuments))]
    public void FastPathReadsWhatTheDocumentPathReads(string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        TokenizerPipeline fast = TokenizerJsonLoader.Load(bytes, useVocabularyFastPath: true);
        TokenizerPipeline document = TokenizerJsonLoader.Load(bytes, useVocabularyFastPath: false);
        Assert.True(fast.VocabularyFastPath);
        AssertSameModel(document, fast);
    }

    // merges before vocab: not the order tokenizers writes; the document path reads it.
    private static readonly string _reordered =
        _minimal.Replace(
            "\"vocab\": {\"a\": 0, \"b\": 1, \"c\": 2, \"\\u0120\": 3, \"ab\": 4, \"abc\": 5, \"\\u0120a\": 6},\n    \"merges\": [[\"a\", \"b\"], [\"ab\", \"c\"], [\"\\u0120\", \"a\"]]",
            "\"merges\": [[\"a\", \"b\"], [\"ab\", \"c\"], [\"\\u0120\", \"a\"]],\n    \"vocab\": {\"a\": 0, \"b\": 1, \"c\": 2, \"\\u0120\": 3, \"ab\": 4, \"abc\": 5, \"\\u0120a\": 6}",
            StringComparison.Ordinal);

    private static readonly string _floatId = _minimal.Replace("\"ab\": 4", "\"ab\": 4.0e0", StringComparison.Ordinal);

    [Fact]
    public void FastPathDeclinesOtherLayoutsAndTheDocumentPathDecides()
    {
        string reordered = _reordered;
        Assert.NotEqual(_minimal, reordered);
        TokenizerPipeline p = TokenizerJsonLoader.Load(Encoding.UTF8.GetBytes(reordered), useVocabularyFastPath: true);
        Assert.False(p.VocabularyFastPath);
        Assert.Equal([5, 3, 4], Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(reordered)).Encode("abc ab").Ids);

        // A float id: the reference rejects it, so the document path must fail, not the fast path guess.
        Assert.Throws<TokenizerFormatException>(() => Load(_floatId));
    }

    public static TheoryData<string, string> UnsupportedCases() => new()
    {
        { Mutate(r => r["model"]!["dropout"] = 0.1), "model.type=BPE.dropout" },
        { Mutate(r => r["model"]!["unk_token"] = "zz"), "model.type=BPE.unk_token" },
        { Mutate(r => r["model"]!["continuing_subword_prefix"] = "##"), "model.type=BPE.continuing_subword_prefix" },
        { Mutate(r => r["model"]!["end_of_word_suffix"] = "</w>"), "model.type=BPE.end_of_word_suffix" },
        { Mutate(r => r["model"]!["unexpected"] = 1), "model.type=BPE.unexpected" },
        { Mutate(r => r["pre_tokenizer"]!["pretokenizers"]![0]!["pattern"] = new JsonObject { ["Regex"] = "\\s+" }), "pre_tokenizer.pretokenizers[0].type=Split.pattern" },
        { Mutate(r => r["pre_tokenizer"]!["pretokenizers"]![0]!["pattern"] = new JsonObject { ["String"] = "" }), "pre_tokenizer.pretokenizers[0].type=Split.pattern" },
        { Mutate(r => r["pre_tokenizer"]!["pretokenizers"]![0]!["pattern"] = new JsonObject { ["String"] = " ", ["Regex"] = " " }), "pre_tokenizer.pretokenizers[0].type=Split.pattern" },
        { Mutate(r => r["pre_tokenizer"]!["pretokenizers"]![0]!["behavior"] = "Removed"), "pre_tokenizer.pretokenizers[0].type=Split.behavior" },
        { Mutate(r => r["pre_tokenizer"]!["pretokenizers"]![0]!["invert"] = true), "pre_tokenizer.pretokenizers[0].type=Split.invert" },
        { Mutate(r => r["pre_tokenizer"]!["pretokenizers"]![1]!.AsObject().Remove("use_regex")), "pre_tokenizer.pretokenizers[1].type=ByteLevel.use_regex" },
        { Mutate(r => r["normalizer"] = new JsonObject { ["type"] = "NFKC" }), "normalizer.type=NFKC" },
        { Mutate(r => r["post_processor"] = new JsonObject { ["type"] = "RobertaProcessing" }), "post_processor.type=RobertaProcessing" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "Strip", ["content"] = " ", ["start"] = 1, ["stop"] = 0 }), "decoder.type=Strip" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "CTC" }), "decoder.type=CTC" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "Sequence", ["decoders"] = new JsonArray(new JsonObject { ["type"] = "Fuse" }, new JsonObject { ["type"] = "BPEDecoder" }) }), "decoder.decoders[1].type=BPEDecoder" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "Replace", ["pattern"] = new JsonObject { ["Regex"] = "\\s+" }, ["content"] = " " }), "decoder.type=Replace" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "Replace", ["pattern"] = new JsonObject { ["String"] = "" }, ["content"] = " " }), "decoder.type=Replace" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "ByteFallback", ["extra"] = 1 }), "decoder.type=ByteFallback.extra" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "Fuse", ["extra"] = 1 }), "decoder.type=Fuse.extra" },
        { _minimal.Replace("\"abc\": 5", "\"abc\": 5, \"ab\": 8", StringComparison.Ordinal), "model.vocab" },
        { _minimal.Replace("\"abc\": 5", "\"abc\": 4", StringComparison.Ordinal), "model.vocab" },
    };

    [Theory]
    [MemberData(nameof(UnsupportedCases))]
    public void RejectsUnsupportedComponentsByName(string json, string component)
    {
        UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(() => Load(json));
        Assert.Equal(component, e.Component);
    }

    public static TheoryData<string> MalformedCases() =>
    [
        Mutate(r => r["model"]!["merges"]!.AsArray().Add(new JsonArray("c", "a"))),           // "ca" not in vocab
        Mutate(r => r["model"]!["merges"]!.AsArray().Add(new JsonArray("a", "zz"))),          // part not in vocab
        Mutate(r => r["model"]!["merges"] = new JsonArray("a b c")),                          // legacy line with three parts
        Mutate(r => r["model"]!["merges"] = new JsonArray(new JsonArray("a", "b"), "ab c")),  // mixed forms
        Mutate(r => r["model"]!["vocab"]!["a"] = -1),
        Mutate(r => r["model"]!.AsObject().Remove("merges")),
    ];

    [Theory]
    [MemberData(nameof(MalformedCases))]
    public void RejectsMalformedModels(string json)
    {
        TokenizerFormatException e = Assert.Throws<TokenizerFormatException>(() => Load(json));
        Assert.IsNotType<UnsupportedTokenizerComponentException>(e);
    }

    [Fact]
    public void NfcComposesAndKeepsOffsets()
    {
        // "e" + U+0301 composes to U+00E9 (two byte-level characters). The composite replaces "e"
        // and removes the mark, so it keeps the alignment of "e" only (tokenizers 0.23.2 gives
        // offsets (0, 1) for the Qwen asset too).
        JsonObject Vocab(JsonObject v)
        {
            v["\u00c3"] = 7;
            v["\u00a9"] = 8;
            return v;
        }

        Tokenizer tokenizer = Load(Mutate(r =>
        {
            Vocab(r["model"]!["vocab"]!.AsObject());
            r["added_tokens"]![0]!["id"] = 9;
        }));
        TokenEncoding e = tokenizer.Encode("e\u0301");
        Assert.Equal([7, 8], e.Ids);
        Assert.Equal([new TokenOffset(0, 1), new TokenOffset(0, 1)], e.Offsets);
        Assert.Equal("\u00e9", tokenizer.Decode(e.Ids.AsSpan(), skipSpecialTokens: false));
    }

    private static void AssertSameModel(TokenizerPipeline expected, TokenizerPipeline actual)
    {
        Assert.Equal(expected.Components, actual.Components);
        BpeModel a = expected.Bpe!;
        BpeModel b = actual.Bpe!;
        Assert.Equal(a.VocabularySize, b.VocabularySize);
        Assert.Equal(a.Vocabulary.IdLimit, b.Vocabulary.IdLimit);
        for (int id = 0; id < a.Vocabulary.IdLimit; id++)
        {
            Assert.True(a.Vocabulary.TokenBytes(id).SequenceEqual(b.Vocabulary.TokenBytes(id)), $"id {id}");
            Assert.Equal(a.Vocabulary.Contains(id), b.Vocabulary.Contains(id));
        }

        Assert.Equal(a.Merges.Count, b.Merges.Count);
        Assert.Equal(a.Merges.Entries().Order(), b.Merges.Entries().Order());
        Assert.Equal(a.IgnoreMerges, b.IgnoreMerges);
        Assert.Equal(a.ByteFallback, b.ByteFallback);
        Assert.Equal(a.FuseUnk, b.FuseUnk);
        Assert.Equal(a.UnkId, b.UnkId);
    }
}
