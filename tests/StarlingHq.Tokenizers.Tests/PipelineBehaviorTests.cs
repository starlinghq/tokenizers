using System.Text;
using System.Text.Json.Nodes;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>
/// Behavioral unit tests on a tiny synthetic <c>tokenizer.json</c>. They run without model
/// assets. Expected values are derived by hand from the algorithm, so they are regression tests,
/// not conformance evidence; conformance comes from the protected fixtures.
/// </summary>
public class PipelineBehaviorTests
{
    private const string _minimal = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 5, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Strip", "strip_left": true, "strip_right": true},
            {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}
          ]},
          "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
          "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [5], "tokens": ["<s>"]}}
          },
          "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
          "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["▁a", -1.0], ["b", -2.0], ["▁", -3.0], ["▁b", -9.0], ["<s>", 0.0]]}
        }
        """;

    private static Tokenizer Load(string json) => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json));

    private static string Mutate(Action<JsonObject> change)
    {
        JsonObject root = JsonNode.Parse(_minimal)!.AsObject();
        change(root);
        return root.ToJsonString();
    }

    [Fact]
    public void EncodesThroughEveryStage()
    {
        Tokenizer tokenizer = Load(_minimal);
        TokenEncoding e = tokenizer.Encode("  a   b ");

        // Strip, collapse spaces, Metaspace: "▁a▁b" -> "▁a" + "▁b"; "▁"+"b" (-5) beats "▁b" (-9).
        Assert.Equal([5, 1, 3, 2], e.Ids);
        Assert.Equal(["<s>", "▁a", "▁", "b"], e.Tokens);
        Assert.Equal([1, 0, 0, 0], e.SpecialTokensMask);

        // A Replace inserts its content with the alignment of the last matched byte, so the
        // collapsed space maps to the last of the three original spaces (upstream behavior).
        Assert.Equal([new TokenOffset(0, 0), new TokenOffset(2, 3), new TokenOffset(5, 6), new TokenOffset(6, 7)], e.Offsets);
        Assert.Equal("a b", tokenizer.Decode(e.Ids.AsSpan(), skipSpecialTokens: true));
        Assert.Equal("<s> a b", tokenizer.Decode(e.Ids.AsSpan(), skipSpecialTokens: false));
    }

    [Fact]
    public void FusesConsecutiveUnknownCharacters()
    {
        TokenEncoding e = Load(_minimal).Encode("xx\U0001F600", addSpecialTokens: false);
        Assert.Equal([3, 0], e.Ids);
        Assert.Equal(["▁", "xx\U0001F600"], e.Tokens);
        Assert.Equal([new TokenOffset(0, 1), new TokenOffset(0, 4)], e.Offsets);
        Assert.Equal([new TokenOffset(0, 1), new TokenOffset(0, 3)], e.ScalarOffsets);
    }

    [Fact]
    public void MatchesAddedTokensInText()
    {
        TokenEncoding e = Load(_minimal).Encode("a<s>b", addSpecialTokens: false);
        Assert.Equal(["▁a", "<s>", "▁", "b"], e.Tokens);
        Assert.Equal([0, 0, 0, 0], e.SpecialTokensMask);
    }

    [Fact]
    public void PairUsesTemplateTypeIdsAndSequenceIds()
    {
        TokenEncoding e = Load(_minimal).Encode("a", "b");
        Assert.Equal([5, 1, 3, 2], e.Ids);
        Assert.Equal([0, 0, 1, 1], e.TypeIds);
        Assert.Equal([null, 0, 1, 1], e.SequenceIds);
        Assert.Equal([new TokenOffset(0, 0), new TokenOffset(0, 1), new TokenOffset(0, 1), new TokenOffset(0, 1)], e.Offsets);
    }

    [Fact]
    public void EmptyInputKeepsTemplateTokens()
    {
        TokenEncoding e = Load(_minimal).Encode(string.Empty);
        Assert.Equal([5], e.Ids);
        Assert.Empty(Load(_minimal).Encode(string.Empty, addSpecialTokens: false).Ids);
    }

    [Fact]
    public void TruncatesAndPadsBothDirections()
    {
        Tokenizer tokenizer = Load(_minimal);
        TokenEncoding right = tokenizer.WithTruncation(new TruncationOptions(3)).Encode("a b");
        Assert.Equal([5, 1, 3], right.Ids);

        TokenEncoding left = tokenizer.WithTruncation(new TruncationOptions(3, Direction: TruncationDirection.Left)).Encode("a b");
        Assert.Equal([5, 3, 2], left.Ids);

        Tokenizer padded = tokenizer.WithPadding(new PaddingOptions(0, "<unk>", Direction: PaddingDirection.Left, PadToMultipleOf: 8));
        IReadOnlyList<TokenEncoding> batch = padded.EncodeBatch(["a", "a b"]);
        Assert.All(batch, b => Assert.Equal(8, b.Count));
        Assert.Equal([0, 0, 0, 0, 0, 0, 5, 1], batch[0].Ids);
        Assert.Equal([0, 0, 0, 0, 0, 0, 1, 1], batch[0].AttentionMask);
        Assert.Equal([null, null, null, null, null, null, null, 0], batch[0].SequenceIds);
    }

    [Fact]
    public void RejectsUnpairedSurrogates()
    {
        Assert.Throws<ArgumentException>(() => Load(_minimal).Encode("a\uD800b"));
    }

    [Fact]
    public void ReportsLoadedComponents()
    {
        Assert.Equal(
            ["normalizer:Strip", "normalizer:Replace", "normalizer:Sequence", "pre_tokenizer:Metaspace", "model:Unigram", "added_tokens", "post_processor:TemplateProcessing", "decoder:Metaspace"],
            Load(_minimal).Components);
    }

    public static TheoryData<string, string> UnsupportedCases() => new()
    {
        { Mutate(r => r["version"] = "2.0"), "version=2.0" },
        { Mutate(r => r["normalizer"] = new JsonObject { ["type"] = "NFKC" }), "normalizer.type=NFKC" },
        { Mutate(r => r["normalizer"] = new JsonObject { ["type"] = "Replace", ["pattern"] = new JsonObject { ["Regex"] = "\\s+" }, ["content"] = " " }), "normalizer.type=Replace" },
        { Mutate(r => r["pre_tokenizer"] = new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = true, ["use_regex"] = true }), "pre_tokenizer.type=ByteLevel.use_regex" },
        { Mutate(r => r["pre_tokenizer"]!["unexpected"] = 1), "pre_tokenizer.type=Metaspace.unexpected" },
        { Mutate(r => r["model"]!["type"] = "UnknownModel"), "model.type=UnknownModel" },
        { Mutate(r => r["model"]!["byte_fallback"] = true), "model.type=Unigram.byte_fallback" },
        { Mutate(r => r["added_tokens"]![0]!["lstrip"] = true), "added_tokens[0].lstrip" },
        { Mutate(r => r["post_processor"] = new JsonObject { ["type"] = "BertProcessing" }), "post_processor.type=BertProcessing" },
        { Mutate(r => r["decoder"] = new JsonObject { ["type"] = "BPEDecoder" }), "decoder.type=BPEDecoder" },
        { Mutate(r => r["truncation"] = new JsonObject { ["direction"] = "Right", ["max_length"] = 8, ["strategy"] = "LongestFirst", ["stride"] = 2 }), "truncation.stride" },
        { Mutate(r => r["added_tokens"]![1]!["id"] = 6), "added_tokens" },
    };

    [Theory]
    [MemberData(nameof(UnsupportedCases))]
    public void RejectsUnsupportedComponentsByName(string json, string component)
    {
        UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(() => Load(json));
        Assert.Equal(component, e.Component);
        Assert.Contains(component, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectsMalformedJson()
    {
        TokenizerFormatException e = Assert.Throws<TokenizerFormatException>(() => Load("{\"version\": \"1.0\""));
        Assert.IsNotType<UnsupportedTokenizerComponentException>(e);
    }

    [Fact]
    public void ReadsTruncationAndPaddingFromFile()
    {
        string json = Mutate(r =>
        {
            r["truncation"] = new JsonObject { ["direction"] = "Left", ["max_length"] = 3, ["strategy"] = "OnlyFirst", ["stride"] = 0 };
            r["padding"] = new JsonObject
            {
                ["strategy"] = new JsonObject { ["Fixed"] = 6 },
                ["direction"] = "Right",
                ["pad_to_multiple_of"] = null,
                ["pad_id"] = 0,
                ["pad_type_id"] = 0,
                ["pad_token"] = "<unk>",
            };
        });
        Tokenizer tokenizer = Load(json);
        Assert.Equal(new TruncationOptions(3, TruncationStrategy.OnlyFirst, TruncationDirection.Left), tokenizer.Truncation);
        Assert.Equal(new PaddingOptions(0, "<unk>", 6), tokenizer.Padding);
        TokenEncoding e = tokenizer.Encode("a b");
        Assert.Equal([5, 3, 2, 0, 0, 0], e.Ids);
    }
}
