using System.Text;
using System.Text.Json.Nodes;
using Starlings.Tokenizers.Pipeline;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// The <c>Sequence</c> post-processor on a tiny synthetic byte-level BPE tokenizer: the loader gate
/// (accepted keys, recursion, rejected children), and behavior against reference values. The expected
/// encodings were produced by huggingface/tokenizers 0.23.3 (<c>Tokenizer.from_str</c> on the same JSON,
/// with <c>enable_truncation(3)</c> and <c>enable_padding(pad_id=7, pad_token="&lt;|x|&gt;")</c>), so they
/// are reference evidence, including the sequence ids the reference gives when a ByteLevel
/// post-processor follows a TemplateProcessing (one sequence id per template piece).
/// </summary>
public sealed class SequencePostProcessorTests
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
          "post_processor": null,
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
            "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "ab": 4, "abc": 5, "Ġa": 6},
            "merges": [["a", "b"], ["ab", "c"], ["Ġ", "a"]]
          }
        }
        """;

    private static readonly Dictionary<string, Variant> _expected = new()
    {
        ["bytelevel_then_template"] = new Variant(
            Added: [1, 1],
            Single: new Expected([3, 4, 3, 5, 7], ["Ġ", "ab", "Ġ", "abc", "<|x|>"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7), (0, 0)], [0, 0, 0, 0, null]),
            SinglePlain: new Expected([3, 4, 3, 5], ["Ġ", "ab", "Ġ", "abc"], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7)], [0, 0, 0, 0]),
            Pair: new Expected([3, 4, 3, 5, 6, 7], ["Ġ", "ab", "Ġ", "abc", "Ġa", "<|x|>"], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], [(1, 1), (1, 3), (1, 1), (1, 4), (5, 6), (0, 0)], [0, 0, 1, 1, 1, null]),
            PairPlain: new Expected([3, 4, 3, 5, 6], ["Ġ", "ab", "Ġ", "abc", "Ġa"], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (1, 1), (1, 4), (5, 6)], [0, 0, 1, 1, 1]),
            SingleTruncated: new Expected([3, 4, 7], ["Ġ", "ab", "<|x|>"], [0, 0, 0], [0, 0, 1], [1, 1, 1], [(1, 1), (1, 3), (0, 0)], [0, 0, null]),
            SingleTruncatedPlain: new Expected([3, 4, 3], ["Ġ", "ab", "Ġ"], [0, 0, 0], [0, 0, 0], [1, 1, 1], [(1, 1), (1, 3), (4, 4)], [0, 0, 0]),
            PairTruncated: new Expected([3, 3, 7], ["Ġ", "Ġ", "<|x|>"], [0, 1, 1], [0, 0, 1], [1, 1, 1], [(1, 1), (1, 1), (0, 0)], [0, 1, null]),
            Batch: [new Expected([4, 7, 7, 7, 7], ["ab", "<|x|>", "<|x|>", "<|x|>", "<|x|>"], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [(0, 2), (0, 0), (0, 0), (0, 0), (0, 0)], [0, null, null, null, null]), new Expected([3, 4, 3, 5, 7], ["Ġ", "ab", "Ġ", "abc", "<|x|>"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7), (0, 0)], [0, 0, 0, 0, null])]),
        ["template_then_bytelevel"] = new Variant(
            Added: [1, 1],
            Single: new Expected([3, 4, 3, 5, 7], ["Ġ", "ab", "Ġ", "abc", "<|x|>"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7), (0, 0)], [0, 0, 0, 0, 1]),
            SinglePlain: new Expected([3, 4, 3, 5], ["Ġ", "ab", "Ġ", "abc"], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7)], [0, 0, 0, 0]),
            Pair: new Expected([3, 4, 3, 5, 6, 7], ["Ġ", "ab", "Ġ", "abc", "Ġa", "<|x|>"], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], [(1, 1), (1, 3), (1, 1), (1, 4), (5, 6), (0, 0)], [0, 0, 1, 1, 1, 2]),
            PairPlain: new Expected([3, 4, 3, 5, 6], ["Ġ", "ab", "Ġ", "abc", "Ġa"], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (1, 1), (1, 4), (5, 6)], [0, 0, 1, 1, 1]),
            SingleTruncated: new Expected([3, 4, 7], ["Ġ", "ab", "<|x|>"], [0, 0, 0], [0, 0, 1], [1, 1, 1], [(1, 1), (1, 3), (0, 0)], [0, 0, 1]),
            SingleTruncatedPlain: new Expected([3, 4, 3], ["Ġ", "ab", "Ġ"], [0, 0, 0], [0, 0, 0], [1, 1, 1], [(1, 1), (1, 3), (4, 4)], [0, 0, 0]),
            PairTruncated: new Expected([3, 3, 7], ["Ġ", "Ġ", "<|x|>"], [0, 1, 1], [0, 0, 1], [1, 1, 1], [(1, 1), (1, 1), (0, 0)], [0, 1, 2]),
            Batch: [new Expected([4, 7, 7, 7, 7], ["ab", "<|x|>", "<|x|>", "<|x|>", "<|x|>"], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [(0, 2), (0, 0), (0, 0), (0, 0), (0, 0)], [0, 1, null, null, null]), new Expected([3, 4, 3, 5, 7], ["Ġ", "ab", "Ġ", "abc", "<|x|>"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7), (0, 0)], [0, 0, 0, 0, 1])]),
        ["nested"] = new Variant(
            Added: [1, 1],
            Single: new Expected([3, 4, 3, 5, 7], ["Ġ", "ab", "Ġ", "abc", "<|x|>"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7), (0, 0)], [0, 0, 0, 0, null]),
            SinglePlain: new Expected([3, 4, 3, 5], ["Ġ", "ab", "Ġ", "abc"], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7)], [0, 0, 0, 0]),
            Pair: new Expected([3, 4, 3, 5, 6, 7], ["Ġ", "ab", "Ġ", "abc", "Ġa", "<|x|>"], [0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], [(1, 1), (1, 3), (1, 1), (1, 4), (5, 6), (0, 0)], [0, 0, 1, 1, 1, null]),
            PairPlain: new Expected([3, 4, 3, 5, 6], ["Ġ", "ab", "Ġ", "abc", "Ġa"], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (1, 1), (1, 4), (5, 6)], [0, 0, 1, 1, 1]),
            SingleTruncated: new Expected([3, 4, 7], ["Ġ", "ab", "<|x|>"], [0, 0, 0], [0, 0, 1], [1, 1, 1], [(1, 1), (1, 3), (0, 0)], [0, 0, null]),
            SingleTruncatedPlain: new Expected([3, 4, 3], ["Ġ", "ab", "Ġ"], [0, 0, 0], [0, 0, 0], [1, 1, 1], [(1, 1), (1, 3), (4, 4)], [0, 0, 0]),
            PairTruncated: new Expected([3, 3, 7], ["Ġ", "Ġ", "<|x|>"], [0, 1, 1], [0, 0, 1], [1, 1, 1], [(1, 1), (1, 1), (0, 0)], [0, 1, null]),
            Batch: [new Expected([4, 7, 7, 7, 7], ["ab", "<|x|>", "<|x|>", "<|x|>", "<|x|>"], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [(0, 2), (0, 0), (0, 0), (0, 0), (0, 0)], [0, null, null, null, null]), new Expected([3, 4, 3, 5, 7], ["Ġ", "ab", "Ġ", "abc", "<|x|>"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 1, 1, 1], [(1, 1), (1, 3), (4, 4), (4, 7), (0, 0)], [0, 0, 0, 0, null])]),
        ["empty"] = new Variant(
            Added: [0, 0],
            Single: new Expected([3, 4, 3, 5], ["Ġ", "ab", "Ġ", "abc"], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [(0, 1), (1, 3), (3, 4), (4, 7)], [0, 0, 0, 0]),
            SinglePlain: new Expected([3, 4, 3, 5], ["Ġ", "ab", "Ġ", "abc"], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [(0, 1), (1, 3), (3, 4), (4, 7)], [0, 0, 0, 0]),
            Pair: new Expected([3, 4, 3, 5, 6], ["Ġ", "ab", "Ġ", "abc", "Ġa"], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [(0, 1), (1, 3), (0, 1), (1, 4), (4, 6)], [0, 0, 1, 1, 1]),
            PairPlain: new Expected([3, 4, 3, 5, 6], ["Ġ", "ab", "Ġ", "abc", "Ġa"], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [(0, 1), (1, 3), (0, 1), (1, 4), (4, 6)], [0, 0, 1, 1, 1]),
            SingleTruncated: new Expected([3, 4, 3], ["Ġ", "ab", "Ġ"], [0, 0, 0], [0, 0, 0], [1, 1, 1], [(0, 1), (1, 3), (3, 4)], [0, 0, 0]),
            SingleTruncatedPlain: new Expected([3, 4, 3], ["Ġ", "ab", "Ġ"], [0, 0, 0], [0, 0, 0], [1, 1, 1], [(0, 1), (1, 3), (3, 4)], [0, 0, 0]),
            PairTruncated: new Expected([3, 3, 5], ["Ġ", "Ġ", "abc"], [0, 1, 1], [0, 0, 0], [1, 1, 1], [(0, 1), (0, 1), (1, 4)], [0, 1, 1]),
            Batch: [new Expected([4, 7, 7, 7], ["ab", "<|x|>", "<|x|>", "<|x|>"], [0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 0, 0], [(0, 2), (0, 0), (0, 0), (0, 0)], [0, null, null, null]), new Expected([3, 4, 3, 5], ["Ġ", "ab", "Ġ", "abc"], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [(0, 1), (1, 3), (3, 4), (4, 7)], [0, 0, 0, 0])]),

    };

    public static TheoryData<string> Variants() => [.. _expected.Keys];

    [Theory]
    [MemberData(nameof(Variants))]
    public void EncodesLikeTheReference(string variant)
    {
        Variant expected = _expected[variant];
        Tokenizer tokenizer = Load(Json(variant));

        AssertEncoding(expected.Single, tokenizer.Encode(" ab abc"));
        AssertEncoding(expected.SinglePlain, tokenizer.Encode(" ab abc", addSpecialTokens: false));
        AssertEncoding(expected.Pair, tokenizer.Encode(" ab", " abc a"));
        AssertEncoding(expected.PairPlain, tokenizer.Encode(" ab", " abc a", addSpecialTokens: false));
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void TruncatesBeforePostProcessingAndReservesTheAddedTokens(string variant)
    {
        Variant expected = _expected[variant];
        Tokenizer truncating = Load(Json(variant)).WithTruncation(new TruncationOptions(3));

        AssertEncoding(expected.SingleTruncated, truncating.Encode(" ab abc a"));
        AssertEncoding(expected.SingleTruncatedPlain, truncating.Encode(" ab abc a", addSpecialTokens: false));
        AssertEncoding(expected.PairTruncated, truncating.Encode(" ab", " abc a"));
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void PadsABatchLikeTheReference(string variant)
    {
        Variant expected = _expected[variant];
        IReadOnlyList<TokenEncoding> batch = Load(Json(variant)).WithPadding(new PaddingOptions(7, "<|x|>")).EncodeBatch(["ab", " ab abc"]);

        Assert.Equal(expected.Batch.Length, batch.Count);
        for (int i = 0; i < batch.Count; i++)
        {
            AssertEncoding(expected.Batch[i], batch[i]);
        }
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void AddedTokensAreTheSumOfTheChildren(string variant)
    {
        // Upstream num_special_tokens_to_add(is_pair) for false and true; the tokenizer refuses a
        // truncation length below it.
        int[] added = _expected[variant].Added;
        Tokenizer tokenizer = Load(Json(variant));
        int most = Math.Max(added[0], added[1]);
        if (most > 0)
        {
            Assert.Throws<ArgumentException>(() => tokenizer.WithTruncation(new TruncationOptions(most - 1)));
        }

        Assert.NotNull(tokenizer.WithTruncation(new TruncationOptions(most)));
    }

    [Fact]
    public void ReportsChildComponentsBeforeTheSequence()
    {
        Assert.Equal(
            ["normalizer:NFC", "pre_tokenizer:Split", "pre_tokenizer:ByteLevel", "pre_tokenizer:Sequence", "model:BPE", "added_tokens",
             "post_processor:ByteLevel", "post_processor:Sequence", "post_processor:TemplateProcessing", "post_processor:Sequence", "decoder:ByteLevel"],
            Load(Json("nested")).Components);
    }

    public static TheoryData<string, string> UnsupportedCases() => new()
    {
        { WithPost(Sequence(ByteLevel(), Template()).Also("extra", 1)), "post_processor.type=Sequence.extra" },
        { WithPost(Sequence(ByteLevel(), new JsonObject { ["type"] = "BertProcessing" })), "post_processor.processors[1].type=BertProcessing" },
        { WithPost(Sequence(Sequence(new JsonObject { ["type"] = "RobertaProcessing" }), Template())), "post_processor.processors[0].processors[0].type=RobertaProcessing" },
        { WithPost(Sequence(ByteLevel().Also("extra", true), Template())), "post_processor.processors[0].type=ByteLevel.extra" },
        { WithPost(Sequence(ByteLevel(), Template().Also("extra", 1))), "post_processor.processors[1].type=TemplateProcessing.extra" },
        { WithPost(Sequence(Template(), Template())), "post_processor.type=Sequence" },
        { WithPost(Sequence(Sequence(Template()), ByteLevel(), Sequence(Template()))), "post_processor.type=Sequence" },
    };

    [Theory]
    [MemberData(nameof(UnsupportedCases))]
    public void RejectsUnsupportedContentByPath(string json, string component)
    {
        UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(() => Load(json));
        Assert.Equal(component, e.Component);
    }

    public static TheoryData<string> MalformedCases() =>
    [
        WithPost(new JsonObject { ["type"] = "Sequence" }),                                   // no processors
        WithPost(new JsonObject { ["type"] = "Sequence", ["processors"] = new JsonObject() }), // not an array
        WithPost(Sequence(new JsonObject { ["add_prefix_space"] = false })),                  // child without a type
    ];

    [Theory]
    [MemberData(nameof(MalformedCases))]
    public void RejectsMalformedSequences(string json)
    {
        TokenizerFormatException e = Assert.Throws<TokenizerFormatException>(() => Load(json));
        Assert.IsNotType<UnsupportedTokenizerComponentException>(e);
    }

    private static string Json(string variant) => variant switch
    {
        "bytelevel_then_template" => WithPost(Sequence(ByteLevel(trimOffsets: true), Template())),
        "template_then_bytelevel" => WithPost(Sequence(Template(), ByteLevel(trimOffsets: true))),
        "nested" => WithPost(Sequence(Sequence(ByteLevel(trimOffsets: true)), Template())),
        "empty" => WithPost(Sequence()),
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    private static Tokenizer Load(string json) => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json));

    private static string WithPost(JsonObject post)
    {
        JsonObject root = JsonNode.Parse(_minimal)!.AsObject();
        root["post_processor"] = post;
        return root.ToJsonString();
    }

    private static JsonObject Sequence(params JsonObject[] processors) =>
        new() { ["type"] = "Sequence", ["processors"] = new JsonArray([.. processors]) };

    private static JsonObject ByteLevel(bool trimOffsets = false) =>
        new() { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = trimOffsets, ["use_regex"] = false };

    private static JsonObject Template() => JsonNode.Parse("""
        {
          "type": "TemplateProcessing",
          "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<|x|>", "type_id": 0}}],
          "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": "<|x|>", "type_id": 1}}],
          "special_tokens": {"<|x|>": {"id": "<|x|>", "ids": [7], "tokens": ["<|x|>"]}}
        }
        """)!.AsObject();

    private static void AssertEncoding(Expected expected, TokenEncoding actual)
    {
        Assert.Equal(expected.Ids, actual.Ids);
        Assert.Equal(expected.Tokens, actual.Tokens);
        Assert.Equal(expected.TypeIds, actual.TypeIds);
        Assert.Equal(expected.SpecialTokensMask, actual.SpecialTokensMask);
        Assert.Equal(expected.AttentionMask, actual.AttentionMask);
        // ASCII input: scalar and UTF-16 offsets coincide.
        TokenOffset[] offsets = [.. expected.Offsets.Select(o => new TokenOffset(o.Start, o.End))];
        Assert.Equal(offsets, actual.ScalarOffsets);
        Assert.Equal(offsets, actual.Offsets);
        Assert.Equal(expected.SequenceIds, actual.SequenceIds);
    }

    private sealed record Expected(
        int[] Ids, string[] Tokens, int[] TypeIds, int[] SpecialTokensMask, int[] AttentionMask, (int Start, int End)[] Offsets, int?[] SequenceIds);

    private sealed record Variant(
        int[] Added,
        Expected Single,
        Expected SinglePlain,
        Expected Pair,
        Expected PairPlain,
        Expected SingleTruncated,
        Expected SingleTruncatedPlain,
        Expected PairTruncated,
        Expected[] Batch);
}

internal static class JsonObjectExtensions
{
    public static JsonObject Also(this JsonObject o, string key, JsonNode? value)
    {
        o[key] = value;
        return o;
    }
}
