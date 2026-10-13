using System.Text;
using System.Text.Json;
using StarlingHq.Tokenizers.Pipeline;

namespace StarlingHq.Tokenizers.Tests;

/// <summary>
/// The components the Gemma BPE profile added (BPE byte fallback and unknown token, <c>Split</c> with a
/// <c>String</c> pattern in every behavior and inverted, the <c>Sequence</c>, <c>Replace</c>,
/// <c>ByteFallback</c> and <c>Fuse</c> decoders) on small synthetic tokenizers, without the asset.
/// Expected values are tokenizers 0.23.3 output recorded in <c>TestData/gemma-bpe/components.json</c>
/// by reference/gemma-bpe/run_gemma_bpe_tokenizer.py --components-out; the file also
/// holds the tokenizer documents.
/// </summary>
public class GemmaBpeComponentTests
{
    private static readonly Lazy<JsonDocument> _reference = new(() =>
        JsonDocument.Parse(File.ReadAllText(TestAssets.TestDataPath("gemma-bpe", "components.json"))));

    private static readonly Dictionary<string, Tokenizer> _tokenizers = [];
    private static readonly Lock _lock = new();

    public static TheoryData<int> CaseIndexes()
    {
        var data = new TheoryData<int>();
        int n = _reference.Value.RootElement.GetProperty("cases").GetArrayLength();
        for (int i = 0; i < n; i++)
        {
            data.Add(i);
        }

        return data;
    }

    private static Tokenizer Get(string name)
    {
        lock (_lock)
        {
            if (!_tokenizers.TryGetValue(name, out Tokenizer? t))
            {
                string json = _reference.Value.RootElement.GetProperty("tokenizers").GetProperty(name).GetRawText();
                t = Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json));
                _tokenizers[name] = t;
            }

            return t;
        }
    }

    [Fact]
    public void ReferenceCoversEveryVariant()
    {
        JsonElement root = _reference.Value.RootElement;
        Assert.StartsWith("tokenizers 0.23.3", root.GetProperty("reference").GetString(), StringComparison.Ordinal);
        string[] names = [.. root.GetProperty("tokenizers").EnumerateObject().Select(p => p.Name)];
        Assert.Equal(11, names.Length);
        Assert.All(names, n => Assert.Contains(root.GetProperty("cases").EnumerateArray(), c => c.GetProperty("tokenizer").GetString() == n));
    }

    [Theory]
    [MemberData(nameof(CaseIndexes))]
    public void MatchesReference(int index)
    {
        JsonElement c = _reference.Value.RootElement.GetProperty("cases")[index];
        Tokenizer tokenizer = Get(c.GetProperty("tokenizer").GetString()!);
        if (c.TryGetProperty("decode_ids", out JsonElement decodeIds))
        {
            int[] ids = FixtureAssert.Ints(decodeIds);
            Assert.Equal(c.GetProperty("keep").GetString(), tokenizer.Decode(ids, skipSpecialTokens: false));
            Assert.Equal(c.GetProperty("skip").GetString(), tokenizer.Decode(ids, skipSpecialTokens: true));
            return;
        }

        string text = c.GetProperty("text").GetString()!;
        TokenEncoding enc = tokenizer.Encode(text);
        Assert.Equal(FixtureAssert.Ints(c.GetProperty("ids")), enc.Ids);
        Assert.Equal(c.GetProperty("tokens").EnumerateArray().Select(t => t.GetString()!), enc.Tokens);
        TokenOffset[] scalar = [.. c.GetProperty("offsets").EnumerateArray().Select(o => new TokenOffset(o[0].GetInt32(), o[1].GetInt32()))];
        Assert.Equal(scalar, enc.ScalarOffsets);
        Assert.Equal(scalar.Select(o => GemmaBpeDifferentialTests.ScalarToUtf16(text, o)), enc.Offsets);
        Assert.Equal(c.GetProperty("decode").GetString(), tokenizer.Decode(enc.Ids.AsSpan(), skipSpecialTokens: false));
    }

    [Fact]
    public void ByteFallbackTokensCarryTheCharacterOffsets()
    {
        // "é" is not in the vocabulary: two byte tokens, both with the offsets of the character
        // (UTF-16 (1, 2)); "😀" (two UTF-16 units) gives four byte tokens with offsets (2, 4).
        Tokenizer tokenizer = Get("base");
        TokenEncoding enc = tokenizer.Encode("aé\U0001F600");
        Assert.Equal(["a", "<0xC3>", "<0xA9>", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"], enc.Tokens);
        Assert.Equal(
            [new(0, 1), new(1, 2), new(1, 2), new(2, 4), new(2, 4), new(2, 4), new(2, 4)],
            enc.Offsets);
        Assert.Equal([new(0, 1), new(1, 2), new(1, 2), new(2, 3), new(2, 3), new(2, 3), new(2, 3)], enc.ScalarOffsets);
    }

    [Theory]
    [InlineData("<0x41>", true, 0x41)]
    [InlineData("<0xff>", true, 0xFF)]
    [InlineData("<0xFf>", true, 0xFF)]
    [InlineData("<0x+A>", true, 0x0A)]
    [InlineData("<0x+g>", false, 0)]
    [InlineData("<0x-1>", false, 0)]
    [InlineData("<0x++>", false, 0)]
    [InlineData("<0x4>", false, 0)]
    [InlineData("<0x041>", false, 0)]
    [InlineData("<0X41>", false, 0)]
    [InlineData("<0xé>", false, 0)]
    [InlineData("<0x4é", false, 0)]
    [InlineData("(0x41>", false, 0)]
    public void ByteTokenParsingFollowsTheReference(string token, bool isByte, int value)
    {
        Assert.Equal(isByte, ByteFallbackDecoder.TryParseByteToken(token, out byte b));
        Assert.Equal(value, b);
    }

    [Fact]
    public void ByteFallbackTokenNamesAreUpperCaseHex()
    {
        Assert.Equal("<0x0A>", BpeModel.ByteFallbackToken(0x0A));
        Assert.Equal("<0xFF>", BpeModel.ByteFallbackToken(0xFF));
    }
}
