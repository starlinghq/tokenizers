using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Profile phi4-bpe-chat against huggingface/tokenizers 0.23.3 on 1,500 seeded random strings that mix
/// the Phi-4-mini added tokens (ten of them <c>rstrip</c>, five of those not special), fragments of
/// them, whitespace runs and text (<c>TestData/o200k-bpe/phi4-differential.jsonl.gz</c>, written by
/// reference/o200k-bpe/run_o200k_reference.py --asset): ids, tokens, scalar offsets and decode. The
/// shared corpora have no Phi-4 tags, so this is the profile's evidence for <c>rstrip</c>.
/// </summary>
public class Phi4BpeDifferentialTests(ITestOutputHelper output)
{
    [Fact]
    public void MatchesReferenceOnCommittedDump()
    {
        Tokenizer tokenizer = TestAssets.LoadTokenizer(TestAssets.Phi4MiniTokenizerJson);
        int rows = 0;
        int rstripRows = 0;
        var failures = new List<string>();
        using var stream = new GZipStream(File.OpenRead(TestAssets.TestDataPath("o200k-bpe", "phi4-differential.jsonl.gz")), CompressionMode.Decompress);
        using var reader = new StreamReader(stream, new UTF8Encoding(false, true));
        while (reader.ReadLine() is { } line)
        {
            if (line.Length == 0)
            {
                continue;
            }

            rows++;
            using JsonDocument doc = JsonDocument.Parse(line);
            JsonElement row = doc.RootElement;
            string text = row.GetProperty("t").GetString()!;
            string label = JsonSerializer.Serialize(text);
            TokenEncoding enc = tokenizer.Encode(text);
            int[] ids = FixtureAssert.Ints(row.GetProperty("ids"));
            if (!ids.SequenceEqual(enc.Ids))
            {
                failures.Add($"{label}: ids expected [{string.Join(',', ids)}] actual [{string.Join(',', enc.Ids)}]");
                continue;
            }

            string[] tokens = [.. row.GetProperty("tokens").EnumerateArray().Select(e => e.GetString()!)];
            if (!tokens.SequenceEqual(enc.Tokens))
            {
                failures.Add($"{label}: tokens differ");
                continue;
            }

            for (int i = 0; i < tokens.Length; i++)
            {
                if (ids[i] >= 200019 && tokens[i].Length > tokens[i].TrimEnd().Length)
                {
                    rstripRows++;
                    break;
                }
            }

            int[] flat = FixtureAssert.Ints(row.GetProperty("o"));
            for (int i = 0; i < enc.Count; i++)
            {
                if (enc.ScalarOffsets[i] != new TokenOffset(flat[2 * i], flat[(2 * i) + 1]))
                {
                    failures.Add($"{label}: token {i} offsets expected ({flat[2 * i]}, {flat[(2 * i) + 1]}) actual {enc.ScalarOffsets[i]}");
                    break;
                }
            }

            Check(failures, label + " decode skip", row.GetProperty("skip").GetString(), tokenizer.Decode(enc.Ids.AsSpan(), skipSpecialTokens: true));
            Check(failures, label + " decode keep", row.GetProperty("keep").GetString(), tokenizer.Decode(enc.Ids.AsSpan(), skipSpecialTokens: false));
        }

        output.WriteLine($"{rows} strings, {rstripRows} with an rstrip token that absorbed whitespace, {failures.Count} mismatches");
        Assert.True(rows >= 1500, $"{rows} rows");
        Assert.True(rstripRows >= 100, $"only {rstripRows} rows exercise rstrip");
        Assert.True(failures.Count == 0, string.Join('\n', failures.Take(20)));
    }

    [Fact]
    public void RstripWithAWhitespaceLedTokenIsRejectedAtLoad()
    {
        // Upstream would resume matching inside the whitespace <a> absorbed and give overlapping splits.
        string json = """
            {"version":"1.0","truncation":null,"padding":null,
             "added_tokens":[
               {"id":2,"content":"<a>","single_word":false,"lstrip":false,"rstrip":true,"normalized":false,"special":true},
               {"id":3,"content":" b","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
             "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
             "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,
                      "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"x":0,"y":1},"merges":[]}}
            """;
        UnsupportedTokenizerComponentException e = Assert.Throws<UnsupportedTokenizerComponentException>(
            () => Tokenizer.FromUtf8Json(Encoding.UTF8.GetBytes(json)));
        Assert.Equal("added_tokens", e.Component);
    }

    private static void Check(List<string> failures, string label, string? expected, string actual)
    {
        if (!string.Equals(expected, actual, StringComparison.Ordinal))
        {
            failures.Add($"{label}: expected {JsonSerializer.Serialize(expected)} actual {JsonSerializer.Serialize(actual)}");
        }
    }
}
