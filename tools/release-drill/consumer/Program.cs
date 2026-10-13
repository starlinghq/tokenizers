// Clean-consumer check for the StarlingHq.Tokenizers release drill (docs/releasing.md).
// tools/release-drill.sh copies this file into a temporary directory outside the repository and
// builds it against the packed package from a local folder feed. It must compile and run with nothing but
// that package: no project references, no repository paths, no test framework.
//
//   dotnet run -- <tokenizer.json> <expected sha256> <fixture dir with tokenizer.jsonl and tokenizer-batch.json>
//
// Exit code 0 only if every check passes.
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using StarlingHq.Tokenizers;

if (args.Length != 3)
{
    Console.Error.WriteLine("usage: <tokenizer.json> <expected sha256> <fixture dir>");
    return 2;
}

string tokenizerPath = args[0];
string expectedSha256 = args[1];
string fixtureDir = args[2];
const int ExpectedCaseCount = 49;

Assembly library = typeof(Tokenizer).Assembly;
Console.WriteLine($"library:  {library.GetName().Name} {library.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion}");
Console.WriteLine($"location: {library.Location}");
Console.WriteLine($"runtime:  {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}");
string[] loaded = [.. AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetName().Name ?? "").Where(n => n.Contains("Tokenizers", StringComparison.Ordinal)).Order()];
Console.WriteLine($"loaded tokenizer assemblies: {string.Join(", ", loaded)}");

int failures = 0;
void Check(bool ok, string what)
{
    if (!ok)
    {
        failures++;
        Console.WriteLine($"FAIL: {what}");
    }
}

// Evidence requirements stay enforced: the pinned asset must be present and match its digest.
string actualSha256;
using (FileStream stream = File.OpenRead(tokenizerPath))
{
    actualSha256 = Convert.ToHexStringLower(SHA256.HashData(stream));
}

if (actualSha256 != expectedSha256)
{
    Console.WriteLine($"FAIL: {tokenizerPath} has SHA-256 {actualSha256}, expected {expectedSha256}");
    return 1;
}

Console.WriteLine($"asset:    {tokenizerPath} sha256 {actualSha256}");

Tokenizer tokenizer = Tokenizer.FromFile(tokenizerPath);

var cases = new List<JsonElement>();
foreach (string line in File.ReadAllText(Path.Combine(fixtureDir, "tokenizer.jsonl")).Split('\n'))
{
    if (line.Length > 0)
    {
        cases.Add(JsonDocument.Parse(line).RootElement.Clone());
    }
}

Check(cases.Count == ExpectedCaseCount, $"fixture has {cases.Count} cases, expected {ExpectedCaseCount}");
var byId = cases.ToDictionary(c => c.GetProperty("id").GetString()!);

int encodeExact = 0, decodeExact = 0, truncatedExact = 0, truncatedTotal = 0;
foreach (JsonElement row in cases)
{
    string id = row.GetProperty("id").GetString()!;
    TokenEncoding actual = Encode(tokenizer, row);
    if (EncodingMatches(row.GetProperty("encoding"), actual, out string? why))
    {
        encodeExact++;
    }
    else
    {
        Check(false, $"{id}: encode {why}");
    }

    int[] ids = Ints(row.GetProperty("encoding").GetProperty("ids"));
    JsonElement decode = row.GetProperty("decode");
    bool decodeOk = tokenizer.Decode(ids, skipSpecialTokens: true) == decode.GetProperty("skip_special_tokens").GetString()
        && tokenizer.Decode(ids, skipSpecialTokens: false) == decode.GetProperty("keep_special_tokens").GetString();
    Check(decodeOk, $"{id}: decode");
    decodeExact += decodeOk ? 1 : 0;

    if (row.TryGetProperty("truncated", out JsonElement truncated))
    {
        truncatedTotal++;
        Tokenizer truncating = tokenizer.WithTruncation(new TruncationOptions(truncated.GetProperty("max_length").GetInt32()));
        if (EncodingMatches(truncated.GetProperty("encoding"), Encode(truncating, row), out why))
        {
            truncatedExact++;
        }
        else
        {
            Check(false, $"{id}: truncated encode {why}");
        }
    }
}

Console.WriteLine($"encode:   {encodeExact}/{cases.Count} cases exact (ids, tokens, type ids, masks, both offset units, sequence ids)");
Console.WriteLine($"decode:   {decodeExact}/{cases.Count} cases exact (skip and keep special tokens)");
Console.WriteLine($"truncate: {truncatedExact}/{truncatedTotal} cases exact");
Check(truncatedTotal == 2, $"expected 2 truncation cases, found {truncatedTotal}");

// Padded batch.
using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtureDir, "tokenizer-batch.json"))))
{
    int padId = doc.RootElement.GetProperty("padding").GetProperty("pad_id").GetInt32();
    JsonElement[] rows = [.. doc.RootElement.GetProperty("rows").EnumerateArray()];
    string[] texts = [.. rows.Select(r => byId[r.GetProperty("id").GetString()!].GetProperty("text").GetString()!)];
    IReadOnlyList<TokenEncoding> batch = tokenizer.WithPadding(new PaddingOptions(padId, tokenizer.IdToToken(padId)!)).EncodeBatch(texts);
    int batchExact = 0;
    for (int i = 0; i < rows.Length; i++)
    {
        bool ok = batch[i].Ids.SequenceEqual(Ints(rows[i].GetProperty("ids")))
            && batch[i].AttentionMask.SequenceEqual(Ints(rows[i].GetProperty("attention_mask")));
        Check(ok, $"batch row {i}");
        batchExact += ok ? 1 : 0;
    }

    Console.WriteLine($"batch:    {batchExact}/{rows.Length} padded rows exact");
}

// Owned fix shipped without an upstream release: consecutive unknown pieces fuse into one <unk> (PROVENANCE L5,
// the reference behavior). The adopted upstream source emits one <unk> per character; no upstream PR is merged.
foreach (string id in new[] { "unknown-char", "combining-stack" })
{
    if (byId.TryGetValue(id, out JsonElement row))
    {
        TokenEncoding e = tokenizer.Encode(row.GetProperty("text").GetString()!);
        Console.WriteLine($"L5 case {id}: ids [{string.Join(",", e.Ids)}]");
    }
}

Console.WriteLine(failures == 0 ? "RESULT: PASS" : $"RESULT: FAIL ({failures} failures)");
return failures == 0 ? 0 : 1;

static TokenEncoding Encode(Tokenizer tokenizer, JsonElement row)
{
    string text = row.GetProperty("text").GetString()!;
    return row.TryGetProperty("pair", out JsonElement pair) ? tokenizer.Encode(text, pair.GetString()!) : tokenizer.Encode(text);
}

static int[] Ints(JsonElement array) => [.. array.EnumerateArray().Select(e => e.GetInt32())];

static TokenOffset[] Offsets(JsonElement array) => [.. array.EnumerateArray().Select(p => new TokenOffset(p[0].GetInt32(), p[1].GetInt32()))];

static bool EncodingMatches(JsonElement expected, TokenEncoding actual, out string? why)
{
    why = null;
    if (!actual.Ids.SequenceEqual(Ints(expected.GetProperty("ids"))))
    {
        why = "ids";
    }
    else if (!actual.Tokens.SequenceEqual(expected.GetProperty("tokens").EnumerateArray().Select(e => e.GetString()!)))
    {
        why = "tokens";
    }
    else if (!actual.TypeIds.SequenceEqual(Ints(expected.GetProperty("type_ids"))))
    {
        why = "type_ids";
    }
    else if (!actual.AttentionMask.SequenceEqual(Ints(expected.GetProperty("attention_mask"))))
    {
        why = "attention_mask";
    }
    else if (!actual.SpecialTokensMask.SequenceEqual(Ints(expected.GetProperty("special_tokens_mask"))))
    {
        why = "special_tokens_mask";
    }
    else if (!actual.Offsets.SequenceEqual(Offsets(expected.GetProperty("offsets").GetProperty("utf16-code-units"))))
    {
        why = "utf16 offsets";
    }
    else if (!actual.ScalarOffsets.SequenceEqual(Offsets(expected.GetProperty("offsets").GetProperty("unicode-scalars"))))
    {
        why = "scalar offsets";
    }
    else if (expected.TryGetProperty("sequence_ids", out JsonElement seq)
        && !actual.SequenceIds.SequenceEqual(seq.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.Null ? (int?)null : e.GetInt32())))
    {
        why = "sequence_ids";
    }

    return why is null;
}
