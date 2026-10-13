#:project ../src/Starlings.Tokenizers/Starlings.Tokenizers.csproj
#:property PublishAot=false

// Does a tokenizer.json load, and which components does it use? The first step of adding a profile
// (docs/adding-a-tokenizer.md). Prints the pipeline, vocabulary sizes and a sample encode, or the
// unsupported component.
//
//   dotnet run tools/inspect.cs -- <tokenizer.json> ["sample text"]
using System.Security.Cryptography;
using Starlings.Tokenizers;

if (args.Length is < 1 or > 2)
{
    Console.Error.WriteLine("usage: dotnet run tools/inspect.cs -- <tokenizer.json> [\"sample text\"]");
    return 2;
}

string path = args[0];
string sample = args.Length == 2 ? args[1] : "Hello, world! Grüße aus Köln 👋 1234";
byte[] bytes = File.ReadAllBytes(path);
Console.WriteLine($"file:       {path}");
Console.WriteLine($"sha256:     {Convert.ToHexStringLower(SHA256.HashData(bytes))}");
Console.WriteLine($"size:       {bytes.Length}");

Tokenizer tokenizer;
try
{
    tokenizer = Tokenizer.FromUtf8Json(bytes);
}
catch (UnsupportedTokenizerComponentException ex)
{
    Console.WriteLine($"result:     UNSUPPORTED {ex.Component}");
    Console.WriteLine($"detail:     {ex.Message}");
    return 1;
}
catch (TokenizerFormatException ex)
{
    Console.WriteLine($"result:     MALFORMED {ex.Message}");
    return 1;
}

Console.WriteLine("result:     loads");
Console.WriteLine($"components: {string.Join(", ", tokenizer.Components)}");
Console.WriteLine($"vocabulary: {tokenizer.GetVocabularySize(includeAddedTokens: false)} model, {tokenizer.GetVocabularySize(includeAddedTokens: true)} with added tokens");
Console.WriteLine($"truncation: {tokenizer.Truncation?.ToString() ?? "none"}");
Console.WriteLine($"padding:    {tokenizer.Padding?.ToString() ?? "none"}");
TokenEncoding encoding = tokenizer.Encode(sample);
Console.WriteLine($"sample:     {sample}");
Console.WriteLine($"ids:        {string.Join(' ', encoding.Ids)}");
Console.WriteLine($"tokens:     {string.Join(' ', encoding.Tokens)}");
Console.WriteLine($"decode:     {tokenizer.Decode(encoding.Ids.AsSpan(), skipSpecialTokens: true)}");
return 0;
