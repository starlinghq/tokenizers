using Starlings.Tokenizers;

namespace Starlings.Tokenizers.Tests;

/// <summary>
/// Profile-specific checks the data-driven <c>ProfileConformanceTests</c> cannot make: the declared
/// pipeline components and the GPT-2 split (use_regex true) path, plus golden encode cases.
/// </summary>
public sealed class WhisperBpeConformanceTests
{
    private static Tokenizer Load() => TestAssets.LoadTokenizer(TestAssets.WhisperTokenizerJson);

    [Fact]
    public void DeclaresTheWhisperPipeline()
    {
        Assert.Equal(
            ["pre_tokenizer:ByteLevel", "model:BPE", "added_tokens", "post_processor:TemplateProcessing", "decoder:ByteLevel"],
            Load().Components);
    }

    [Fact]
    public void EncodesHelloWorld()
    {
        TokenEncoding e = Load().Encode("Hello world");
        Assert.Equal([50258, 50364, 15947, 1002, 50257], e.Ids);
        Assert.Equal(["<|startoftranscript|>", "<|notimestamps|>", "Hello", "Ġworld", "<|endoftext|>"], e.Tokens);
    }

    [Fact]
    public void EncodesContraction()
    {
        TokenEncoding e = Load().Encode("don't stop");
        Assert.Equal([50258, 50364, 13966, 380, 1590, 50257], e.Ids);
        Assert.Equal(["<|startoftranscript|>", "<|notimestamps|>", "don", "'t", "Ġstop", "<|endoftext|>"], e.Tokens);
    }

    [Fact]
    public void EncodesLeadingSpaces()
    {
        TokenEncoding e = Load().Encode("  leading spaces");
        Assert.Equal([50258, 50364, 220, 5775, 7673, 50257], e.Ids);
    }

    [Fact]
    public void Gpt2SplitSeparatesNumbersAndPunctuation()
    {
        // The GPT-2 regex: " ?\p{N}+" and " ?[^\s\p{L}\p{N}]+" absorb their leading space.
        TokenEncoding e = Load().Encode("a 42 !");
        Assert.Equal(["<|startoftranscript|>", "<|notimestamps|>", "a", "Ġ42", "Ġ!", "<|endoftext|>"], e.Tokens);
    }
}
