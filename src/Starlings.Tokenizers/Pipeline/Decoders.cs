// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3):
// tokenizers/src/decoders/{sequence,byte_fallback,fuse}.rs and tokenizers/src/normalizers/replace.rs
// (Decoder impl), Copyright The HuggingFace Inc. team, Apache License 2.0 (see
// THIRD-PARTY-NOTICES.md). Modified by StarlingHq: translated to C#.

using System.Text;

namespace Starlings.Tokenizers.Pipeline;

/// <summary><c>Sequence</c> decoder: each decoder's <c>decode_chain</c> in order.</summary>
internal sealed class SequenceDecoder(IReadOnlyList<IDecoder> decoders) : IDecoder
{
    public List<string> DecodeChain(List<string> tokens)
    {
        foreach (IDecoder decoder in decoders)
        {
            tokens = decoder.DecodeChain(tokens);
        }

        return tokens;
    }
}

/// <summary><c>Replace</c> decoder: in every token, each pattern match becomes the content.</summary>
internal sealed class ReplaceDecoder(IPattern pattern, string content) : IDecoder
{
    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        var builder = new StringBuilder();
        foreach (string token in tokens)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(token);
            builder.Clear();
            foreach (PatternMatch m in pattern.FindMatches(utf8))
            {
                if (m.IsMatch)
                {
                    builder.Append(content);
                }
                else
                {
                    builder.Append(Encoding.UTF8.GetString(utf8, m.Start, m.End - m.Start));
                }
            }

            result.Add(builder.ToString());
        }

        return result;
    }
}

/// <summary>
/// <c>ByteFallback</c> decoder: each run of consecutive byte tokens (<c>&lt;0xNN&gt;</c>) becomes
/// the string those bytes spell if they are valid UTF-8, else one U+FFFD per byte token (not per
/// ill-formed subsequence, unlike the <c>ByteLevel</c> decoder). Other tokens pass through.
/// </summary>
internal sealed class ByteFallbackDecoder : IDecoder
{
    private const string ReplacementCharacter = "�";
    private static readonly UTF8Encoding _strictUtf8 = new(false, true);

    public List<string> DecodeChain(List<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        var pending = new List<byte>();
        foreach (string token in tokens)
        {
            if (TryParseByteToken(token, out byte b))
            {
                pending.Add(b);
                continue;
            }

            Flush(pending, result);
            result.Add(token);
        }

        Flush(pending, result);
        return result;
    }

    /// <summary>
    /// Upstream: a token of exactly 6 UTF-8 bytes that starts with <c>&lt;0x</c> and ends with
    /// <c>&gt;</c>, whose two middle bytes parse with Rust <c>u8::from_str_radix(_, 16)</c>: two hex
    /// digits of either case, or <c>+</c> and one hex digit.
    /// </summary>
    internal static bool TryParseByteToken(string token, out byte value)
    {
        value = 0;

        // 6 UTF-8 bytes with an ASCII "<0x" prefix and ">" suffix: the middle is two ASCII
        // characters (length 6) or one two-byte character (length 5, never a valid number).
        if (token.Length != 6 || !token.StartsWith("<0x", StringComparison.Ordinal) || token[5] != '>')
        {
            return false;
        }

        char hi = token[3];
        char lo = token[4];
        int low = HexValue(lo);
        if (low < 0)
        {
            return false;
        }

        if (hi == '+')
        {
            value = (byte)low;
            return true;
        }

        int high = HexValue(hi);
        if (high < 0)
        {
            return false;
        }

        value = (byte)((high << 4) | low);
        return true;
    }

    private static int HexValue(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'f' => c - 'a' + 10,
        >= 'A' and <= 'F' => c - 'A' + 10,
        _ => -1,
    };

    private static void Flush(List<byte> pending, List<string> result)
    {
        if (pending.Count == 0)
        {
            return;
        }

        string? text;
        try
        {
            text = _strictUtf8.GetString([.. pending]);
        }
        catch (DecoderFallbackException)
        {
            text = null;
        }

        if (text is not null)
        {
            result.Add(text);
        }
        else
        {
            for (int i = 0; i < pending.Count; i++)
            {
                result.Add(ReplacementCharacter);
            }
        }

        pending.Clear();
    }
}

/// <summary><c>Fuse</c> decoder: all tokens joined into one.</summary>
internal sealed class FuseDecoder : IDecoder
{
    public List<string> DecodeChain(List<string> tokens) => [string.Concat(tokens)];
}
