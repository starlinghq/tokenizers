// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Buffers;
using System.Text;

namespace Starling.Tokenizers.Pipeline;

/// <summary>UTF-8 helpers over well-formed buffers produced by this library.</summary>
internal static class Utf8Util
{
    public static int SequenceLength(ReadOnlySpan<byte> utf8, int index)
    {
        byte b = utf8[index];
        return b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
    }

    public static bool IsCharBoundary(ReadOnlySpan<byte> utf8, int index) =>
        index == 0 || index == utf8.Length || (index < utf8.Length && (utf8[index] & 0xC0) != 0x80);

    public static Rune DecodeAt(ReadOnlySpan<byte> utf8, int index, out int length)
    {
        OperationStatus status = Rune.DecodeFromUtf8(utf8[index..], out Rune rune, out length);
        if (status != OperationStatus.Done)
        {
            throw new InvalidOperationException("Invalid UTF-8 in a normalized string.");
        }

        return rune;
    }

    public static List<Rune> DecodeRunes(ReadOnlySpan<byte> utf8)
    {
        var runes = new List<Rune>(utf8.Length);
        int i = 0;
        while (i < utf8.Length)
        {
            runes.Add(DecodeAt(utf8, i, out int len));
            i += len;
        }

        return runes;
    }

    /// <summary>Encodes <paramref name="text"/> as UTF-8, rejecting unpaired surrogates.</summary>
    /// <remarks>The reference operates on Rust <c>str</c>, which cannot hold unpaired
    /// surrogates; replacing them with U+FFFD would change the input silently.</remarks>
    public static byte[] EncodeStrict(string text, string paramName)
    {
        if (!IsWellFormed(text))
        {
            throw new ArgumentException("Input contains an unpaired UTF-16 surrogate and is not valid Unicode text.", paramName);
        }

        return Encoding.UTF8.GetBytes(text);
    }

    private static bool IsWellFormed(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (char.IsHighSurrogate(c))
            {
                if (i + 1 >= text.Length || !char.IsLowSurrogate(text[i + 1]))
                {
                    return false;
                }

                i++;
            }
            else if (char.IsLowSurrogate(c))
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>Unicode property checks matching Rust <c>char</c> methods used upstream.</summary>
internal static class UnicodeData
{
    /// <summary>Rust <c>char::is_whitespace</c>: the Unicode White_Space property.</summary>
    public static bool IsWhiteSpace(Rune r) => r.Value switch
    {
        >= 0x09 and <= 0x0D => true,
        0x20 or 0x85 or 0xA0 or 0x1680 => true,
        >= 0x2000 and <= 0x200A => true,
        0x2028 or 0x2029 or 0x202F or 0x205F or 0x3000 => true,
        _ => false,
    };
}
