// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from huggingface/tokenizers v0.23.2 (88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3):
// tokenizers/src/pre_tokenizers/{metaspace,sequence,split,byte_level}.rs, Copyright The HuggingFace Inc. team,
// Apache License 2.0 (see THIRD-PARTY-NOTICES.md). Modified by StarlingHq: translated to C#.

using System.Text;

namespace StarlingHq.Tokenizers.Pipeline;

internal interface IPreTokenizer
{
    void PreTokenize(PreTokenizedString pretokenized);
}

/// <summary><c>Sequence</c>: applies each pre-tokenizer in order.</summary>
internal sealed class SequencePreTokenizer(IReadOnlyList<IPreTokenizer> preTokenizers) : IPreTokenizer
{
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        foreach (IPreTokenizer p in preTokenizers)
        {
            p.PreTokenize(pretokenized);
        }
    }
}

/// <summary>Metaspace <c>prepend_scheme</c>.</summary>
internal enum PrependScheme
{
    First,
    Never,
    Always,
}

/// <summary><c>Metaspace</c>: replaces spaces with the replacement character, prepends it
/// according to the scheme, and optionally splits before each replacement character.</summary>
internal sealed class MetaspacePreTokenizer(Rune replacement, PrependScheme prependScheme, bool split) : IPreTokenizer
{
    private static readonly CharPattern _space = new(new Rune(' '));
    private readonly string _replacementString = replacement.ToString();
    private readonly CharPattern _replacementPattern = new(replacement);

    public void PreTokenize(PreTokenizedString pretokenized) =>
        pretokenized.SplitEach(normalized =>
        {
            normalized.Replace(_space.FindMatches(normalized.Normalized), _replacementString);
            bool startsWithReplacement = !normalized.IsEmpty
                && Utf8Util.DecodeAt(normalized.Normalized, 0, out _) == replacement;
            switch (prependScheme)
            {
                case PrependScheme.Always when !startsWithReplacement:
                    normalized.Prepend(_replacementString);
                    break;
                case PrependScheme.First when !startsWithReplacement && normalized.OffsetsOriginal().Start == 0:
                    normalized.Prepend(_replacementString);
                    break;
            }

            if (!split)
            {
                return [new Split(normalized)];
            }

            return normalized
                .Split(_replacementPattern.FindMatches(normalized.Normalized), SplitDelimiterBehavior.MergedWithNext)
                .Select(n => new Split(n));
        });
}

/// <summary><c>Split</c> (upstream <c>tokenizers/src/pre_tokenizers/split.rs</c>) with
/// <c>invert</c> false; the loader accepts only the Qwen regexes with <c>Isolated</c>.</summary>
internal sealed class SplitPreTokenizer(IPattern pattern, SplitDelimiterBehavior behavior) : IPreTokenizer
{
    public void PreTokenize(PreTokenizedString pretokenized) =>
        pretokenized.SplitEach(normalized =>
            normalized.Split(pattern.FindMatches(normalized.Normalized), behavior).Select(n => new Split(n)));
}

/// <summary>
/// <c>ByteLevel</c> pre-tokenizer with <c>use_regex</c> false (upstream
/// <c>tokenizers/src/pre_tokenizers/byte_level.rs</c>): optionally prepends a space to each piece,
/// then maps every UTF-8 byte of each piece to its byte-level character. <c>trim_offsets</c> has
/// no effect in this stage.
/// </summary>
internal sealed class ByteLevelPreTokenizer(bool addPrefixSpace) : IPreTokenizer
{
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        if (addPrefixSpace)
        {
            pretokenized.SplitEach(normalized =>
            {
                if (normalized.IsEmpty || normalized.Normalized[0] != (byte)' ')
                {
                    normalized.Prepend(" ");
                }

                return [new Split(normalized)];
            });
        }

        pretokenized.NormalizeEach(normalized => normalized.ByteLevelTransform());
    }
}

/// <summary>GPT-2's byte-to-character table (upstream <c>bytes_char</c>).</summary>
internal static class ByteLevelAlphabet
{
    /// <summary>Character of each byte.</summary>
    public static readonly char[] ByteToChar = BuildByteToChar();

    /// <summary>Byte of each character below U+0200, or -1.</summary>
    public static readonly short[] CharToByte = BuildCharToByte();

    private static char[] BuildByteToChar()
    {
        var map = new char[256];
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            bool direct = b is (>= '!' and <= '~') or (>= 0xA1 and <= 0xAC) or (>= 0xAE and <= 0xFF);
            map[b] = direct ? (char)b : (char)(256 + n++);
        }

        return map;
    }

    private static short[] BuildCharToByte()
    {
        var map = new short[0x200];
        map.AsSpan().Fill(-1);
        for (int b = 0; b < 256; b++)
        {
            map[ByteToChar[b]] = (short)b;
        }

        return map;
    }
}
