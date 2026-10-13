// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Collections.Immutable;

namespace Starlings.Tokenizers;

/// <summary>An exclusive-end span of the input text.</summary>
/// <param name="Start">First position covered.</param>
/// <param name="End">Position after the last one covered.</param>
public readonly record struct TokenOffset(int Start, int End)
{
    /// <summary>Number of positions covered.</summary>
    public int Length => End - Start;
}

/// <summary>
/// The result of encoding one input (a single text or a text pair). All arrays have the same
/// length, one entry per token, including special and padding tokens.
/// </summary>
/// <remarks>
/// Offsets index the input string the token came from: the first text for tokens of sequence 0
/// and the pair text for tokens of sequence 1. Special and padding tokens have offsets (0, 0).
/// </remarks>
public sealed class TokenEncoding
{
    internal TokenEncoding(
        ImmutableArray<int> ids,
        ImmutableArray<string> tokens,
        ImmutableArray<int> typeIds,
        ImmutableArray<int> attentionMask,
        ImmutableArray<int> specialTokensMask,
        ImmutableArray<TokenOffset> offsets,
        ImmutableArray<TokenOffset> scalarOffsets,
        ImmutableArray<int?> sequenceIds)
    {
        Ids = ids;
        Tokens = tokens;
        TypeIds = typeIds;
        AttentionMask = attentionMask;
        SpecialTokensMask = specialTokensMask;
        Offsets = offsets;
        ScalarOffsets = scalarOffsets;
        SequenceIds = sequenceIds;
    }

    /// <summary>Number of tokens.</summary>
    public int Count => Ids.Length;

    /// <summary>Token ids.</summary>
    public ImmutableArray<int> Ids { get; }

    /// <summary>Token strings as the model's vocabulary spells them.</summary>
    public ImmutableArray<string> Tokens { get; }

    /// <summary>Token type (segment) ids.</summary>
    public ImmutableArray<int> TypeIds { get; }

    /// <summary>1 for real tokens, 0 for padding.</summary>
    public ImmutableArray<int> AttentionMask { get; }

    /// <summary>1 for tokens added by post-processing or padding, 0 for tokens from the text.</summary>
    /// <remarks>Special tokens written literally in the input text are matched as added tokens
    /// but are text tokens, so they have 0 here, as in the reference.</remarks>
    public ImmutableArray<int> SpecialTokensMask { get; }

    /// <summary>Offsets in UTF-16 code units: use them to slice the C# input string.</summary>
    public ImmutableArray<TokenOffset> Offsets { get; }

    /// <summary>Offsets in Unicode scalar values, the reference (Python <c>tokenizers</c>) unit.</summary>
    public ImmutableArray<TokenOffset> ScalarOffsets { get; }

    /// <summary>Which input each token came from (0 or 1), or null for tokens added by the
    /// post-processor template and for padding (reference <c>sequence_ids</c>).</summary>
    public ImmutableArray<int?> SequenceIds { get; }
}
