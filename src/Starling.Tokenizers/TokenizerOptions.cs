// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

namespace Starling.Tokenizers;

/// <summary>Which side tokens are removed from when truncating.</summary>
public enum TruncationDirection
{
    /// <summary>Keep the start, drop tokens at the end.</summary>
    Right,

    /// <summary>Keep the end, drop tokens at the start.</summary>
    Left,
}

/// <summary>How the token budget is shared between the two texts of a pair.</summary>
public enum TruncationStrategy
{
    /// <summary>Truncate the longer sequence first (reference default).</summary>
    LongestFirst,

    /// <summary>Truncate only the first sequence.</summary>
    OnlyFirst,

    /// <summary>Truncate only the second sequence.</summary>
    OnlySecond,
}

/// <summary>Truncation settings, equivalent to the <c>truncation</c> section of <c>tokenizer.json</c>.</summary>
/// <param name="MaxLength">Maximum number of tokens, including special tokens added by post-processing.</param>
/// <param name="Strategy">How a pair shares the budget.</param>
/// <param name="Direction">Which side is dropped.</param>
/// <remarks>A stride (overflowing windows) is not supported; a <c>tokenizer.json</c> with a
/// non-zero stride is rejected.</remarks>
public sealed record TruncationOptions(
    int MaxLength,
    TruncationStrategy Strategy = TruncationStrategy.LongestFirst,
    TruncationDirection Direction = TruncationDirection.Right);

/// <summary>Which side padding tokens are added on.</summary>
public enum PaddingDirection
{
    /// <summary>Append padding after the tokens.</summary>
    Right,

    /// <summary>Insert padding before the tokens.</summary>
    Left,
}

/// <summary>Padding settings, equivalent to the <c>padding</c> section of <c>tokenizer.json</c>.</summary>
/// <param name="PadId">Id of the padding token.</param>
/// <param name="PadToken">String of the padding token.</param>
/// <param name="FixedLength">Pad every encoding to this length; null pads a batch to its longest encoding.</param>
/// <param name="Direction">Which side padding goes on.</param>
/// <param name="PadTypeId">Type id given to padding tokens.</param>
/// <param name="PadToMultipleOf">If set, round the padded length up to a multiple of this value.</param>
public sealed record PaddingOptions(
    int PadId,
    string PadToken,
    int? FixedLength = null,
    PaddingDirection Direction = PaddingDirection.Right,
    int PadTypeId = 0,
    int? PadToMultipleOf = null);
