// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

namespace Starlings.Tokenizers.Pipeline;

/// <summary>The <c>model</c> stage: turns one pre-token (normalized UTF-8) into tokens.</summary>
/// <remarks>Mirrors the parts of upstream's <c>Model</c> trait the interpreter uses.</remarks>
internal interface IModel
{
    /// <summary>Upstream <c>get_vocab_size</c>: number of entries in the model vocabulary.</summary>
    int VocabularySize { get; }

    int? TokenToId(string token);

    string? IdToToken(int id);

    /// <summary>Tokens with byte offsets into <paramref name="sequence"/>.</summary>
    List<ModelToken> Tokenize(ReadOnlySpan<byte> sequence);
}
