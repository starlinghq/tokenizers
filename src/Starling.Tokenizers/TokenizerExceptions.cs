// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

namespace Starling.Tokenizers;

/// <summary>A <c>tokenizer.json</c> document is malformed or internally inconsistent.</summary>
public class TokenizerFormatException : Exception
{
    /// <summary>Creates the exception.</summary>
    public TokenizerFormatException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public TokenizerFormatException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public TokenizerFormatException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>
/// A <c>tokenizer.json</c> uses a component, option or value this library does not implement.
/// The library never approximates an unsupported component.
/// </summary>
public sealed class UnsupportedTokenizerComponentException : TokenizerFormatException
{
    /// <summary>Creates the exception.</summary>
    public UnsupportedTokenizerComponentException()
    {
        Component = string.Empty;
    }

    /// <summary>Creates the exception with a message.</summary>
    public UnsupportedTokenizerComponentException(string message)
        : base(message)
    {
        Component = string.Empty;
    }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public UnsupportedTokenizerComponentException(string message, Exception innerException)
        : base(message, innerException)
    {
        Component = string.Empty;
    }

    /// <summary>Creates the exception for a named component.</summary>
    /// <param name="component">JSON path of the component, for example <c>pre_tokenizer.type=ByteLevel</c>.</param>
    /// <param name="detail">What exactly is unsupported.</param>
    public UnsupportedTokenizerComponentException(string component, string detail)
        : base($"Unsupported tokenizer.json component '{component}': {detail}")
    {
        Component = component;
    }

    /// <summary>JSON path and type tag of the unsupported component.</summary>
    public string Component { get; }
}
