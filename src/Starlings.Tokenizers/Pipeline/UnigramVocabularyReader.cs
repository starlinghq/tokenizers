// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Unicode;

namespace Starlings.Tokenizers.Pipeline;

/// <summary>
/// Fast path for the largest part of a <c>tokenizer.json</c>: the Unigram <c>model.vocab</c>
/// array of <c>[piece, score]</c> pairs (250,002 of them, 98% of the bytes, in the e5 asset).
/// </summary>
/// <remarks>
/// <para>
/// The vocabulary is read straight from the UTF-8 bytes, and the rest of the document is handed to
/// the document model with the vocab array replaced by <c>[]</c>. The fast path accepts only what
/// the document path would read to the same values, and declines (returns false) on anything else:
/// then the caller loads the unmodified file through the document model, so malformed input fails
/// with exactly the errors it always did.
/// </para>
/// <para>
/// Accepted entries are <c>[string, number]</c> with JSON whitespace anywhere between tokens. A
/// string without escapes must be well-formed UTF-8 without control characters (JSON forbids them
/// unescaped); a string with escapes is decoded by <see cref="Utf8JsonReader"/> on that string
/// alone, which is how the document path decodes it. A number must match the JSON number grammar
/// and is parsed by <see cref="SerdeJsonNumber"/>, as the document path does. An entry outside
/// this subset is read by <see cref="Utf8JsonReader"/>, and if that is unusual too the fast path
/// declines. The caller must check, on the document, that <c>model</c> and <c>model.vocab</c> each
/// occur once (<see cref="HasSingleVocab"/>), so the array read here is the one the document path
/// would read.
/// </para>
/// </remarks>
internal static class UnigramVocabularyReader
{
    private static readonly JsonReaderOptions _readerOptions = new() { MaxDepth = 64 };

    /// <summary>
    /// Reads the first root <c>model</c> object's first <c>vocab</c> array. On success returns the
    /// vocabulary and a copy of the document with that array replaced by <c>[]</c>.
    /// </summary>
    public static bool TryRead(ReadOnlySpan<byte> json, out UnigramVocabulary? vocab, out byte[]? documentWithoutVocab)
    {
        vocab = null;
        documentWithoutVocab = null;
        try
        {
            if (!TryFindVocab(json, out int start, out int end, out vocab))
            {
                return false;
            }

            var rest = new byte[json.Length - (end - start) + 2];
            json[..start].CopyTo(rest);
            rest[start] = (byte)'[';
            rest[start + 1] = (byte)']';
            json[end..].CopyTo(rest.AsSpan(start + 2));
            documentWithoutVocab = rest;
            return true;
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or TokenizerFormatException)
        {
            vocab = null;
            return false;
        }
    }

    /// <summary>True if the root has exactly one <c>model</c> key, an object with exactly one <c>vocab</c> key.</summary>
    public static bool HasSingleVocab(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            return false;
        }

        JsonElement? model = null;
        foreach (JsonProperty p in root.EnumerateObject())
        {
            if (p.NameEquals("model"u8))
            {
                if (model is not null)
                {
                    return false;
                }

                model = p.Value;
            }
        }

        if (model is not { ValueKind: JsonValueKind.Object } m)
        {
            return false;
        }

        int vocabKeys = 0;
        foreach (JsonProperty p in m.EnumerateObject())
        {
            if (p.NameEquals("vocab"u8))
            {
                vocabKeys++;
            }
        }

        return vocabKeys == 1;
    }

    private static bool TryFindVocab(ReadOnlySpan<byte> json, out int start, out int end, out UnigramVocabulary? vocab)
    {
        start = end = 0;
        vocab = null;
        var reader = new Utf8JsonReader(json, _readerOptions);
        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
        {
            return false;
        }

        while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
        {
            bool isModel = reader.ValueTextEquals("model"u8);
            reader.Read();
            if (!isModel)
            {
                reader.Skip();
                continue;
            }

            if (reader.TokenType != JsonTokenType.StartObject)
            {
                return false;
            }

            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                bool isVocab = reader.ValueTextEquals("vocab"u8);
                reader.Read();
                if (!isVocab)
                {
                    reader.Skip();
                    continue;
                }

                if (reader.TokenType != JsonTokenType.StartArray)
                {
                    return false;
                }

                start = checked((int)reader.TokenStartIndex);
                vocab = ScanEntries(json, start, out end) ?? ReadEntries(ref reader, json.Length - start, out end);
                return vocab is not null;
            }

            return false;
        }

        return false;
    }

    private static UnigramVocabulary NewVocabulary(int arrayBytes) =>
        // About 68 bytes per entry and 15 bytes of piece text in the pinned assets; only sizes the first buffers.
        new(Math.Max(arrayBytes / 64, 16), Math.Max(arrayBytes / 4, 16));

    /// <summary>Byte scanner for the vocab array starting at <c>json[start] == '['</c>; null if it meets anything unusual.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static UnigramVocabulary? ScanEntries(ReadOnlySpan<byte> json, int start, out int end)
    {
        end = 0;
        int p = SkipWhitespace(json, start + 1);
        UnigramVocabulary vocab = NewVocabulary(json.Length - start);
        if (p < json.Length && json[p] == (byte)']')
        {
            end = p + 1;
            return vocab;
        }

        while (true)
        {
            if (p >= json.Length || json[p] != (byte)'[')
            {
                return null;
            }

            p = SkipWhitespace(json, p + 1);
            if (p >= json.Length || json[p] != (byte)'"')
            {
                return null;
            }

            int quote = p;
            bool escaped = false;
            p++;
            while (true)
            {
                if (p >= json.Length)
                {
                    return null;
                }

                byte b = json[p];
                if (b == (byte)'"')
                {
                    break;
                }

                if (b == (byte)'\\')
                {
                    escaped = true;
                    p += 2;
                    continue;
                }

                if (b < 0x20)
                {
                    return null;
                }

                p++;
            }

            ReadOnlySpan<byte> piece = json[(quote + 1)..p];
            string? decoded = null;
            if (escaped)
            {
                // The string token alone is a JSON document; decode it exactly as the reader does.
                var stringReader = new Utf8JsonReader(json[quote..(p + 1)]);
                if (!stringReader.Read() || stringReader.TokenType != JsonTokenType.String)
                {
                    return null;
                }

                decoded = stringReader.GetString();
            }
            else if (!Utf8.IsValid(piece))
            {
                return null;
            }

            p = SkipWhitespace(json, p + 1);
            if (p >= json.Length || json[p] != (byte)',')
            {
                return null;
            }

            p = SkipWhitespace(json, p + 1);
            int numberStart = p;
            p = ScanNumber(json, p);
            if (p < 0)
            {
                return null;
            }

            // Scores are parsed as the reference's serde_json does, not correctly rounded.
            double score = SerdeJsonNumber.ParseDouble(json[numberStart..p]);
            p = SkipWhitespace(json, p);
            if (p >= json.Length || json[p] != (byte)']')
            {
                return null;
            }

            if (decoded is not null)
            {
                vocab.Add(decoded, score);
            }
            else
            {
                vocab.Add(piece, score);
            }

            p = SkipWhitespace(json, p + 1);
            if (p >= json.Length)
            {
                return null;
            }

            if (json[p] == (byte)']')
            {
                end = p + 1;
                return vocab;
            }

            if (json[p] != (byte)',')
            {
                return null;
            }

            p = SkipWhitespace(json, p + 1);
        }
    }

    /// <summary>End of a JSON number (RFC 8259 grammar) starting at <paramref name="p"/>, or -1.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ScanNumber(ReadOnlySpan<byte> json, int p)
    {
        if (p < json.Length && json[p] == (byte)'-')
        {
            p++;
        }

        if (p >= json.Length)
        {
            return -1;
        }

        if (json[p] == (byte)'0')
        {
            p++;
        }
        else if (IsNonZeroDigit(json[p]))
        {
            p = SkipDigits(json, p + 1);
        }
        else
        {
            return -1;
        }

        if (p < json.Length && json[p] == (byte)'.')
        {
            p++;
            if (p >= json.Length || !IsDigit(json[p]))
            {
                return -1;
            }

            p = SkipDigits(json, p + 1);
        }

        if (p < json.Length && (json[p] | 0x20) == (byte)'e')
        {
            p++;
            if (p < json.Length && json[p] is (byte)'+' or (byte)'-')
            {
                p++;
            }

            if (p >= json.Length || !IsDigit(json[p]))
            {
                return -1;
            }

            p = SkipDigits(json, p + 1);
        }

        return p;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SkipDigits(ReadOnlySpan<byte> json, int p)
    {
        while (p < json.Length && IsDigit(json[p]))
        {
            p++;
        }

        return p;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int SkipWhitespace(ReadOnlySpan<byte> json, int p)
    {
        while (p < json.Length && json[p] is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t')
        {
            p++;
        }

        return p;
    }

    private static bool IsDigit(byte b) => (uint)(b - '0') <= 9;

    private static bool IsNonZeroDigit(byte b) => (uint)(b - '1') <= 8;

    /// <summary>The entries of a vocab array through <see cref="Utf8JsonReader"/>, the reader on its start; null if any entry is unusual.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static UnigramVocabulary? ReadEntries(ref Utf8JsonReader reader, int remainingBytes, out int end)
    {
        end = 0;
        UnigramVocabulary vocab = NewVocabulary(remainingBytes);
        while (true)
        {
            reader.Read();
            if (reader.TokenType == JsonTokenType.EndArray)
            {
                end = checked((int)reader.BytesConsumed);
                return vocab;
            }

            if (reader.TokenType != JsonTokenType.StartArray || !reader.Read() || reader.TokenType != JsonTokenType.String)
            {
                return null;
            }

            string? decoded = null;
            ReadOnlySpan<byte> piece = default;
            if (reader.ValueIsEscaped)
            {
                decoded = reader.GetString();
            }
            else
            {
                piece = reader.ValueSpan;
                if (!Utf8.IsValid(piece))
                {
                    return null;
                }
            }

            if (!reader.Read() || reader.TokenType != JsonTokenType.Number)
            {
                return null;
            }

            double score = SerdeJsonNumber.ParseDouble(reader.ValueSpan);
            if (!reader.Read() || reader.TokenType != JsonTokenType.EndArray)
            {
                return null;
            }

            if (decoded is not null)
            {
                vocab.Add(decoded, score);
            }
            else
            {
                vocab.Add(piece, score);
            }
        }
    }
}
