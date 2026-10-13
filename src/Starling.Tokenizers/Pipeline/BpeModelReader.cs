// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Unicode;

namespace Starling.Tokenizers.Pipeline;

/// <summary>A BPE model's vocabulary and merge map, read from <c>tokenizer.json</c>.</summary>
internal sealed record BpeModelData(BpeVocabulary Vocabulary, BpeMergeTable Merges);

/// <summary>
/// Reads the BPE <c>model.vocab</c> object and <c>model.merges</c> array (248,044 and 247,587
/// entries, nearly all of the Qwen asset's 20 MB). The fast path scans them from the UTF-8 bytes and
/// hands the rest of the document, with the two replaced by <c>{}</c> and <c>[]</c>, to the document
/// model; the document path reads the same values from <see cref="JsonElement"/>s.
/// </summary>
/// <remarks>
/// The fast path accepts only input the document path reads to the same values: <c>vocab</c>
/// directly followed by <c>merges</c> in the first root <c>model</c> object (the order tokenizers
/// writes), keys that are JSON strings (escaped ones decoded by <see cref="Utf8JsonReader"/>),
/// ids that are plain non-negative integers within <see cref="int"/>, and merges all in one form.
/// Anything else, including any error (an out-of-vocabulary merge, a duplicate key), makes it
/// decline, and the unmodified file goes through the document path, which reports the error.
/// The caller checks on the rest document that <c>model</c>, <c>vocab</c> and <c>merges</c> each
/// occur once (<see cref="HasSingleVocabAndMerges"/>).
/// </remarks>
internal static class BpeModelReader
{
    private static readonly JsonReaderOptions _readerOptions = new() { MaxDepth = 64 };
    private static readonly UTF8Encoding _strictUtf8 = new(false, true);

    public static bool TryRead(ReadOnlySpan<byte> json, out BpeModelData? data, out byte[]? documentWithout)
    {
        data = null;
        documentWithout = null;
        try
        {
            if (!TryLocateVocab(json, out int vocabStart))
            {
                return false;
            }

            BpeVocabulary? vocab = ScanVocab(json, vocabStart, out int vocabEnd);
            if (vocab is null)
            {
                return false;
            }

            // "merges" must be the next property of the same object.
            int p = SkipWhitespace(json, vocabEnd);
            if (p >= json.Length || json[p] != (byte)',')
            {
                return false;
            }

            p = SkipWhitespace(json, p + 1);
            if (!json[p..].StartsWith("\"merges\""u8))
            {
                return false;
            }

            p = SkipWhitespace(json, p + 8);
            if (p >= json.Length || json[p] != (byte)':')
            {
                return false;
            }

            int mergesStart = SkipWhitespace(json, p + 1);
            vocab.Seal();
            BpeMergeTable? merges = ScanMerges(json, mergesStart, vocab, out int mergesEnd);
            if (merges is null)
            {
                return false;
            }

            var rest = new byte[json.Length - (vocabEnd - vocabStart) - (mergesEnd - mergesStart) + 4];
            int w = 0;
            json[..vocabStart].CopyTo(rest);
            w += vocabStart;
            rest[w++] = (byte)'{';
            rest[w++] = (byte)'}';
            json[vocabEnd..mergesStart].CopyTo(rest.AsSpan(w));
            w += mergesStart - vocabEnd;
            rest[w++] = (byte)'[';
            rest[w++] = (byte)']';
            json[mergesEnd..].CopyTo(rest.AsSpan(w));
            documentWithout = rest;
            data = new BpeModelData(vocab, merges);
            return true;
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or TokenizerFormatException
            or UnsupportedTokenizerComponentException or DecoderFallbackException or ArgumentException)
        {
            data = null;
            documentWithout = null;
            return false;
        }
    }

    /// <summary>True if the root has one <c>model</c> key, an object with one <c>vocab</c> and one <c>merges</c> key.</summary>
    public static bool HasSingleVocabAndMerges(JsonElement root)
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

        int vocab = 0;
        int merges = 0;
        foreach (JsonProperty p in m.EnumerateObject())
        {
            vocab += p.NameEquals("vocab"u8) ? 1 : 0;
            merges += p.NameEquals("merges"u8) ? 1 : 0;
        }

        return vocab == 1 && merges == 1;
    }

    /// <summary>Document path: <c>vocab</c> as a string-to-u32 map, <c>merges</c> in pair or legacy form.</summary>
    public static BpeModelData ReadDocument(JsonElement vocabElement, JsonElement mergesElement, string where)
    {
        if (vocabElement.ValueKind != JsonValueKind.Object)
        {
            throw new TokenizerFormatException($"{where}: vocab must be an object.");
        }

        var vocab = new BpeVocabulary();
        foreach (JsonProperty p in vocabElement.EnumerateObject())
        {
            if (p.Value.ValueKind != JsonValueKind.Number || !p.Value.TryGetUInt32(out uint id))
            {
                throw new TokenizerFormatException($"{where}: vocab ids must be non-negative 32-bit integers.");
            }

            if (id > int.MaxValue)
            {
                throw new UnsupportedTokenizerComponentException($"{where}.vocab", $"id {id} is larger than {int.MaxValue}.");
            }

            vocab.Add(p.Name, (int)id);
        }

        vocab.Seal();
        if (mergesElement.ValueKind != JsonValueKind.Array)
        {
            throw new TokenizerFormatException($"{where}: merges must be an array.");
        }

        int length = mergesElement.GetArrayLength();
        var table = new BpeMergeTable(length);
        bool tuples = length == 0 || mergesElement[0].ValueKind == JsonValueKind.Array;
        int rank = 0;
        foreach (JsonElement item in mergesElement.EnumerateArray())
        {
            if (tuples)
            {
                if (item.ValueKind != JsonValueKind.Array || item.GetArrayLength() != 2
                    || item[0].ValueKind != JsonValueKind.String || item[1].ValueKind != JsonValueKind.String)
                {
                    throw new TokenizerFormatException($"{where}: merges must all be [string, string] pairs or all \"a b\" strings.");
                }

                BpeModel.AddMerge(vocab, table, Encoding.UTF8.GetBytes(item[0].GetString()!), Encoding.UTF8.GetBytes(item[1].GetString()!), rank++);
            }
            else
            {
                if (item.ValueKind != JsonValueKind.String)
                {
                    throw new TokenizerFormatException($"{where}: merges must all be [string, string] pairs or all \"a b\" strings.");
                }

                string line = item.GetString()!;
                if (line.StartsWith("#version", StringComparison.Ordinal))
                {
                    continue;
                }

                string[] parts = line.Split(' ');
                if (parts.Length != 2)
                {
                    throw new TokenizerFormatException($"{where}: merges line {rank + 1} is not two tokens separated by one space.");
                }

                BpeModel.AddMerge(vocab, table, Encoding.UTF8.GetBytes(parts[0]), Encoding.UTF8.GetBytes(parts[1]), rank++);
            }
        }

        return new BpeModelData(vocab, table);
    }

    private static bool TryLocateVocab(ReadOnlySpan<byte> json, out int start)
    {
        start = 0;
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
                if (reader.ValueTextEquals("merges"u8))
                {
                    return false;
                }

                reader.Read();
                if (!isVocab)
                {
                    reader.Skip();
                    continue;
                }

                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    return false;
                }

                start = checked((int)reader.TokenStartIndex);
                return true;
            }

            return false;
        }

        return false;
    }

    /// <summary>The vocab object at <c>json[start] == '{'</c>, or null if anything is unusual.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static BpeVocabulary? ScanVocab(ReadOnlySpan<byte> json, int start, out int end)
    {
        end = 0;
        // About 40 bytes per entry and 7 bytes of token text in the Qwen assets; only sizes the buffers.
        int expected = Math.Max((json.Length - start) / 80, 16);
        var vocab = new BpeVocabulary(expected, expected * 8);
        int p = SkipWhitespace(json, start + 1);
        if (p < json.Length && json[p] == (byte)'}')
        {
            end = p + 1;
            return vocab;
        }

        while (true)
        {
            if (!ScanString(json, ref p, out ReadOnlySpan<byte> key, out byte[]? decoded))
            {
                return null;
            }

            p = SkipWhitespace(json, p);
            if (p >= json.Length || json[p] != (byte)':')
            {
                return null;
            }

            p = SkipWhitespace(json, p + 1);
            if (!ScanId(json, ref p, out int id))
            {
                return null;
            }

            vocab.Add(decoded ?? key, id);
            p = SkipWhitespace(json, p);
            if (p >= json.Length)
            {
                return null;
            }

            if (json[p] == (byte)'}')
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

    /// <summary>The merges array at <c>json[start] == '['</c>, resolved against the sealed vocabulary, or null.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static BpeMergeTable? ScanMerges(ReadOnlySpan<byte> json, int start, BpeVocabulary vocab, out int end)
    {
        end = 0;
        if (start >= json.Length || json[start] != (byte)'[')
        {
            return null;
        }

        var table = new BpeMergeTable(Math.Max((json.Length - start) / 16, 16));
        int p = SkipWhitespace(json, start + 1);
        if (p < json.Length && json[p] == (byte)']')
        {
            end = p + 1;
            return table;
        }

        bool tuples = p < json.Length && json[p] == (byte)'[';
        int rank = 0;
        while (true)
        {
            if (tuples)
            {
                if (p >= json.Length || json[p] != (byte)'[')
                {
                    return null;
                }

                p = SkipWhitespace(json, p + 1);
                if (!ScanString(json, ref p, out ReadOnlySpan<byte> left, out byte[]? leftDecoded))
                {
                    return null;
                }

                p = SkipWhitespace(json, p);
                if (p >= json.Length || json[p] != (byte)',')
                {
                    return null;
                }

                p = SkipWhitespace(json, p + 1);
                if (!ScanString(json, ref p, out ReadOnlySpan<byte> right, out byte[]? rightDecoded))
                {
                    return null;
                }

                p = SkipWhitespace(json, p);
                if (p >= json.Length || json[p] != (byte)']')
                {
                    return null;
                }

                p++;
                BpeModel.AddMerge(vocab, table, leftDecoded ?? left, rightDecoded ?? right, rank++);
            }
            else
            {
                if (!ScanString(json, ref p, out ReadOnlySpan<byte> line, out byte[]? lineDecoded))
                {
                    return null;
                }

                ReadOnlySpan<byte> text = lineDecoded ?? line;
                if (!text.StartsWith("#version"u8))
                {
                    int space = text.IndexOf((byte)' ');
                    if (space < 0 || text[(space + 1)..].Contains((byte)' '))
                    {
                        return null;
                    }

                    BpeModel.AddMerge(vocab, table, text[..space], text[(space + 1)..], rank++);
                }
            }

            p = SkipWhitespace(json, p);
            if (p >= json.Length)
            {
                return null;
            }

            if (json[p] == (byte)']')
            {
                end = p + 1;
                return table;
            }

            if (json[p] != (byte)',')
            {
                return null;
            }

            p = SkipWhitespace(json, p + 1);
        }
    }

    /// <summary>
    /// A JSON string at <paramref name="p"/>: <paramref name="raw"/> is its UTF-8 content when it has
    /// no escapes (validated), else <paramref name="decoded"/> holds the decoded UTF-8. Advances past
    /// the closing quote.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ScanString(ReadOnlySpan<byte> json, ref int p, out ReadOnlySpan<byte> raw, out byte[]? decoded)
    {
        raw = default;
        decoded = null;
        if (p >= json.Length || json[p] != (byte)'"')
        {
            return false;
        }

        int quote = p;
        bool escaped = false;
        p++;
        while (true)
        {
            if (p >= json.Length)
            {
                return false;
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
                return false;
            }

            p++;
        }

        raw = json[(quote + 1)..p];
        p++;
        if (escaped)
        {
            // The string token alone is a JSON document; decode it exactly as the reader does.
            var stringReader = new Utf8JsonReader(json[quote..p]);
            if (!stringReader.Read() || stringReader.TokenType != JsonTokenType.String)
            {
                return false;
            }

            decoded = _strictUtf8.GetBytes(stringReader.GetString()!);
            return true;
        }

        return Utf8.IsValid(raw);
    }

    /// <summary>A non-negative JSON integer (no fraction or exponent) within <see cref="int"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool ScanId(ReadOnlySpan<byte> json, ref int p, out int id)
    {
        id = 0;
        if (p >= json.Length || !IsDigit(json[p]))
        {
            return false;
        }

        if (json[p] == (byte)'0')
        {
            p++;
        }
        else
        {
            long value = 0;
            while (p < json.Length && IsDigit(json[p]))
            {
                value = (value * 10) + (json[p] - '0');
                if (value > int.MaxValue)
                {
                    return false;
                }

                p++;
            }

            id = (int)value;
        }

        // A fraction or exponent makes it a float, which the reference rejects for an id.
        return p >= json.Length || (json[p] != (byte)'.' && (json[p] | 0x20) != (byte)'e');
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
}
