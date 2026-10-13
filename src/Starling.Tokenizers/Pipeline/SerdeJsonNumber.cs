// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Ported from serde_json 1.0.151 src/de.rs (parse_integer, parse_number, parse_decimal,
// parse_exponent, parse_long_integer, parse_decimal_overflow, parse_exponent_overflow, and the
// non-float_roundtrip f64_from_parts), Copyright David Tolnay and serde_json contributors,
// MIT OR Apache-2.0 (used here under MIT; see THIRD-PARTY-NOTICES.md). serde_json 1.0.151 is
// the version locked by huggingface/tokenizers v0.23.2's Python binding, which does not enable
// the float_roundtrip feature.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Starling.Tokenizers.Pipeline;

/// <summary>
/// Parses JSON numbers to <see cref="double"/> exactly as the reference does. serde_json's
/// default float parser multiplies or divides the decimal significand by a power of ten, which
/// can differ from a correctly rounded parse by one unit in the last place. Unigram scores are
/// summed and compared, so the reference's rounding is reproduced instead of System.Text.Json's.
/// </summary>
internal static class SerdeJsonNumber
{
    private static readonly double[] _pow10 = BuildPow10();

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double ParseDouble(ReadOnlySpan<byte> raw)
    {
        int pos = 0;
        bool positive = true;
        if (pos < raw.Length && raw[pos] == (byte)'-')
        {
            positive = false;
            pos++;
        }

        if (pos >= raw.Length)
        {
            throw Invalid(raw);
        }

        byte first = raw[pos++];
        ulong significand;
        if (first == (byte)'0')
        {
            if (pos < raw.Length && IsDigit(raw[pos]))
            {
                throw Invalid(raw);
            }

            significand = 0;
        }
        else if (first is >= (byte)'1' and <= (byte)'9')
        {
            significand = (ulong)(first - '0');
            while (pos < raw.Length && IsDigit(raw[pos]))
            {
                ulong digit = (ulong)(raw[pos] - '0');
                if (significand > (ulong.MaxValue - digit) / 10)
                {
                    return ParseLongInteger(raw, ref pos, positive, significand);
                }

                pos++;
                significand = (significand * 10) + digit;
            }
        }
        else
        {
            throw Invalid(raw);
        }

        return ParseNumber(raw, ref pos, positive, significand);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double ParseNumber(ReadOnlySpan<byte> raw, ref int pos, bool positive, ulong significand)
    {
        if (pos < raw.Length && raw[pos] == (byte)'.')
        {
            return ParseDecimal(raw, ref pos, positive, significand, 0);
        }

        if (pos < raw.Length && (raw[pos] | 0x20) == (byte)'e')
        {
            return ParseExponent(raw, ref pos, positive, significand, 0);
        }

        if (positive)
        {
            return significand;
        }

        long neg = unchecked(-(long)significand);
        return neg >= 0 ? -(double)significand : neg;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double ParseDecimal(ReadOnlySpan<byte> raw, ref int pos, bool positive, ulong significand, int exponentBeforeDecimalPoint)
    {
        pos++;
        int exponentAfterDecimalPoint = 0;
        while (pos < raw.Length && IsDigit(raw[pos]))
        {
            ulong digit = (ulong)(raw[pos] - '0');
            if (significand > (ulong.MaxValue - digit) / 10)
            {
                int exponent = exponentBeforeDecimalPoint + exponentAfterDecimalPoint;
                return ParseDecimalOverflow(raw, ref pos, positive, significand, exponent);
            }

            pos++;
            significand = (significand * 10) + digit;
            exponentAfterDecimalPoint--;
        }

        if (exponentAfterDecimalPoint == 0)
        {
            throw Invalid(raw);
        }

        int exp = exponentBeforeDecimalPoint + exponentAfterDecimalPoint;
        if (pos < raw.Length && (raw[pos] | 0x20) == (byte)'e')
        {
            return ParseExponent(raw, ref pos, positive, significand, exp);
        }

        return FromParts(positive, significand, exp);
    }

    private static double ParseExponent(ReadOnlySpan<byte> raw, ref int pos, bool positive, ulong significand, int startingExp)
    {
        pos++;
        bool positiveExp = true;
        if (pos < raw.Length && raw[pos] == (byte)'+')
        {
            pos++;
        }
        else if (pos < raw.Length && raw[pos] == (byte)'-')
        {
            positiveExp = false;
            pos++;
        }

        if (pos >= raw.Length || !IsDigit(raw[pos]))
        {
            throw Invalid(raw);
        }

        int exp = raw[pos++] - '0';
        while (pos < raw.Length && IsDigit(raw[pos]))
        {
            int digit = raw[pos++] - '0';
            if (exp > (int.MaxValue - digit) / 10)
            {
                if (significand != 0 && positiveExp)
                {
                    throw new TokenizerFormatException("JSON number out of range.");
                }

                while (pos < raw.Length && IsDigit(raw[pos]))
                {
                    pos++;
                }

                return positive ? 0.0 : -0.0;
            }

            exp = (exp * 10) + digit;
        }

        int finalExp = positiveExp ? SaturatingAdd(startingExp, exp) : SaturatingAdd(startingExp, -exp);
        return FromParts(positive, significand, finalExp);
    }

    private static double ParseLongInteger(ReadOnlySpan<byte> raw, ref int pos, bool positive, ulong significand)
    {
        int exponent = 0;
        while (pos < raw.Length && IsDigit(raw[pos]))
        {
            pos++;
            exponent++;
        }

        return pos < raw.Length && raw[pos] == (byte)'.'
            ? ParseDecimal(raw, ref pos, positive, significand, exponent)
            : pos < raw.Length && (raw[pos] | 0x20) == (byte)'e'
                ? ParseExponent(raw, ref pos, positive, significand, exponent)
                : FromParts(positive, significand, exponent);
    }

    private static double ParseDecimalOverflow(ReadOnlySpan<byte> raw, ref int pos, bool positive, ulong significand, int exponent)
    {
        while (pos < raw.Length && IsDigit(raw[pos]))
        {
            pos++;
        }

        if (pos < raw.Length && (raw[pos] | 0x20) == (byte)'e')
        {
            return ParseExponent(raw, ref pos, positive, significand, exponent);
        }

        return FromParts(positive, significand, exponent);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double FromParts(bool positive, ulong significand, int exponent)
    {
        double f = significand;
        while (true)
        {
            int index = exponent == int.MinValue ? int.MinValue : Math.Abs(exponent);
            if (index >= 0 && index < _pow10.Length)
            {
                if (exponent >= 0)
                {
                    f *= _pow10[index];
                    if (double.IsInfinity(f))
                    {
                        throw new TokenizerFormatException("JSON number out of range.");
                    }
                }
                else
                {
                    f /= _pow10[index];
                }

                break;
            }

            if (f == 0.0)
            {
                break;
            }

            if (exponent >= 0)
            {
                throw new TokenizerFormatException("JSON number out of range.");
            }

            f /= 1e308;
            exponent += 308;
        }

        return positive ? f : -f;
    }

    private static int SaturatingAdd(int a, int b)
    {
        long r = (long)a + b;
        return r > int.MaxValue ? int.MaxValue : r < int.MinValue ? int.MinValue : (int)r;
    }

    private static bool IsDigit(byte b) => b is >= (byte)'0' and <= (byte)'9';

    private static TokenizerFormatException Invalid(ReadOnlySpan<byte> raw) =>
        new($"Invalid JSON number '{System.Text.Encoding.UTF8.GetString(raw)}'.");

    private static double[] BuildPow10()
    {
        var table = new double[309];
        for (int i = 0; i < table.Length; i++)
        {
            table[i] = double.Parse("1e" + i.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
        }

        return table;
    }
}
