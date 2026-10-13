// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
//
// Adopted from dotnet/machinelearning at 5a44dfdbf20d1b64211da313dfbaa9ffbf63ea5d,
// src/Microsoft.ML.Tokenizers/Utils/DoubleArrayTrie.cs (MIT, see THIRD-PARTY-NOTICES.md).
// Modified by StarlingHq (see PROVENANCE.md): namespace changed; DoubleArrayUnit gains a
// constructor from its raw 32-bit value so precompiled charsmaps load without unsafe code; the
// DAWG pass (DawgBuilder, BitVector, build_from_dawg) is removed and the builder places sorted,
// unique keys directly with darts-clone's build_from_keyset/arrange_from_keyset, ported from
// s-yata/darts-clone 87b71afd6cf784953e3c08f24c64203397f3b724 include/darts.h (BSD 2-Clause,
// notice below); hot builder methods are marked AggressiveOptimization (load-time JIT).

// The implementation of the DoubleArrayBuilder class is based on the following C# port of the C++ implementation of the Double-Array Trie (DART) data structure.
// The original C++ implementation is available at https://github.com/s-yata/darts-clone/blob/master/include/darts.h and used under BSD 2-clause license.
//
// darts-clone notice (BSD 2-clause), reproduced by StarlingHq:
// Copyright (c) 2008-2014, Susumu Yata All rights reserved.
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice, this list of conditions
// and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials provided
// with the distribution.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

using System.Runtime.CompilerServices;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Starlings.Tokenizers.ThirdParty.MLNet
{
    internal class AutoPool<T>
    {
        private T[] _buf = Array.Empty<T>();
        private int _size;
        private int _capacity;

        public AutoPool() { }

        public ref T this[int id]
        {
            get => ref _buf[id];
        }

        public T[] Buffer => _buf;

        public bool Empty => _size == 0;

        public int Size => _size;

        public void Clear()
        {
            _buf = Array.Empty<T>();
            _size = 0;
            _capacity = 0;
        }

        public void ResizeBuf(int size)
        {
            if (size <= _capacity)
            {
                return;
            }

            int capacity;
            if (size >= _capacity * 2)
            {
                capacity = size;
            }
            else
            {
                capacity = 1;
                while (capacity < size)
                {
                    capacity <<= 1;
                }
            }

            T[] buf = new T[capacity];

            if (_buf is not null)
            {
                Array.Copy(_buf, buf, _size);
            }

            _buf = buf;
            _capacity = capacity;
        }

        public void Append(T value)
        {
            if (_size == _capacity)
            {
                ResizeBuf(_size + 1);
            }

            _buf[_size++] = value!;
        }

        public void Append()
        {
            if (_size == _capacity)
            {
                ResizeBuf(_size + 1);
            }
            _buf[_size++] = default!;
        }

        public void PushBack(T value) => Append(value);

        public void PopBack()
        {
            if (Empty)
            {
                return;
            }

            _buf[--_size] = default!;
        }

        public void Resize(int size)
        {
            while (_size > size)
            {
                _buf[--_size] = default!;
            }

            if (size > _capacity)
            {
                ResizeBuf(size);
            }

            while (_size < size)
            {
                _buf[_size++] = default!;
            }
        }

        public void Resize(int size, T value)
        {
            while (_size > size)
            {
                _buf[--_size] = default!;
            }

            if (size > _capacity)
            {
                ResizeBuf(size);
            }

            while (_size < size)
            {
                _buf[_size++] = value;
            }
        }

        public void Reserve(int size)
        {
            if (size > _capacity)
            {
                ResizeBuf(size);
            }
        }
    }

    internal struct DoubleArrayUnit
    {
        private uint _unit;
        public DoubleArrayUnit() { }
        public DoubleArrayUnit(uint unit) => _unit = unit;

        // returns whether a leaf unit is immediately derived from the unit (true) or not (false).
        public bool HasLeaf
        {
            get => ((_unit >> 8) & 1) == 1;
            set
            {
                if (value)
                {
                    _unit |= 1U << 8;
                }
                else
                {
                    _unit &= ~(1U << 8);
                }
            }
        }

        // value() returns the value stored in the unit, and thus value() is
        // available when and only when the unit is a leaf unit.
        public uint Value
        {
            get => _unit & ((1U << 31) - 1);
            set => _unit = value | (1U << 31);
        }

        // returns the label associated with the unit. Note that a leaf unit always returns an invalid label.
        // For this feature, leaf unit's label returns an id that has the MSB of 1.
        public uint Label
        {
            get => _unit & ((1U << 31) | 0xFF);
            set
            {
                _unit = (_unit & ~0xFFU) | value;
            }
        }

        // offset() returns the offset from the unit to its derived units.
        public uint Offset
        {
            get => (_unit >> 10) << (int)((_unit & (1U << 9)) >> 6);
            set
            {
                if (value >= 1U << 29)
                {
                    throw new InvalidOperationException("failed to modify unit: too large offset");
                }

                _unit &= (1U << 31) | (1U << 8) | 0xFF;

                if (value < 1U << 21)
                {
                    _unit |= value << 10;
                }
                else
                {
                    _unit |= (value << 2) | (1U << 9);
                }
            }
        }

    }

    //
    // Extra unit of double-array builder.
    //

    internal struct DoubleArrayBuilderExtraUnit
    {

        private uint _prev;
        private uint _next;
        private bool _isFixed;
        private bool _isUsed;

        public DoubleArrayBuilderExtraUnit() { }

        public uint Prev
        {
            get => _prev;
            set => _prev = value;
        }

        public uint Next
        {
            get => _next;
            set => _next = value;
        }

        public bool IsFixed
        {
            get => _isFixed;
            set => _isFixed = value;
        }

        public bool IsUsed
        {
            get => _isUsed;
            set => _isUsed = value;
        }
    }

    internal class DoubleArrayBuilder
    {
        private const int BlockSize = 256;
        private const int NumExtraBlock = 16;
        private const int NumExtras = BlockSize * NumExtraBlock;
        private const int UpperMask = 0xFF << 21;
        private const int LowerMask = 0xFF;

        private readonly AutoPool<DoubleArrayUnit> _units = new();
        private readonly DoubleArrayBuilderExtraUnit[] _extras = new DoubleArrayBuilderExtraUnit[NumExtras];
        private readonly AutoPool<byte> _labels = new();
        private uint _extrasHead;

        private int NumBlocks() => _units.Size / BlockSize;

        public DoubleArrayUnit[] Units => _units.Buffer;
        public int UnitsSize => _units.Size;

        private ref DoubleArrayBuilderExtraUnit this[uint id]
        {
            get => ref _extras[id % NumExtras];
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void FixBlock(uint blockId)
        {
            uint begin = blockId * BlockSize;
            uint end = begin + BlockSize;

            uint unusedOffset = 0;
            for (uint offset = begin; offset != end; ++offset)
            {
                if (!this[offset].IsUsed)
                {
                    unusedOffset = offset;
                    break;
                }
            }

            for (uint id = begin; id != end; ++id)
            {
                if (!this[id].IsFixed)
                {
                    ReserveId(id);
                    _units[(int)id].Label = (byte)(id ^ unusedOffset);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void ExpandUnits()
        {
            uint srcNumUnits = (uint)_units.Size;
            uint srcNumBlocks = (uint)NumBlocks();

            uint destNumUnits = srcNumUnits + BlockSize;
            uint destNumBlocks = srcNumBlocks + 1;

            if (destNumBlocks > NumExtraBlock)
            {
                FixBlock(srcNumBlocks - NumExtraBlock);
            }

            _units.Resize((int)destNumUnits);

            if (destNumBlocks > NumExtraBlock)
            {
                for (uint id = srcNumUnits; id < destNumUnits; ++id)
                {
                    this[id].IsUsed = false;
                    this[id].IsFixed = false;
                }
            }

            for (uint i = srcNumUnits + 1; i < destNumUnits; ++i)
            {
                this[i - 1].Next = i;
                this[i].Prev = i - 1;
            }

            this[srcNumUnits].Prev = destNumUnits - 1;
            this[destNumUnits - 1].Next = srcNumUnits;
            this[srcNumUnits].Prev = this[_extrasHead].Prev;
            this[destNumUnits - 1].Next = _extrasHead;
            this[this[_extrasHead].Prev].Next = srcNumUnits;
            this[_extrasHead].Prev = destNumUnits - 1;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void ReserveId(uint id)
        {
            if (id >= _units.Size)
            {
                ExpandUnits();
            }

            if (id == _extrasHead)
            {
                _extrasHead = this[id].Next;
                if (_extrasHead == id)
                {
                    _extrasHead = (uint)_units.Size;
                }
            }

            this[this[id].Prev].Next = this[id].Next;
            this[this[id].Next].Prev = this[id].Prev;
            this[id].IsFixed = true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal bool IsValidOffset(uint id, uint offset)
        {
            if (this[offset].IsUsed)
            {
                return false;
            }

            uint relOffset = id ^ offset;
            if ((relOffset & LowerMask) != 0 && (relOffset & UpperMask) != 0)
            {
                return false;
            }

            for (int i = 1; i < _labels.Size; ++i)
            {
                if (this[offset ^ _labels[i]].IsFixed)
                {
                    return false;
                }
            }

            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal uint FindValidOffset(uint id)
        {
            if (_extrasHead >= _units.Size)
            {
                return (uint)_units.Size | (id & LowerMask);
            }

            uint unfixedId = _extrasHead;
            do
            {
                uint offset = unfixedId ^ _labels[0];
                if (IsValidOffset(id, offset))
                {
                    return offset;
                }

                unfixedId = this[unfixedId].Next;
            } while (unfixedId != _extrasHead);

            return (uint)_units.Size | (id & LowerMask);
        }

        // darts-clone DoubleArrayBuilder::arrange_from_keyset.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal uint ArrangeFromKeyset(in DoubleArrayKeyset keyset, int begin, int end, int depth, uint dicId)
        {
            _labels.Resize(0);

            int value = -1;
            for (int i = begin; i < end; ++i)
            {
                byte label = keyset.Label(i, depth);
                if (label == 0)
                {
                    if (depth < keyset.Length(i))
                    {
                        throw new InvalidOperationException("failed to build double-array: invalid null character");
                    }
                    else if (keyset.Value(i) < 0)
                    {
                        throw new InvalidOperationException("failed to build double-array: negative value");
                    }

                    if (value == -1)
                    {
                        value = keyset.Value(i);
                    }
                }

                if (_labels.Empty)
                {
                    _labels.Append(label);
                }
                else if (label != _labels[_labels.Size - 1])
                {
                    if (label < _labels[_labels.Size - 1])
                    {
                        throw new InvalidOperationException("failed to build double-array: wrong key order");
                    }

                    _labels.Append(label);
                }
            }

            uint offset = FindValidOffset(dicId);
            _units[(int)dicId].Offset = dicId ^ offset;

            for (int i = 0; i < _labels.Size; ++i)
            {
                uint dicChildId = offset ^ _labels[i];
                ReserveId(dicChildId);
                if (_labels[i] == 0)
                {
                    _units[(int)dicId].HasLeaf = true;
                    _units[(int)dicChildId].Value = (uint)value;
                }
                else
                {
                    _units[(int)dicChildId].Label = _labels[i];
                }
            }

            this[offset].IsUsed = true;

            return offset;
        }

        // darts-clone DoubleArrayBuilder::build_from_keyset (recursive part).
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal void BuildFromKeyset(in DoubleArrayKeyset keyset, int begin, int end, int depth, uint dicId)
        {
            uint offset = ArrangeFromKeyset(keyset, begin, end, depth, dicId);

            while (begin < end)
            {
                if (keyset.Label(begin, depth) != 0)
                {
                    break;
                }

                ++begin;
            }

            if (begin == end)
            {
                return;
            }

            int lastBegin = begin;
            byte lastLabel = keyset.Label(begin, depth);
            while (++begin < end)
            {
                byte label = keyset.Label(begin, depth);
                if (label != lastLabel)
                {
                    BuildFromKeyset(keyset, lastBegin, begin, depth + 1, offset ^ lastLabel);
                    lastBegin = begin;
                    lastLabel = keyset.Label(begin, depth);
                }
            }

            BuildFromKeyset(keyset, lastBegin, end, depth + 1, offset ^ lastLabel);
        }

        internal void FixAllBlocks()
        {
            uint begin = 0;
            if (NumBlocks() > NumExtraBlock)
            {
                begin = (uint)NumBlocks() - NumExtraBlock;
            }

            uint end = (uint)NumBlocks();

            for (uint blockId = begin; blockId != end; ++blockId)
            {
                FixBlock(blockId);
            }
        }

        // darts-clone DoubleArrayBuilder::build_from_keyset: the keys are placed directly, without
        // the DAWG (suffix-sharing) pass. Lookups give the same results; the array is larger.
        public void Build(in DoubleArrayKeyset keyset)
        {
            int numUnits = 1;
            while (numUnits < keyset.Count)
            {
                numUnits <<= 1;
            }

            _units.Reserve(numUnits);

            ReserveId(0);
            this[0].IsUsed = true;
            _units[0].Offset = 1;
            _units[0].Label = 0;

            if (keyset.Count > 0)
            {
                BuildFromKeyset(keyset, 0, keyset.Count, 0, 0);
            }

            FixAllBlocks();
        }
    }

    // darts-clone Keyset: sorted, unique keys with values. Key k is bytes[starts[k]..starts[k + 1]]
    // and its value is values[k].
    internal readonly struct DoubleArrayKeyset
    {
        private readonly byte[] _bytes;
        private readonly int[] _starts;
        private readonly int[] _values;

        public DoubleArrayKeyset(byte[] bytes, int[] starts, int[] values, int count)
        {
            _bytes = bytes;
            _starts = starts;
            _values = values;
            Count = count;
        }

        public int Count { get; }

        public int Length(int key) => _starts[key + 1] - _starts[key];

        public int Value(int key) => _values[key];

        public byte Label(int key, int depth)
        {
            int position = _starts[key] + depth;
            return position < _starts[key + 1] ? _bytes[position] : (byte)0;
        }
    }

    internal struct DoubleArrayResultPair
    {
        public int Value { get; set; }
        public int Length { get; set; }
    };

    internal class DoubleArrayTrie
    {
        private readonly int _size;
        private readonly DoubleArrayUnit[] _array;

        internal DoubleArrayUnit[] ArrayUnits => _array;
        internal int Size => _size;

        // Keys must be unique, non-empty, free of NUL bytes, and sorted by unsigned byte order.
        public DoubleArrayTrie(in DoubleArrayKeyset keyset)
        {
            DoubleArrayBuilder builder = new DoubleArrayBuilder();
            builder.Build(keyset);

            _size = builder.UnitsSize;
            _array = builder.Units;
        }

        // The value of the key, or -1 if the key is not in the trie.
        public int ExactMatch(ReadOnlySpan<byte> key)
        {
            if (key.IsEmpty)
            {
                return -1;
            }

            int nodePos = 0;
            int keyPos = 0;
            int result = Traverse(key, ref nodePos, ref keyPos, key.Length);
            return result >= 0 ? result : -1;
        }

        public DoubleArrayTrie(DoubleArrayUnit[] preCompiledData)
        {
            if (preCompiledData is null)
            {
                throw new ArgumentNullException(nameof(preCompiledData));
            }

            _size = preCompiledData.Length;
            _array = preCompiledData;
        }

        public int CommonPrefixSearch(ReadOnlySpan<byte> key, Span<DoubleArrayResultPair> results, int nodePos = 0)
        {
            int numResults = 0;

            DoubleArrayUnit unit = _array[nodePos];
            nodePos ^= (int)unit.Offset;

            for (int i = 0; i < key.Length; ++i)
            {
                nodePos ^= key[i];
                unit = _array[nodePos];

                if (unit.Label != key[i])
                {
                    return numResults;
                }

                nodePos ^= (int)unit.Offset;

                if (unit.HasLeaf)
                {
                    if (numResults < results.Length)
                    {
                        results[numResults].Value = (int)_array[nodePos].Value;
                        results[numResults].Length = i + 1;
                    }

                    ++numResults;
                }
            }

            return numResults;
        }

        public int Traverse(ReadOnlySpan<byte> key, ref int nodePos, ref int keyPos, int length)
        {
            uint id = (uint)nodePos;
            DoubleArrayUnit unit = _array[id];

            if (length != 0)
            {
                for (; keyPos < length; ++keyPos)
                {
                    id ^= unit.Offset ^ key[keyPos];
                    unit = _array[id];
                    if (unit.Label != key[keyPos])
                    {
                        return -2;
                    }

                    nodePos = (int)id;
                }
            }
            else
            {
                for (; key[keyPos] != 0; ++keyPos)
                {
                    id ^= unit.Offset ^ key[keyPos];
                    unit = _array[id];
                    if (unit.Label != key[keyPos])
                    {
                        return -2;
                    }

                    nodePos = (int)id;
                }
            }

            if (!unit.HasLeaf)
            {
                return -1;
            }

            unit = _array[id ^ unit.Offset];
            return (int)unit.Value;
        }
    }
}

