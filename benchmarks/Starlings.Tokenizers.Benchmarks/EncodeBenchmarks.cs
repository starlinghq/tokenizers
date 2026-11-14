// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace Starlings.Tokenizers.Benchmarks;

/// <summary>
/// One <c>Encode</c> of a short query after <c>Tokenizer.FromFile</c> (in setup, not measured). <c>Cold</c>: the first
/// encode in each of 20 fresh processes, so the encode path is compiled during the measurement. <c>Warm</c>: steady
/// state. A real cold start pays load (<see cref="LoadBenchmarks"/>) plus this first encode.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RunStrategy.ColdStart, launchCount: 20, warmupCount: 0, iterationCount: 1, id: "Cold")]
[SimpleJob(RunStrategy.Throughput, launchCount: 1, warmupCount: 5, iterationCount: 15, id: "Warm")]
public class EncodeBenchmarks
{
    private const string Query = "query: how much protein should a female eat";
    private Tokenizer? _tokenizer;

    [ParamsSource(nameof(AssetNames))]
    public string Asset { get; set; } = "";

    public static IEnumerable<string> AssetNames => Assets.Names;

    [GlobalSetup]
    public void Setup() => _tokenizer = Tokenizer.FromFile(Assets.Path(Asset));

    [Benchmark]
    public TokenEncoding FirstEncode() => _tokenizer!.Encode(Query);
}
