// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace Starlings.Tokenizers.Benchmarks;

/// <summary>
/// <c>Tokenizer.FromFile</c>. <c>Cold</c>: one call in each of 20 fresh processes, so JIT compilation of the loader,
/// type loading and static constructors are included (the file itself is in the OS page cache). <c>Warm</c>: steady
/// state, after the loader has been compiled and tiered up. Cold minus warm is the first-call cost.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RunStrategy.ColdStart, launchCount: 20, warmupCount: 0, iterationCount: 1, id: "Cold")]
[SimpleJob(RunStrategy.Throughput, launchCount: 1, warmupCount: 5, iterationCount: 15, id: "Warm")]
public class LoadBenchmarks
{
    private string _path = "";

    [ParamsSource(nameof(AssetNames))]
    public string Asset { get; set; } = "";

    public static IEnumerable<string> AssetNames => Assets.Names;

    [GlobalSetup]
    public void Setup()
    {
        _path = Assets.Path(Asset);
        // Page-cache the file without touching the tokenizer code paths.
        _ = File.ReadAllBytes(_path);
    }

    [Benchmark]
    public Tokenizer FromFile() => Tokenizer.FromFile(_path);
}
