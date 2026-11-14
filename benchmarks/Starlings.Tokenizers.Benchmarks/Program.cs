// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// BenchmarkDotNet entry point. Examples (docs/performance.md):
//
//   dotnet run -c Release --project benchmarks/Starlings.Tokenizers.Benchmarks -- --filter '*LoadBenchmarks*'
//   ... -- --filter '*LoadBenchmarks*' --runtimes net10.0 nativeaot10.0         (JIT against Native AOT)
//   ... -- --filter '*LoadBenchmarks*' --envVars DOTNET_TieredPGO:0              (runtime knobs)

using BenchmarkDotNet.Running;

BenchmarkSwitcher.FromAssembly(typeof(Starlings.Tokenizers.Benchmarks.Assets).Assembly).Run(args);
