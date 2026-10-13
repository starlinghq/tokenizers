// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Load-time benchmark for StarlingHq.Tokenizers.
//
//   LoadBench cold <tokenizer.json> [runs]        fresh process per run: Main to loaded, process wall, allocations, peak RSS
//   LoadBench warm <tokenizer.json> [iterations]  one warm-up load, then timed reloads in the same process
//   LoadBench child <tokenizer.json>              one load (used by cold); prints one JSON line
//   LoadBench encode <tokenizer.json> [rounds]    encode throughput check (the load change must not slow encoding)
//
// Cold runs measure what a user's first Tokenizer.FromFile costs in a new process with the file in the
// OS page cache: JIT of the loader included, process start-up excluded from load_ms but included in wall_ms.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using StarlingHq.Tokenizers;

long mainTimestamp = Stopwatch.GetTimestamp();
if (args.Length < 2)
{
    Console.Error.WriteLine("usage: LoadBench cold|warm|child <tokenizer.json> [count]");
    return 2;
}

string path = args[1];
int count = args.Length > 2 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 0;
return args[0] switch
{
    "child" => Child(path, mainTimestamp),
    "cold" => Cold(path, count > 0 ? count : 15),
    "warm" => Warm(path, count > 0 ? count : 10),
    "encode" => EncodeCheck(path, count > 0 ? count : 20),
    _ => 2,
};

static int Child(string path, long mainTimestamp)
{
    long t0 = Stopwatch.GetTimestamp();
    Tokenizer tokenizer = Tokenizer.FromFile(path);
    long t1 = Stopwatch.GetTimestamp();
    var ids = tokenizer.Encode("query: how much protein should a female eat").Ids;
    long t2 = Stopwatch.GetTimestamp();
    var result = new Dictionary<string, object>
    {
        ["main_to_load_start_ms"] = Stopwatch.GetElapsedTime(mainTimestamp, t0).TotalMilliseconds,
        ["load_ms"] = Stopwatch.GetElapsedTime(t0, t1).TotalMilliseconds,
        ["first_encode_ms"] = Stopwatch.GetElapsedTime(t1, t2).TotalMilliseconds,
        ["allocated_bytes"] = GC.GetTotalAllocatedBytes(precise: true),
        ["peak_working_set_bytes"] = Process.GetCurrentProcess().PeakWorkingSet64,
        ["retained_bytes"] = GC.GetTotalMemory(forceFullCollection: true),
        ["gc_count"] = new[] { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) },
        ["ids"] = string.Join(' ', ids),
    };
    Console.WriteLine(JsonSerializer.Serialize(result));
    GC.KeepAlive(tokenizer);
    return 0;
}

static int Cold(string path, int runs)
{
    string self = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
    var load = new List<double>();
    var wall = new List<double>();
    var alloc = new List<double>();
    var peak = new List<double>();
    var retained = new List<double>();
    string? ids = null;
    Console.Error.WriteLine($"load average: {ReadLoadAverage()}");
    for (int i = 0; i < runs; i++)
    {
        var psi = new ProcessStartInfo(self) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        psi.ArgumentList.Add("child");
        psi.ArgumentList.Add(path);
        long t0 = Stopwatch.GetTimestamp();
        using Process process = Process.Start(psi)!;
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        double w = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        if (process.ExitCode != 0)
        {
            Console.Error.WriteLine(stderr);
            return 1;
        }

        using JsonDocument doc = JsonDocument.Parse(stdout);
        JsonElement r = doc.RootElement;
        load.Add(r.GetProperty("load_ms").GetDouble());
        wall.Add(w);
        alloc.Add(r.GetProperty("allocated_bytes").GetDouble() / (1 << 20));
        peak.Add(r.GetProperty("peak_working_set_bytes").GetDouble() / (1 << 20));
        retained.Add(r.GetProperty("retained_bytes").GetDouble() / (1 << 20));
        string runIds = r.GetProperty("ids").GetString()!;
        if (ids is not null && ids != runIds)
        {
            Console.Error.WriteLine("ids differ between runs");
            return 1;
        }

        ids = runIds;
        Console.Error.WriteLine(string.Create(CultureInfo.InvariantCulture, $"run {i + 1}: load {load[^1]:F1} ms, wall {w:F1} ms"));
    }

    Console.WriteLine($"cold runs: {runs}");
    Console.WriteLine($"load average after: {ReadLoadAverage()}");
    Console.WriteLine(Summary("load_ms (Main: FromFile)", load));
    Console.WriteLine(Summary("process wall_ms", wall));
    Console.WriteLine(Summary("allocated MiB (whole process)", alloc));
    Console.WriteLine(Summary("peak working set MiB", peak));
    Console.WriteLine(Summary("managed heap after full GC, tokenizer alive, MiB", retained));
    Console.WriteLine($"ids: {ids}");
    return 0;
}

static int Warm(string path, int iterations)
{
    _ = Tokenizer.FromFile(path);
    var ms = new List<double>();
    var alloc = new List<double>();
    for (int i = 0; i < iterations; i++)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long a0 = GC.GetTotalAllocatedBytes(precise: true);
        long t0 = Stopwatch.GetTimestamp();
        Tokenizer t = Tokenizer.FromFile(path);
        ms.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        alloc.Add((GC.GetTotalAllocatedBytes(precise: true) - a0) / (double)(1 << 20));
        GC.KeepAlive(t);
    }

    Console.WriteLine($"warm reloads: {iterations} (after one warm-up load); load average: {ReadLoadAverage()}");
    Console.WriteLine(Summary("load_ms", ms));
    Console.WriteLine(Summary("allocated MiB per load", alloc));
    return 0;
}

static int EncodeCheck(string path, int rounds)
{
    Tokenizer tokenizer = Tokenizer.FromFile(path);
    string[] texts =
    [
        "query: how much protein should a female eat",
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Der schnelle braune Fuchs springt über den faulen Hund. Ça va très bien, merci !",
        "東京は日本の首都です。北京是中国的首都。서울은 한국의 수도입니다.",
        string.Concat(Enumerable.Repeat("Tokenizers split text into subword pieces; unigram models pick the most likely segmentation. ", 40)),
    ];
    var ms = new List<double>();
    long checksum = 0;
    for (int round = 0; round < rounds + 3; round++)
    {
        long t0 = Stopwatch.GetTimestamp();
        for (int i = 0; i < 200; i++)
        {
            foreach (string text in texts)
            {
                checksum += tokenizer.Encode(text).Ids.Sum();
            }
        }

        if (round >= 3)
        {
            ms.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        }
    }

    Console.WriteLine($"encode: {rounds} rounds of 200 x {texts.Length} texts (after 3 warm-up rounds); checksum {checksum}");
    Console.WriteLine(Summary("ms per round", ms));
    return 0;
}

static string Summary(string name, List<double> values)
{
    double[] s = [.. values.Order()];
    double Q(double p) => s[(int)Math.Round(p * (s.Length - 1))];
    return string.Create(
        CultureInfo.InvariantCulture,
        $"{name}: median {Q(0.5):F1}, min {s[0]:F1}, p25 {Q(0.25):F1}, p75 {Q(0.75):F1}, max {s[^1]:F1} (n={s.Length})");
}

static string ReadLoadAverage() =>
    File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : "n/a";
