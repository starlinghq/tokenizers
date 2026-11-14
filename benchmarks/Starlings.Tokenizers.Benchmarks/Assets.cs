// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

namespace Starlings.Tokenizers.Benchmarks;

/// <summary>
/// The pinned <c>tokenizer.json</c> files the benchmarks load, by short name, from
/// <c>$STARLING_MODELS/&lt;repo&gt;/&lt;commit&gt;/&lt;file&gt;</c> (default <c>&lt;repo root&gt;/.models</c>, filled by
/// <c>tools/fetch-assets.sh</c>). BenchmarkDotNet runs each benchmark in a generated project under this project's
/// <c>bin/</c>, so the repository root is found by walking up from the working directory or the binary.
/// </summary>
public static class Assets
{
    private static readonly Dictionary<string, string> s_paths = new(StringComparer.Ordinal)
    {
        // WordPiece, BertNormalizer: 0.5 MB, 30,522 tokens
        ["minilm"] = "sentence-transformers/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json",
        // GPT-2 byte-level BPE (whisper-bpe): 2.7 MB, 50,257 tokens and 1,609 added tokens
        ["whisper"] = "openai/whisper-large-v3-turbo/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/tokenizer.json",
        // SentencePiece Unigram (xlmr-unigram-e5): 17 MB, 250,002 pieces
        ["e5"] = "intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3/tokenizer.json",
        // Qwen byte-level BPE (qwen35-bpe-chat, derived asset): 20 MB, 248,044 tokens
        ["qwen35"] = "Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/transformers-5.18.0/tokenizer.json",
        // SentencePiece-style BPE with byte fallback (gemma-bpe-embeddinggemma2): 32 MB, 262,144 tokens
        ["gemma"] = "google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json",
    };

    /// <summary>The short names, smallest file first.</summary>
    public static IEnumerable<string> Names => ["minilm", "whisper", "e5", "qwen35", "gemma"];

    /// <summary>The full path of a named asset; throws if it is not on disk.</summary>
    public static string Path(string name)
    {
        string path = System.IO.Path.Combine(ModelsRoot(), s_paths[name]);
        return File.Exists(path)
            ? path
            : throw new FileNotFoundException($"asset '{name}' not found; run tools/fetch-assets.sh or set STARLING_MODELS", path);
    }

    private static string ModelsRoot()
    {
        string? env = Environment.GetEnvironmentVariable("STARLING_MODELS");
        if (!string.IsNullOrEmpty(env))
        {
            return env;
        }

        foreach (string start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            for (DirectoryInfo? dir = new(start); dir is not null; dir = dir.Parent)
            {
                if (File.Exists(System.IO.Path.Combine(dir.FullName, "tools", "pinned-assets.txt")))
                {
                    return System.IO.Path.Combine(dir.FullName, ".models");
                }
            }
        }

        throw new DirectoryNotFoundException("repository root not found; set STARLING_MODELS");
    }
}
