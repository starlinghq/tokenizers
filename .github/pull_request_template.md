## Summary

<!-- What changed and why. Link the issue. -->

## Evidence

<!-- Test output, conformance or differential results, benchmark numbers. -->

## Checklist

- [ ] `dotnet test --solution Tokenizers.slnx` passes with `STARLING_REQUIRE_ASSETS=1`.
- [ ] No file under `conformance/` or `tests/*/TestData/` changed, or the PR carries only that change and says which reference produced it.
- [ ] New or changed behavior has a `CHANGELOG.md` entry under `Unreleased`; a new profile has `profiles/<id>.yaml`, `conformance/<id>/` and a pin in `tools/pinned-assets.txt`.
- [ ] Unsupported components still fail at load; capability status in the profile is explicit.
- [ ] `src/` gains no package dependency, and nothing references Microsoft.ML.Tokenizers.
- [ ] Adopted third-party source is recorded in `PROVENANCE.md` and `THIRD-PARTY-NOTICES.md`.
