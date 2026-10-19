# Releasing

Starlings.Tokenizers ships often: a new tokenizer profile, a component that unlocks new models, or a fix can each be its own release. Pushing a tag does the whole release: it rebuilds from a clean clone, runs every test with every asset, packs, checks the package from outside the repository, waits for a maintainer's approval, then publishes the exact bytes it checked.

## Versions

- **Scheme.** SemVer 2, set in [`Directory.Build.props`](../Directory.Build.props) (`VersionPrefix`, `VersionSuffix`). Releases are `0.MINOR.PATCH` (from 0.3.0); a candidate before a release may be `0.MINOR.PATCH-preview.N`. Earlier releases were all previews. The 0.x major keeps the API-stability signal honest: the API can change between minor versions.
  - **MINOR**: the public API changes, or a profile is dropped.
  - **PATCH**: a behavior fix, a new component, or a new profile.
  - **preview.N**: counts candidates of the same version (optional).
- **Tokenizer output is behavior.** A change in token ids, offsets or decoded text is a behavior change even if no C# signature moves. It needs a `CHANGELOG.md` entry and a rerun of the conformance tests of every profile it touches.
- **Immutability.** A published version is never replaced. Any change to shipped bytes needs a new version. The release workflow pushes without `--skip-duplicate`, so publishing an existing version fails.
- **Consumers pin.** Model packages pin the tokenizer version they were qualified with. A newer tokenizer is not automatically certified for existing model bundles.

## Cutting a release

1. In a pull request, set the version in `Directory.Build.props` and move the `Unreleased` entries in `CHANGELOG.md` under a heading `## [<version>] - <YYYY-MM-DD>`. Update the `library:` line of every profile you requalified.
2. Merge it to `main`.
3. Tag the merge commit and push the tag:

   ```sh
   git switch main && git pull
   git tag v0.3.1
   git push origin v0.3.1
   ```

4. The [`release`](../.github/workflows/release.yml) workflow runs:
   - **candidate**: runs the boundary check (`tools/check-boundary.py`), refuses a tag that differs from the project version, a tag on a commit that is not on `main`, or a version without a `CHANGELOG.md` section. Then it runs the release drill (below) and uploads the verified `.nupkg` and `.snupkg` as the `package` artifact.
   - **publish**: waits for a reviewer to approve the `release` environment. It then gets a short-lived nuget.org key through trusted publishing, pushes the package (the symbol package goes with it), and creates the GitHub release with the changelog section as notes and the packages attached. A version with a suffix is marked as a pre-release.
5. Approve the deployment in the workflow run. The package shows on nuget.org once its validation finishes, usually within minutes.

Running the `release` workflow by hand (Actions > release > Run workflow) builds and uploads a candidate but never publishes. Use it to dry-run a release from a branch.

## Release drill

```sh
tools/release-drill.sh            # DRILL_KEEP=1 keeps the temporary directory
```

The drill builds the committed `HEAD` from a clean clone with an empty private NuGet cache. It then:

1. Checks that no project, props or solution file references Microsoft.ML.Tokenizers.
2. Runs the tests with `STARLING_REQUIRE_ASSETS=1` and locked restore, and fails on any skip.
3. Packs and checks the nuspec: version, the `MIT OR Apache-2.0` license expression, repository commit, no dependencies, and the packed README, notices and every `profiles/*.yaml`.
4. Pushes the package to a temporary local folder feed. It never pushes to nuget.org.
5. Builds a consumer in a temporary directory outside the repository. The consumer's `NuGet.config` lists only that feed and nuget.org, with `Starlings.*` mapped to the local feed.
6. Fails if Microsoft.ML.Tokenizers appears in the consumer's restore graph or package cache, if the restored package differs from the packed one, or if any consumer file names the repository path.
7. Runs the consumer. It checks the pinned e5 `tokenizer.json` digest and compares every e5 fixture case (encode, decode, truncation, padded batch) with `conformance/xlmr-unigram-e5/`.

The package and logs land in `artifacts/release-drill/` (ignored by Git). CI runs the drill on every pull request.

## One-time setup

These are account settings. An agent or contributor cannot apply them.

1. **GitHub environment.** In the repository, go to Settings > Environments > New environment, and name it `release`.
   - Add the maintainers as required reviewers.
   - Under deployment branches and tags, allow only tags matching `v*`.
   - Add the environment variable `NUGET_USER`: the nuget.org **user name of the person who created** the trusted publishing policy, not the organization that owns it. With the organization's name, the token exchange fails with "No matching trust policy owned by user ... was found".
   - No secret is needed.
2. **nuget.org trusted publishing.** Sign in as that user and open Trusted Publishing. Create a policy with:

   | Field | Value |
   | --- | --- |
   | Package owner | `starlinghq` |
   | Repository owner | `starlinghq` |
   | Repository | `tokenizers` |
   | Workflow file | `release.yml` |
   | Environment | `release` |

   The user must be an administrator of the `starlinghq` organization on nuget.org, and the organization needs a confirmed email address (organization Settings on nuget.org); without one the token exchange fails with "The organization ... does not have a confirmed email address".
3. **NuGet ID prefix.** Packages use the `Starlings.` prefix. `Starling.` is reserved on nuget.org by another owner (a push under it fails with 409 "The package ID is reserved"). Ask account@nuget.org to reserve `Starlings` for the `starlinghq` organization; see the NuGet ID prefix reservation documentation. Reservation is not required to publish, but it marks packages as verified and stops others from publishing under the prefix.
4. **Branch protection.** Protect `main`: require pull requests, require the `ci` checks (`boundary`, `test (ubuntu-latest)`, `test (windows-latest)`, `release drill`, `reference runners (ruff)`), and require review from code owners. `CODEOWNERS` covers `conformance/`, `tests/*/TestData/`, `tools/pinned-assets.txt` and the release files.
5. **Actions.** Keep the default workflow token read-only (Settings > Actions > General). The release workflow asks for `contents: write` and `id-token: write` on the publish job only.
