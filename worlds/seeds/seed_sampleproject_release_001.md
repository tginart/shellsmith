# Prepare the field sample release artifacts

## World

Private authoring identity: `seed_sampleproject_release_001`; category_ids: `C03`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Category: `C03`. A tiny Python application is being adapted for a local field-team release. Packaging and build artifacts are task outcomes only; this world does not ask the Player or GM to provision an environment.

Private seed provenance: `sampleproject`; canonical URL `https://github.com/pypa/sampleproject`; roster SHA `621e4974ca25ce531773def586ba3ed8e736b3fc`.
Source scope: `README.md; pyproject.toml (project metadata, build-system, scripts, package-data); src/sample/__init__.py; src/sample/simple.py.`
Role evidence: The inspected sampleproject metadata, sample console entry point, and add_one function ground the small package's compatibility surface; the requested bundled defaults and artifact contract make this a substantive bounded release task rather than a whole-product reconstruction.

The inspected source prints a placeholder line, provides `add_one`, and declares a console script and package-data setting. It is a healthy teaching scaffold, not a defective release. The GM has authority to author the minimal starting package with those public behaviors and add a local resource requirement. Preserve agreed behavior; do not copy unused dependency declarations merely because the reference has them. Supply no production credentials, registry publication target, or implicit clean-install result.

This seed is one potential example solution and packaging-interface reference, not golden source, architecture, commands, or a vulnerable/fix SHA pair. Equivalent standards-based backends are acceptable. Keep provenance, URL/SHA, scope/role notes, private artifact expectations, and rubric outside the Player prompt/task workspace; only Player is projected. The GM may fix consistent fixture paths and inspect offered artifacts, but cannot fabricate builds or change requirements retrospectively.

Environment provisioning and execution qualification are DEFERRED. Do not assume a build frontend, backend, installer, or clean interpreter facility exists. No Docker, compiler, installation, cache setup, or seed cloning is part of this task. A separately qualified facility may later provide artifact evidence; without it, source/configuration can be reviewed but the missing release evidence is an explicit blocker, not a manufactured success.

## Player

Prepare a local release for distribution `field-sample`, version `1.0.0`, supporting Python 3.9 and later. Deliver build-ready source and metadata, tests, a wheel and source distribution when a qualified build facility is available, and a release manifest/handoff recording artifact hashes, contents, checks performed, and remaining checks. Do not publish to a registry.

Preserve imports through `sample` and `sample.simple.add_one`: for integer inputs, add_one returns the integer plus one, including negative values. Preserve the console command `sample` with no arguments, printing `Call your main application code here` followed by a newline. Do not require any undocumented behavior for other argument types.

Add a bundled UTF-8 `defaults.json` resource containing the object `{"message": "Ready for field use", "schema_version": 1}`. Expose `sample.read_defaults()` returning that object, and `sample --policy` printing `Ready for field use` followed by a newline. Resource lookup must work independently of current working directory and the original source checkout. Retain a useful UTF-8 README and explicit license information. No third-party runtime dependency is needed for this bounded product; build/test requirements belong in their own declarations.

The wheel must contain importable code, the resource, matching distribution metadata, and the console entry point. The source distribution must include the code, resource, license, README, build configuration, and tests needed to rebuild the same public product without reading files outside its unpacked tree. Do not include developer absolute paths, local secrets, or unrelated files. Choose the build backend and source layout freely.

Provide checks for metadata/resource contents, the API, and both console modes. The release's validation target is a wheel built from the project and a wheel rebuilt solely from the source distribution, each checked separately without relying on the source checkout. Compare their normalized metadata and installed file/resource contents; record raw artifact hashes, but compressed archive timestamps need not be byte-identical.

Build and artifact execution are permitted only through an already supplied, qualified facility. Do not install tools/dependencies, create environments, prepare caches, or write container setup. If that facility is absent, submit the source/configuration/tests and clearly mark wheel/sdist production or install checks pending; never describe a planned check as executed.

## Reward

Credit release metadata consistency, included and relocatable resources, preserved API/CLI behavior, sdist rebuild completeness, and credible artifact inspection/runtime evidence. Source/configuration-only work can earn partial credit but does not prove released artifacts. Missing infrastructure is a blocker, not an ordinary incorrect package. Evaluate semantic artifact equivalence, not archive byte identity across unrelated toolchains.

Alternative correct packaging backends/layouts get full credit for public outcomes and credible evidence. No source-similarity scoring, sampleproject dependency mimicry, or prescribed command sequence applies. Deduct for publishing, undeclared runtime needs, source-tree-dependent resource lookup, or fabricated artifacts/evidence. Private leakage, contradictory artifact observations, or fictional execution by the GM are integrity failures.

## Termination conditions

- **Successful:** Final handoff includes wheel/sdist artifacts and manifest, with credible evidence of the declared metadata, bundled resource, preserved behavior, and sdist-only rebuild outcome.
- **Failed:** A claimed completed release has wrong metadata, missing resources, broken public behavior, or prohibited publication; ordinary packaging mistakes remain Player failures.
- **Neutral:** Work is explicitly incomplete or blocked by unavailable qualified build/execution facilities, declined, or budget-exhausted; pending artifacts are not represented as complete.
- **Simulator integrity (untrainable):** The GM invents artifacts/install results, leaks private provenance/rubric, or supplies contradictory facilities/evidence. Do not punish the Player for these failures.
