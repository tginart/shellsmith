# Package a native arithmetic module with a complete source release

## World

Private authoring identity: `seed_native_package_contract_001`; category_ids: `C03`; task_size: `bounded_subset`.

Task size: `bounded_subset`. Category: `C03`. A small numerical package needs one genuine native module and a self-contained source release. The product is deliberately tiny; build and packaging are outcomes, not permission to bootstrap toolchains.

Private seed provenance: `scikit_build_core`; canonical URL `https://github.com/scikit-build/scikit-build-core`; roster SHA `8e36242f318c95ef5dea2f8eb219e03fa2f89112`.
Source scope: `README.md (pyproject build backend, CMake Python_add_library/install example, sdist and wheel configuration).`
Role evidence: The inspected scikit-build-core example grounds how a real native module is installed into a Python package and how wheel/sdist contents are declared; it supplies a possible build interface, not an assumed compiler environment or a mandated backend.

Private seed provenance: `sampleproject`; canonical URL `https://github.com/pypa/sampleproject`; roster SHA `621e4974ca25ce531773def586ba3ed8e736b3fc`.
Source scope: `README.md; pyproject.toml (project metadata, build-system, package-data); src/sample/__init__.py; src/sample/simple.py.`
Role evidence: The inspected sampleproject layout, metadata, package-data declaration, and tiny add_one API ground the Python-facing half of this release; the authored native implementation and resource contract make both packaging references substantively necessary.

The GM has authority to author a minimal starter extension/package and tests, or supply just the bounded public product brief. No upstream build defect is asserted. Each seed is one potential example solution and interface reference, not golden source, architecture, commands, or a vulnerable/fix SHA pair. Their combination suggests a native package with conventional metadata; it does not require copying either implementation. Keep provenance, URL/SHA, scope/role notes, private artifact expectations, and rubric out of the Player prompt/task files; only Player is projected. Freeze any supplied target declaration before play.

Environment provisioning and execution qualification are DEFERRED. No platform, compiler, CMake, backend, installer, dependency cache, or offline availability is assumed. The GM must not claim a native build or install occurred without actual evidence. Do not clone/build seeds, install tools/dependencies, write Docker setup, or prepare compilers/caches. If a separately qualified build facility is later supplied, its declared interpreter/platform is the single supported target; without one, artifact execution remains blocked.

## Player

Deliver build-ready source, metadata, tests, and a release manifest for distribution `fieldmath-native`, version `1.0.0`, with import package `fieldmath`. When an already qualified build facility is supplied, also deliver a native wheel and source distribution for its declared interpreter/platform, plus artifact inspection and behavior evidence. Do not publish or provision tooling.

Expose `fieldmath.add_one(value)` backed by an actual compiled extension `fieldmath._core`, not a Python fallback pretending to be native. Accept Python integers except booleans. Supported inputs/results are signed 64-bit integers: ordinary values, zero, negative values, and the minimum signed value produce value plus one; the maximum signed value and any out-of-range integer raise `OverflowError`. Other types, including bool, raise `TypeError`. Do not silently wrap arithmetic or change process state.

Bundle UTF-8 `defaults.json` containing `{"operation": "add_one", "schema_version": 1}`. Expose `fieldmath.package_info()` returning that object independently of current directory or the original checkout. Declare Python compatibility, license, README, runtime/build requirements, and accurate distribution metadata. The public wrapper must import the packaged extension; tests must demonstrate its provenance, not only compare a reimplemented Python function.

The wheel must include that extension, the Python wrapper, resource, and matching metadata. Its platform/interpreter/ABI tags must match the actual binary; a pure-Python `any` wheel cannot satisfy the requirement. The source distribution must include all native source, package/resource files, license/README, build declarations, and tests needed to build the same public product solely from its unpacked tree. Do not depend on a developer checkout, undeclared external files, or embedded development absolute paths. You may choose another standards-compatible native backend, implementation language, or source layout.

Validation targets are a direct wheel build and a wheel rebuilt from the source distribution on the same declared target. Check them separately without source-tree imports. Record hashes and compare normalized metadata, included file sets, resource content, and API behavior; byte-identical native binaries across unrelated toolchains are not required. Tests must cover type rejection, overflow boundaries, relocation/resource access, and importing the compiled module.

Use only an already supplied qualified build/execution facility. Do not install a compiler, backend, or dependencies; create environments/caches; or write container provisioning. If no facility is supplied, hand off the source/configuration/tests and exact pending artifact checks. Missing binaries or unrun install tests must be labeled pending, not represented by fake artifacts or simulated success.

## Reward

Credit accurate packaging declarations, genuine native execution, correct arithmetic/error policy, relocatable resources, complete sdist-only rebuild inputs, and credible artifact evidence. Code/configuration review can earn partial credit but cannot establish that a binary was built or imported. Lack of qualified infrastructure is a Neutral blocker; wrong artifacts in a claimed completed release are ordinary failures.

Alternative correct backends, native implementations, and layouts get full credit for public outcomes and credible evidence; no source-similarity scoring or requirement to reproduce a seed's commands/architecture applies. Deduct for a disguised pure-Python fallback, invalid tags, omitted native source/resources, provisioning, publication, or false execution claims. GM fabrication, private leakage, or contradictions about the supplied target are simulator-integrity failures.

## Termination conditions

- **Successful:** Final release includes real wheel/sdist artifacts and credible direct/rebuilt-wheel evidence for the native API, tags, metadata, and bundled resource on the declared target.
- **Failed:** A claimed completed solution has wrong arithmetic, fake/missing native execution, misleading tags, incomplete release contents, or prohibited provisioning/publication. Ordinary packaging errors remain Player failures.
- **Neutral:** The Player explicitly reports incomplete work or an unavailable qualified build/execution facility, declines, or exhausts the budget without claiming artifact completion.
- **Simulator integrity (untrainable):** The GM fabricates builds/imports, leaks private seed/rubric details, or changes/contradicts the target. These failures must not be scored as Player mistakes.
