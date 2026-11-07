"""Pinned Docker/Codex policy, not a claim that seccomp overrides AppArmor.

Offline tests pin the complete upstream policy and the exact reviewed delta.
Linux tests compile the full policy with real libseccomp and load its BPF in a
compiled child. For safety, ALLOW for the one syscall under test becomes a
unique errno; a second filter guarantees no probed operation can execute. This
checks kernel argument matching, NOT successful namespaces/mounts/confinement.

Opt in to the separate actual-engine test with SHELLSMITH_SECCOMP_TEST_IMAGE
naming an already-built local Shellsmith image. It runs no model, uses no real
credentials, never pulls/builds an image, and must FAIL (not skip or relax an
LSM) when the selected engine cannot run the restricted pinned Codex sandbox.
"""

from __future__ import annotations

import copy
import ctypes
import ctypes.util
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import struct
import subprocess
import sys
import uuid

import pytest

from shellsmith.config import CODEX_CLI_VERSION

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "docker/seccomp-codex.json"
PROVENANCE_PATH = ROOT / "docker/seccomp-codex.provenance.json"
UPSTREAM_SHA256 = "785b2429264afba4d594320337cb17f144f3c7d51585f9805eef72e28f4f9334"
UPSTREAM_CANONICAL_SHA256 = "20ef2b6bfd980e7dd56a971d9ec0d71bc0f2d1184415f2d968a19bbcff9ee2b2"
DEFAULT_CAPS = {
    "CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FSETID", "CAP_FOWNER", "CAP_MKNOD",
    "CAP_NET_RAW", "CAP_SETGID", "CAP_SETUID", "CAP_SETFCAP", "CAP_SETPCAP",
    "CAP_NET_BIND_SERVICE", "CAP_SYS_CHROOT", "CAP_KILL", "CAP_AUDIT_WRITE",
}
NEWNS, NEWCGROUP, NEWUTS = 0x00020000, 0x02000000, 0x04000000
NEWIPC, NEWUSER, NEWPID, NEWNET = 0x08000000, 0x10000000, 0x20000000, 0x40000000
BWRAP_CLONE = NEWUSER | NEWNS | NEWPID | NEWIPC | 17  # SIGCHLD
MOUNT_EXACT = {0x44000, 0x8C000, 0x4C000, 6, 14, 10, 0xC0EDD000, 0x9000, 0xD000}
REMOUNT_REQUIRED = 0x9022  # SILENT | BIND | REMOUNT | NOSUID
REMOUNT_OPTIONAL_BITS = (1, 4, 8, 0x400, 0x800, 0x200000)
REMOUNT_OPTIONAL = sum(REMOUNT_OPTIONAL_BITS)
ALLOW = "SCMP_ACT_ALLOW"
ERRNO = "SCMP_ACT_ERRNO"


def _read_policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


def _additions(policy: dict) -> list[dict]:
    return [r for r in policy["syscalls"] if r.get("comment", "").startswith("Shellsmith:")]


def _effective_rules(policy: dict, arch: str, caps: set[str] | None = None,
                     kernel: tuple[int, int] = (6, 1)) -> list[dict]:
    """Docker includes/excludes are resolved at creation, not after userns entry."""
    caps = DEFAULT_CAPS if caps is None else caps
    result = []
    for rule in policy["syscalls"]:
        include, exclude = rule.get("includes", {}), rule.get("excludes", {})
        if include.get("arches") and arch not in include["arches"]:
            continue
        if not set(include.get("caps", [])) <= caps:
            continue
        if arch in exclude.get("arches", []) or set(exclude.get("caps", [])) & caps:
            continue
        if include.get("minKernel") and kernel < tuple(map(int, include["minKernel"].split("."))):
            continue
        if exclude.get("minKernel") and kernel >= tuple(map(int, exclude["minKernel"].split("."))):
            continue
        result.append(rule)
    return result


def _decision(name: str, args: tuple[int, ...] = (), *, arch: str = "amd64",
              caps: set[str] | None = None) -> tuple[str, int | None]:
    """Structural oracle only; the native tests below also exercise real BPF."""
    policy = _read_policy()
    args = (*args, *([0] * (6 - len(args))))
    matches = []
    for rule in _effective_rules(policy, arch, caps):
        if name not in rule["names"]:
            continue
        def compare(cmp: dict) -> bool:
            value, first = args[cmp["index"]], cmp["value"]
            return {
                "SCMP_CMP_EQ": lambda: value == first,
                "SCMP_CMP_LT": lambda: value < first,
                "SCMP_CMP_MASKED_EQ": lambda: value & first == cmp.get("valueTwo", 0),
            }[cmp["op"]]()
        if all(compare(cmp) for cmp in rule.get("args", [])):
            matches.append((rule["action"], rule.get("errnoRet")))
    # ERRNO takes priority over ALLOW if a future edit introduces overlap.
    return next((m for m in matches if m[0] == ERRNO),
                matches[0] if matches else (policy["defaultAction"], policy["defaultErrnoRet"]))


def test_profile_has_reproducible_pinned_docker_ancestry() -> None:
    profile = _read_policy()
    provenance = json.loads(PROVENANCE_PATH.read_text())
    assert provenance["policy_sha256"] == hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
    assert provenance["upstream"]["sha256"] == UPSTREAM_SHA256
    assert provenance["upstream"]["canonical_sha256"] == UPSTREAM_CANONICAL_SHA256
    assert provenance["upstream"]["commit"] == "245180c51918481c0525424b3ee025d2b435d46c"
    assert provenance["upstream"]["license"] == "Apache-2.0"
    assert "Apache License" in provenance["upstream"]["license_text"]
    assert CODEX_CLI_VERSION == provenance["codex"]["version"] == "0.159.2", "Reaudit on Codex upgrades"
    assert provenance["codex"]["commit"] == "ff6aec96948b70d94983af2641a6b67c94faeff5"
    assert (profile["defaultAction"], profile["defaultErrnoRet"]) == (ERRNO, 1)

    # Reverse only the documented edits and hash ALL remaining upstream fields.
    # This catches accidental allowlist growth, removed restrictions and changes
    # to architecture mappings, capability gates or clone3 errno semantics.
    baseline = copy.deepcopy(profile)
    baseline["syscalls"] = [r for r in baseline["syscalls"] if r not in _additions(profile)]
    changes = provenance["changes"]
    baseline["archMap"] = changes["original_arch_map"]
    baseline["syscalls"].insert(changes["removed_rule_index"], changes["removed_rule"])
    names = baseline["syscalls"][0]["names"]
    index = names.index("splice")
    names[index:index] = ["socketcall", "socketpair"]
    for rule in baseline["syscalls"]:
        if rule["names"] == ["socket", "socketpair"]:
            rule["names"] = ["socket"]
    canonical = json.dumps(baseline, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == UPSTREAM_CANONICAL_SHA256


def test_only_reviewed_native_argument_rules_are_added() -> None:
    def eq(name: str, index: int, value: int) -> tuple:
        return (name, ((index, "SCMP_CMP_EQ", value, 0),))
    expected = {
        eq("unshare", 0, NEWUSER), eq("unshare", 0, NEWUSER | NEWNS),
        eq("clone", 0, BWRAP_CLONE), eq("clone", 0, BWRAP_CLONE | NEWNET),
        eq("umount2", 1, 2), ("pivot_root", ()),
        *(eq("mount", 3, flags) for flags in MOUNT_EXACT),
        ("mount", ((3, "SCMP_CMP_MASKED_EQ", ((1 << 64) - 1) ^ REMOUNT_OPTIONAL, REMOUNT_REQUIRED),)),
    }
    additions = _additions(_read_policy())
    actual = set()
    for rule in additions:
        assert rule["action"] == ALLOW
        assert rule["includes"] == {"arches": ["amd64", "arm64"]}
        assert set(rule) <= {"names", "action", "includes", "args", "comment"}
        assert len(rule["names"]) == 1
        # Multiple comparisons on the same argument cannot be portably ANDed
        # by libseccomp; one mask expresses both required and forbidden bits.
        assert len(rule.get("args", [])) <= 1
        actual.add((rule["names"][0], tuple(
            (a["index"], a["op"], a["value"], a.get("valueTwo", 0)) for a in rule.get("args", [])
        )))
    assert actual == expected
    assert len(additions) == len(expected) == 16
    assert not any("CAP_SYS_ADMIN" in r.get("includes", {}).get("caps", [])
                   for r in _read_policy()["syscalls"])


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_namespace_creation_cannot_join_or_create_unreviewed_namespaces(arch: str) -> None:
    namespace_bits = (NEWNS, NEWCGROUP, NEWUTS, NEWIPC, NEWUSER, NEWPID, NEWNET)
    for selected in itertools.product((False, True), repeat=len(namespace_bits)):
        flags = sum(bit for bit, enabled in zip(namespace_bits, selected) if enabled)
        expected_clone = flags in (0, BWRAP_CLONE & ~17, (BWRAP_CLONE & ~17) | NEWNET)
        assert (_decision("clone", (flags | 17,), arch=arch)[0] == ALLOW) == expected_clone
        assert (_decision("unshare", (flags,), arch=arch)[0] == ALLOW) == (flags in (NEWUSER, NEWUSER | NEWNS))
    for extra in (0x100, 0x400, 0x10000, 1 << 32, 1 << 63):
        assert _decision("clone", (BWRAP_CLONE | extra,), arch=arch) == (ERRNO, 1)
        assert _decision("unshare", (NEWUSER | NEWNS | extra,), arch=arch) == (ERRNO, 1)
    assert _decision("clone", (0x100 | 0x200 | 0x400 | 0x800 | 0x10000,), arch=arch)[0] == ALLOW
    assert _decision("clone3", (0, 88), arch=arch) == (ERRNO, 38)
    assert _decision("setns", (0, NEWUSER), arch=arch) == (ERRNO, 1)
    assert _decision("setns", (0, NEWNET), arch=arch) == (ERRNO, 1)
    assert _decision("setns", caps=DEFAULT_CAPS | {"CAP_SYS_ADMIN"}) == (ERRNO, 1)


def _remount_flags():
    for selected in itertools.product((False, True), repeat=len(REMOUNT_OPTIONAL_BITS)):
        yield REMOUNT_REQUIRED | sum(bit for bit, enabled in zip(REMOUNT_OPTIONAL_BITS, selected) if enabled)


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_mount_flags_cover_bwrap_but_not_shared_move_or_arbitrary_remounts(arch: str) -> None:
    for flags in [*MOUNT_EXACT, *_remount_flags()]:
        assert _decision("mount", (0, 0, 0, flags, 0), arch=arch)[0] == ALLOW
    for flags in (0, 1, 0x20, 0x1000, 0x2000, 0x100000, 0x104000,
                  REMOUNT_REQUIRED & ~2, REMOUNT_REQUIRED | 0x2000,
                  REMOUNT_REQUIRED | 0x100000, REMOUNT_REQUIRED | (1 << 63)):
        assert _decision("mount", (0, 0, 0, flags, 0), arch=arch) == (ERRNO, 1)
    assert _decision("pivot_root", arch=arch)[0] == ALLOW
    for flags in (0, 1, 3, 4, 8, 0xFFFFFFFF):
        assert _decision("umount2", (0, flags), arch=arch) == (ERRNO, 1)
    assert _decision("umount2", (0, 2), arch=arch)[0] == ALLOW


@pytest.mark.parametrize("name", [
    "setns", "umount", "fsopen", "fsconfig", "fsmount", "fspick", "open_tree",
    "move_mount", "mount_setattr", "open_by_handle_at", "bpf", "perf_event_open",
    "io_uring_setup", "io_uring_enter", "io_uring_register", "userfaultfd",
    "init_module", "finit_module", "delete_module", "kexec_load", "keyctl",
    "add_key", "request_key", "socketcall", "not_a_real_syscall",
])
def test_sensitive_unneeded_operations_remain_default_denied(name: str) -> None:
    assert _decision(name) == (ERRNO, 1)


@pytest.mark.parametrize("arch", ["amd64", "arm64"])
def test_no_vm_socket_crypto_or_compat_socketcall_bypass(arch: str) -> None:
    for name in ("socket", "socketpair"):
        for family in (1, 2, 10, 16):  # Unix, IPv4, IPv6, netlink (bwrap loopback)
            assert _decision(name, (family, 1, 0), arch=arch)[0] == ALLOW
        for family in (38, 40, 46, 0xFFFFFFFF, 1 << 32):
            assert _decision(name, (family, 1, 0), arch=arch) == (ERRNO, 1)
    assert _decision("socketcall", arch=arch) == (ERRNO, 1)


def test_native_only_abis_prevent_libseccomp_socketcall_rewrite_bypass() -> None:
    # libseccomp's non-strict rule_add rewrites x86/ARM socket rules into
    # socketcall dispatch rules, losing pointer-contained family predicates.
    # Merely deleting the explicit socketcall allow is therefore insufficient.
    assert _read_policy()["archMap"] == [
        {"architecture": "SCMP_ARCH_X86_64", "subArchitectures": []},
        {"architecture": "SCMP_ARCH_AARCH64", "subArchitectures": []},
    ]


def test_unsupported_native_architecture_does_not_get_new_privileged_rules() -> None:
    for name, args in (("unshare", (NEWUSER | NEWNS,)), ("mount", (0, 0, 0, 6)),
                       ("pivot_root", ()), ("umount2", (0, 2))):
        assert _decision(name, args, arch="s390x") == (ERRNO, 1)


class _ArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


@pytest.fixture(scope="module")
def native_libseccomp():
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        pytest.skip("requires native Linux amd64/arm64 and real libseccomp; macOS cannot validate Linux enforcement")
    path = ctypes.util.find_library("seccomp")
    if not path:
        pytest.skip("real libseccomp shared library is not installed")
    lib = ctypes.CDLL(path)
    signatures = {
        "seccomp_init": (ctypes.c_void_p, [ctypes.c_uint32]),
        "seccomp_release": (None, [ctypes.c_void_p]),
        "seccomp_arch_resolve_name": (ctypes.c_uint32, [ctypes.c_char_p]),
        "seccomp_arch_add": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32]),
        "seccomp_syscall_resolve_name": (ctypes.c_int, [ctypes.c_char_p]),
        "seccomp_rule_add_array": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                                ctypes.c_uint, ctypes.POINTER(_ArgCmp)]),
        "seccomp_export_bpf": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    return lib


def _export_bpf(lib, target: str, path: Path) -> int:
    """Compile the whole policy, preserving Docker capability/arch predicates."""
    number = lib.seccomp_syscall_resolve_name(target.encode())
    assert number >= 0, f"libseccomp cannot resolve required native syscall {target}"
    policy = _read_policy()
    context = lib.seccomp_init(0x00050000 | policy["defaultErrnoRet"])
    assert context
    try:
        arch = "amd64" if platform.machine() == "x86_64" else "arm64"
        native_name = "SCMP_ARCH_X86_64" if arch == "amd64" else "SCMP_ARCH_AARCH64"
        mapping = next(a for a in policy["archMap"] if a["architecture"] == native_name)
        for name in mapping["subArchitectures"]:
            token = lib.seccomp_arch_resolve_name(name.removeprefix("SCMP_ARCH_").lower().encode())
            assert token
            assert lib.seccomp_arch_add(context, token) == 0
        kernel = tuple(int(x) for x in platform.release().split(".")[:2])
        for rule in _effective_rules(policy, arch, kernel=kernel):
            comparisons = rule.get("args", [])
            array = (_ArgCmp * len(comparisons))(*(
                _ArgCmp(c["index"], {"SCMP_CMP_LT": 2, "SCMP_CMP_EQ": 4, "SCMP_CMP_MASKED_EQ": 7}[c["op"]],
                        c["value"], c.get("valueTwo", 0)) for c in comparisons
            ))
            for name in rule["names"]:
                syscall = lib.seccomp_syscall_resolve_name(name.encode())
                if syscall == -1:
                    # As in runc, names unknown to the installed libseccomp
                    # cannot be added. Required probe syscalls are checked above.
                    continue
                action = 0x7FFF0000 if rule["action"] == ALLOW else 0x00050000 | rule["errnoRet"]
                if name == target and rule["action"] == ALLOW:
                    action = 0x00050000 | 123  # observed ALLOW; no operation executes
                result = lib.seccomp_rule_add_array(context, action, syscall, len(array), array)
                assert result == 0, f"libseccomp rejected {name}: {result}"
        with path.open("wb") as stream:
            assert lib.seccomp_export_bpf(context, stream.fileno()) == 0
    finally:
        lib.seccomp_release(context)
    return number


_PROBE_C = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 9 && argc != 10) return 2;
    struct rlimit cores = {0, 0};
    if (setrlimit(RLIMIT_CORE, &cores)) return 7;
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 3;
    struct sock_filter instructions[4096];
    size_t length = fread(instructions, sizeof(instructions[0]), 4096, file);
    if (ferror(file) || !feof(file) || !length) return 4;
    fclose(file);
    long nr = strtol(argv[2], NULL, 10);
    unsigned long args[6];
    for (int i = 0; i < 6; i++) args[i] = strtoul(argv[i + 3], NULL, 10);
    // Never perform the target operation, even if a broken profile allows it.
    // An equal-precedence ERRNO in the newer tested filter supersedes 124.
    struct sock_filter guard[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, nr, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | 124),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
    };
    struct sock_fprog guard_program = {4, guard};
    struct sock_fprog program = {(unsigned short) length, instructions};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &guard_program) ||
        prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program)) {
        perror("install seccomp");
        return 5;
    }
    errno = 0;
    long result;
    if (argc == 10) {
#ifdef __x86_64__
        // Explicit i386 ABI; syscall 102 is socketcall, with a NULL argument
        // pointer. The guard blocks it even if the policy accidentally adds
        // a compat ABI; a correct native-only profile kills it with SIGSYS.
        // Older compat entry paths clear r8-r11. Account for those registers
        // even on the guarded-return path; GCC may keep errno's address there.
        __asm__ volatile("int $0x80" : "=a"(result)
                         : "0"(nr), "b"(1UL), "c"(0UL)
                         : "r8", "r9", "r10", "r11", "cc", "memory");
        if (result < 0 && result >= -4095) { errno = (int)-result; result = -1; }
#else
        return 6;
#endif
    } else {
        result = syscall(nr, args[0], args[1], args[2], args[3], args[4], args[5]);
    }
    int error = errno;
    printf("%ld %d\n", result, error);
    return 0;
}
"""


@pytest.fixture(scope="module", params=["-O0", "-O2"])
def kernel_probe(native_libseccomp, tmp_path_factory, request):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler is unavailable for the native kernel BPF probe")
    directory = tmp_path_factory.mktemp("seccomp-probe")
    headers = subprocess.run([compiler, "-x", "c", "-E", "-"],
                             input="#include <linux/filter.h>\n#include <linux/seccomp.h>\n#include <sys/prctl.h>\n",
                             capture_output=True, text=True, timeout=30)
    if headers.returncode:
        pytest.skip("Linux C development headers are unavailable for the kernel BPF probe")
    source, executable = directory / "probe.c", directory / "probe"
    source.write_text(_PROBE_C)
    subprocess.run([compiler, "-std=c11", request.param, "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
                   capture_output=True, text=True, check=True, timeout=30)
    return executable


def test_real_libseccomp_and_compiled_kernel_probe(native_libseccomp, kernel_probe, tmp_path) -> None:
    cases = {
        "unshare": [(NEWUSER,), (NEWUSER | NEWNS,), (NEWNS,), (NEWNET,), (NEWUSER | NEWUTS,),
                    (NEWUSER | NEWNS | (1 << 63),)],
        "clone": [(17,), (BWRAP_CLONE,), (BWRAP_CLONE | NEWNET,), (NEWNS | 17,),
                  (NEWUSER | NEWNS | 17,), (BWRAP_CLONE | NEWUTS,), (BWRAP_CLONE | (1 << 32),)],
        "clone3": [(0, 88)],
        "mount": [(0, 0, 0, flags, 0) for flags in (
            *MOUNT_EXACT, *_remount_flags(), 0, 0x2000, 0x104000, REMOUNT_REQUIRED & ~2,
            REMOUNT_REQUIRED | (1 << 63))],
        "pivot_root": [(0, 0)],
        "umount2": [(0, flags) for flags in (0, 1, 2, 3, 4, 8)],
        "setns": [(0, 0), (0, NEWUSER), (0, NEWNET)],
        "socket": [(family, 1, 0) for family in (1, 2, 10, 16, 38, 40, 46, 1 << 32)],
        "socketpair": [(family, 1, 0, 0) for family in (1, 2, 10, 38, 40, 46)],
        "io_uring_setup": [(0, 0)], "bpf": [(0, 0, 0)],
        "open_tree": [(0, 0, 0)], "mount_setattr": [(0, 0, 0, 0, 0)],
    }
    for name, vectors in cases.items():
        bpf = tmp_path / f"{name}.bpf"
        number = _export_bpf(native_libseccomp, name, bpf)
        for vector in vectors:
            action, error = _decision(name, vector)
            expected = 123 if action == ALLOW else error
            arguments = [str(a) for a in (*vector, *([0] * (6 - len(vector))))]
            result = subprocess.run([str(kernel_probe), str(bpf), str(number), *arguments],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == f"-1 {expected}", (name, vector, result.stdout, result.stderr)


@pytest.fixture(scope="module")
def i386_entry_available(kernel_probe) -> bool:
    """Probe the compat entry independently, without the policy or C register state.

    A 64-bit kernel need not provide IA32 emulation. A bare, side-effect-free
    i386 getpid followed by native exit distinguishes that environment from a
    failure of the shipped policy or the compiled C probe being tested.
    """
    if platform.machine() != "x86_64":
        return False
    source = kernel_probe.parent / "ia32-availability.s"
    executable = kernel_probe.parent / "ia32-availability"
    source.write_text('''
.text
.globl _start
_start:
    mov $20, %eax
    int $0x80
    test %eax, %eax
    jle unavailable
    xor %edi, %edi
    jmp finish
unavailable:
    mov $1, %edi
finish:
    mov $60, %eax
    syscall
.section .note.GNU-stack,"",@progbits
''')
    subprocess.run([shutil.which("cc"), "-nostdlib", "-no-pie", str(source), "-o", str(executable)],
                   capture_output=True, text=True, check=True, timeout=30)
    result = subprocess.run(
        ["/bin/sh", "-c", 'ulimit -c 0 || exit 2; exec "$1"', "ia32-probe", str(executable)],
        capture_output=True, text=True, timeout=10, cwd=kernel_probe.parent,
    )
    if result.returncode in (-signal.SIGSEGV, -signal.SIGSYS, 1):
        return False  # kernel lacks IA32 emulation or an outer policy blocks it
    assert result.returncode == 0, (result.returncode, result.stderr)
    return True


def test_compat_probe_return_path_preserves_compiler_state(kernel_probe, tmp_path, i386_entry_available) -> None:
    if not i386_entry_available:
        pytest.skip("independent probe: i386 entry unavailable (IA32 emulation disabled or externally blocked)")
    # Only the independent guard handles this syscall. It returns errno 124,
    # so the assembly MUST return safely rather than accidentally crashing.
    # This is a harness self-test, not a substitute for loading the shipped
    # policy in the separate SIGSYS regression below. No socket is created.
    bpf = tmp_path / "guard-only.bpf"
    bpf.write_bytes(struct.pack("=HBBI", 0x06, 0, 0, 0x7FFF0000))
    result = subprocess.run(
        [str(kernel_probe), str(bpf), "102", "0", "0", "0", "0", "0", "0", "compat"],
        capture_output=True, text=True, timeout=10, cwd=tmp_path,
    )
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert result.stdout.strip() == "-1 124"


@pytest.mark.parametrize("number,mode", [(102, ["compat"]), (0x40000000 | 41, [])], ids=["i386", "x32"])
def test_real_libseccomp_rejects_i386_and_x32_socket_bypasses(native_libseccomp, kernel_probe, tmp_path, i386_entry_available, number, mode) -> None:
    if platform.machine() != "x86_64":
        pytest.skip("i386/x32 syscall entry probes require an x86_64 CPU")
    bpf = tmp_path / "native-only.bpf"
    _export_bpf(native_libseccomp, "socket", bpf)
    # i386 socketcall and x32 socket both used to bypass naive socket-family
    # predicates. The independent guard prevents the call even under a bad
    # filter; merely receiving that guard errno is NOT accepted as a pass.
    if mode and not i386_entry_available:
        pytest.skip("independent probe: i386 entry unavailable (IA32 emulation disabled or externally blocked)")
    # x32 uses the native syscall instruction and is tested even without IA32.
    result = subprocess.run(
        [str(kernel_probe), str(bpf), str(number), "40", "1", "0", "0", "0", "0", *mode],
        capture_output=True, text=True, timeout=10, cwd=tmp_path,
    )
    assert result.returncode == -signal.SIGSYS, (number, result.returncode, result.stdout, result.stderr)


_DOCKER_SMOKE = r'''
import ctypes
import errno
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

from shellsmith.episode.gm_process.codex import CodexWorldConfig, CodexWorldProcess

codex = shutil.which("codex")
assert codex and subprocess.check_output([codex, "--version"], text=True).strip() == "codex-cli 0.159.2"
status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
assert int(status["CapBnd"].strip(), 16) & (1 << 21) == 0, "CAP_SYS_ADMIN must not be granted"
assert int(status["CapEff"].strip(), 16) & (1 << 21) == 0
assert status["NoNewPrivs"].strip() == "1"
assert status["Seccomp"].strip() == "2"
for family in (38, 40):
    try:
        s = socket.socket(family, socket.SOCK_STREAM)
    except OSError as e:
        assert e.errno == errno.EPERM, (family, e)
    else:
        s.close()
        raise AssertionError("outer profile failed to deny socket family")

with tempfile.TemporaryDirectory(prefix="seccomp-codex-") as tmp:
    root = Path(tmp)
    world = root / "world"
    world.mkdir()
    secret = root / "synthetic-auth.json"
    secret.write_text("SYNTHETIC-NOT-A-REAL-CREDENTIAL")
    link = world / "outside-link"
    link.symlink_to(secret)
    process = CodexWorldProcess(CodexWorldConfig(
        executable=Path(codex), model="unused", workspace_dir=world,
        log_dir=root / "logs", bootstrap_response_schema_path=world / "schema.json"))
    # Test the existing compatibility screen AND the actual bundled helper.
    subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--", "/bin/true"],
                   check=True, capture_output=True, text=True, timeout=20)
    for network in (True, False):
        permissions = process._sandbox_args()
        if not network:
            permissions = [a.replace("network={ enabled=true }", "network={ enabled=false }") for a in permissions]
        # Unlike exec, pinned `sandbox` has no --ignore-user-config option.
        # HOME/CODEX_HOME are empty, container-owned test directories.
        base = [codex, *permissions, "sandbox", "-P", "game_master", "-C", str(world), "--"]
        def run(args):
            return subprocess.run(base + args, capture_output=True, text=True, timeout=30)
        good = run(["/bin/sh", "-c", "printf allowed > permitted.txt; cat permitted.txt"])
        assert good.returncode == 0 and good.stdout == "allowed", (good.stdout, good.stderr)
        assert (world / "permitted.txt").read_text() == "allowed"
        for target in (secret, link, Path(f"/proc/{os.getpid()}/root{secret}")):
            denied = run(["/bin/cat", str(target)])
            assert denied.returncode != 0 and "SYNTHETIC-NOT-A-REAL-CREDENTIAL" not in denied.stdout, denied
        forbidden = root / "outside-write"
        denied = run(["/bin/sh", "-c", 'printf forbidden > "$1"', "probe", str(forbidden)])
        assert denied.returncode != 0 and not forbidden.exists(), denied
        # Joining the ancestor network namespace is forbidden even if its fd
        # happens to be reachable. Probe fd 0: policy EPERM, not kernel EBADF.
        nr = 308 if os.uname().machine == "x86_64" else 268
        code = f"import ctypes,errno; l=ctypes.CDLL(None,use_errno=True); assert l.syscall({nr},0,0)==-1 and ctypes.get_errno()==errno.EPERM"
        denied = run(["/usr/bin/python3", "-c", code])
        assert denied.returncode == 0, (denied.stdout, denied.stderr)
print("verified pinned Codex allowed writes and denied filesystem/namespace escapes")
'''


@pytest.mark.parametrize("user", ["0:0", "65534:65534"])
def test_real_docker_pinned_codex_confinement(user: str) -> None:
    image = os.environ.get("SHELLSMITH_SECCOMP_TEST_IMAGE")
    if not image:
        pytest.skip("opt in with SHELLSMITH_SECCOMP_TEST_IMAGE; Linux Docker runtime was unavailable during authoring")
    docker = shutil.which("docker")
    assert docker, "Docker CLI required for explicitly requested integration test"
    name = f"shellsmith-seccomp-test-{uuid.uuid4().hex}"
    command = [docker, "run", "--rm", "-i", "--name", name, "--pull=never", "--user", user,
               "--network=none", "--security-opt", "no-new-privileges=true",
               "--security-opt", f"seccomp={POLICY_PATH.resolve()}",
               "--env", "HOME=/tmp/seccomp-home", "--env", "CODEX_HOME=/tmp/seccomp-codex-home",
               image, "python3", "-"]
    try:
        result = subprocess.run(command, input=_DOCKER_SMOKE, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (
            "Pinned Codex sandbox did not pass; fail closed. docker-default AppArmor deny mount, "
            "LSM/userns restrictions or missing primitives cannot be fixed with seccomp alone.\n"
            + result.stdout + result.stderr
        )
        assert "verified pinned Codex" in result.stdout
    finally:
        # A client timeout is not container termination, even with --rm.
        subprocess.run([docker, "rm", "--force", name], capture_output=True, timeout=30)
