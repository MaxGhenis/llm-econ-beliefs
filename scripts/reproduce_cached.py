"""Rebuild the retained manuscript analysis using only stdlib and cached inputs.

--check requires a clean checkout, reads a pinned commit into a temporary source
archive (no .git), deletes every declared output there, and rebuilds them with
Python network access disabled. It compares bytes with the original committed
outputs, verifies evidence hashes and final commit/source identity, and leaves the
caller's checkout alone.
--write updates outputs in the current checkout for a deliberate, reviewable edit.
Neither mode refreshes the evidence manifest, calls models, or runs PolicyEngine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_manifest import GENERATED_RESULTS, verify_manifest  # noqa: E402

# Fixed output inventory, not a glob of whatever the build happened to leave.
TABLE_STEMS = (
    "armington-clarify-delta",
    "benchmark-comparison-labor-tax",
    "cap-gains-convention-audit",
    "correlates-country",
    "correlates-model-summary",
    "flat-tax-demogrant-appendix",
    "harness-disclosure",
    "ies-clarify-delta",
    "leave-one-organization-out-appendix",
    "leave-one-provider-out-appendix",
    "mechanism-ablation",
    "model-overview-labor-tax",
    "model-overview-macro-trade",
    "model-overview-simulation",
    "policybench-correlates",
    "pooling-robustness-appendix",
    "quantile-rule-appendix",
    "quantity-disagreement-simulation",
    "quantity-disagreement",
    "resampling-stability",
    "stability-appendix",
    "support-bounds",
    "tool-use-appendix",
    "top-rate-robustness",
    "toy-top-rate-labor-tax",
    "variance-decomposition",
    "wording-comparison-tau",
    "wording-comparison",
)
OUTPUTS = (
    *(f"results/{name}" for name in GENERATED_RESULTS),
    *(f"paper/tables/{stem}.{ext}" for stem in TABLE_STEMS for ext in ("csv", "md")),
    "CITATION.cff",
    "release-metadata.json",
)
# Citation contains authored metadata; regenerate only its abstract from a seed.
SEED_OUTPUTS = {"CITATION.cff"}
STEPS = (
    ("scripts/check_panel_grid.py", "--require-parsed"),
    ("scripts/verify_cached_summaries.py",),
    ("scripts/build_model_registry.py", "--no-stage-dashboard"),
    ("scripts/build_comparison_artifacts.py",),
    ("scripts/build_correlates.py",),
    ("paper/build_tables.py",),
    ("scripts/build_release_metadata.py",),
    ("scripts/verify_paper_prose.py",),
)
OFFLINE_RUNNER = """
import runpy, sys
from pathlib import Path
def offline(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
        raise RuntimeError('Cached reproduction forbids network access and child processes: ' + event)
sys.addaudithook(offline)
script = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parents[1]))
runpy.run_path(str(script), run_name='__main__')
"""


def require_clean_tree(root: Path) -> None:
    # Git status deliberately suppresses changes behind these index flags. A
    # working-copy replay cannot claim committed bytes while either flag is set.
    index = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-v", "-z"],
        text=True,
    )
    hidden = [
        entry[2:]
        for entry in index.split("\0")
        if entry and (entry[0].islower() or entry[0] == "S")
    ]
    if hidden:
        raise ValueError(
            "--check requires a clean checkout without assume-unchanged or "
            "skip-worktree index flags: " + ", ".join(hidden)
        )
    status = subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    if status:
        raise ValueError(
            "--check requires a clean checkout (including staged and untracked files):\n"
            + status
        )


def compare_outputs(root: Path, expected: dict[str, bytes]) -> None:
    failures = []
    for path, data in expected.items():
        target = root / path
        if not target.is_file():
            failures.append(f"missing: {path}")
        elif target.read_bytes() != data:
            failures.append(f"changed: {path}")
    if failures:
        raise ValueError(
            "Cached reproduction differs from committed outputs:\n"
            + "\n".join(failures)
        )


def rebuild(root: Path) -> None:
    verify_manifest(root)
    for step in STEPS:
        print("Cached step: " + " ".join(step), flush=True)
        subprocess.run(
            [sys.executable, "-I", "-S", "-c", OFFLINE_RUNNER, *step],
            cwd=root,
            check=True,
        )
    verify_manifest(root)
    missing = [path for path in OUTPUTS if not (root / path).is_file()]
    if missing:
        raise ValueError("Builder omitted required outputs: " + ", ".join(missing))


def head_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        text=True,
    ).strip()


def materialize_commit(root: Path, commit: str, target: Path) -> dict[str, bytes]:
    """Read source and expected bytes from a pinned commit, never the working tree."""
    committed = {}
    # Use a file-backed archive to avoid holding a second copy of all raw evidence
    # in memory. Extract regular files ourselves; reject links and path escapes.
    with tempfile.TemporaryFile() as stream:
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar", commit],
            stdout=stream,
            check=True,
        )
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            for member in archive:
                if member.isdir():
                    continue
                relative = Path(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise ValueError(f"Unsupported source archive entry: {member.name}")
                data = archive.extractfile(member).read()
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                committed[member.name] = data
    tracked = set(
        subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", commit],
            text=True,
        )
        .strip("\0")
        .split("\0")
    )
    if set(committed) != tracked:
        raise ValueError(
            "Source archive must include every tracked file (no export-ignore omissions)"
        )
    return committed


def verify_checkout_commit(
    root: Path, commit: str, committed: dict[str, bytes]
) -> None:
    """Detect a changed HEAD or tracked bytes even when Git's status cache misses it."""
    require_clean_tree(root)
    if head_commit(root) != commit:
        raise ValueError("Checkout HEAD changed during cached reproduction")
    compare_outputs(root, committed)


def check(root: Path = ROOT) -> None:
    require_clean_tree(root)
    commit = head_commit(root)
    verify_manifest(root)
    with tempfile.TemporaryDirectory(prefix="llm-econ-cached-") as temporary:
        clean = Path(temporary)
        before = materialize_commit(root, commit, clean)
        verify_checkout_commit(root, commit, before)
        expected = {path: before[path] for path in OUTPUTS}
        for path in OUTPUTS:
            if path not in SEED_OUTPUTS:
                (clean / path).unlink()
        rebuild(clean)
        compare_outputs(clean, expected)
        # Detect any unexpected output or mutation beyond the declared products.
        after = {
            p.relative_to(clean).as_posix(): p.read_bytes()
            for p in clean.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        if before != after:
            changed = sorted(
                p for p in before.keys() | after.keys() if before.get(p) != after.get(p)
            )
            raise ValueError(
                "Reproduction changed the source archive: " + ", ".join(changed)
            )
        verify_checkout_commit(root, commit, before)
    print(
        f"PASS: {len(OUTPUTS)} outputs reproduced byte-for-byte from commit {commit} "
        "in a source archive without Git history; checkout remains clean"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        check() if args.check else rebuild(ROOT)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
