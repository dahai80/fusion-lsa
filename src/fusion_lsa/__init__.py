import logging
import os
import subprocess

logger = logging.getLogger(__name__)

__version__ = "0.2.5"


def _resolve_commit():
    # Best-effort implementation stamp for run provenance: env override,
    # then the git worktree containing this source tree, else unknown.
    # Walks up from src/fusion_lsa/ to the repo root: under pip install -e
    # the package dir itself holds no .git.
    commit = os.environ.get("FUSION_LSA_COMMIT")
    if commit:
        return commit
    probe = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(probe, ".git")):
            break
        parent = os.path.dirname(probe)
        if parent == probe:
            return "unknown-commit"
        probe = parent
    else:
        return "unknown-commit"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=probe,
        )
        return result.stdout.strip() or "unknown-commit"
    except (OSError, subprocess.SubprocessError):
        return "unknown-commit"


FUSION_LSA_COMMIT = _resolve_commit()
logger.info("fusion-lsa %s (commit %s)", __version__, FUSION_LSA_COMMIT)
