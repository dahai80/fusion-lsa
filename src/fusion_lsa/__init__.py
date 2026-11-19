import logging
import os
import subprocess

logger = logging.getLogger(__name__)

__version__ = "0.2.1"


def _resolve_commit():
    # Best-effort implementation stamp for run provenance: env override,
    # then the git worktree containing this source tree, else unknown.
    commit = os.environ.get("FUSION_LSA_COMMIT")
    if commit:
        return commit
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(source_dir, ".git")):
        return "unknown-commit"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=source_dir,
        )
        return result.stdout.strip() or "unknown-commit"
    except (OSError, subprocess.SubprocessError):
        return "unknown-commit"


FUSION_LSA_COMMIT = _resolve_commit()
logger.info("fusion-lsa %s (commit %s)", __version__, FUSION_LSA_COMMIT)
