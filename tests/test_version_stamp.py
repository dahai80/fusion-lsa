import importlib
import logging
import tomllib
from pathlib import Path

import fusion_lsa

logger = logging.getLogger(__name__)


def test_version_stamp_present():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert fusion_lsa.__version__ == declared
    commit = fusion_lsa.FUSION_LSA_COMMIT
    assert commit, "FUSION_LSA_COMMIT must be a non-empty string"
    assert commit == "unknown-commit" or all(c in "0123456789abcdef" for c in commit)


def test_import_logs_banner(caplog):
    # Serve logs must identify the implementation a run was produced with:
    # the import-time banner carries version + commit (issue #5 acceptance).
    with caplog.at_level(logging.INFO, logger="fusion_lsa"):
        importlib.reload(fusion_lsa)
    records = [r for r in caplog.records if r.name == "fusion_lsa"]
    assert records, "import did not log the version banner"
    assert fusion_lsa.__version__ in records[-1].getMessage()
    assert fusion_lsa.FUSION_LSA_COMMIT in records[-1].getMessage()
