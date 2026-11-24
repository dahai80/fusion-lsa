import json
import logging

logger = logging.getLogger(__name__)


def read_manifest(path, fields=None, encoding="utf-8"):
    # Faithful subset of nemo's read_manifest: JSONL in, list of dicts out.
    # Malformed lines are skipped with a warning instead of raising, matching
    # the upstream behavior the RULER scripts rely on.
    manifest = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed manifest line in %s", path)
                continue
            if not isinstance(sample, dict):
                logger.warning("skipping non-dict manifest entry in %s", path)
                continue
            if fields is not None:
                sample = {key: sample[key] for key in fields if key in sample}
            manifest.append(sample)
    return manifest


def write_manifest(path, target, fields=None, encoding="utf-8"):
    with open(path, "w", encoding=encoding) as f:
        for sample in target:
            if fields is not None:
                sample = {key: sample[key] for key in fields if key in sample}
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
