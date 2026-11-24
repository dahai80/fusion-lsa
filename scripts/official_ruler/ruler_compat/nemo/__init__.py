# Compat shim: minimal stand-in for the nemo toolkit package. The pinned
# RULER scripts import only nemo.collections.asr.parts.utils.manifest_utils,
# which is a thin JSONL reader/writer; installing the full nemo_toolkit
# distribution just for that is disproportionate. PYTHONPATH points the
# pinned scripts at this package instead (see run_official_ruler.py).
