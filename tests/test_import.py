"""Lightweight smoke tests that do not touch the network or the disk."""
import inspect

import miro


def test_version_is_set():
    assert isinstance(miro.__version__, str)
    assert miro.__version__.count(".") == 2


def test_pipeline_export():
    from miro import MiroPipeline

    assert "MiroPipeline" in miro.__all__
    sig = inspect.signature(MiroPipeline.from_pretrained)
    # The public entry point must accept a repo id / local path and an optional variant.
    assert "repo_id_or_path" in sig.parameters
    assert "variant" in sig.parameters


def test_coherence_keys():
    from miro import MiroPipeline

    assert len(MiroPipeline.COHERENCE_KEYS) == 7
    assert "clip_score" in MiroPipeline.COHERENCE_KEYS
    assert "aesthetic_score" in MiroPipeline.COHERENCE_KEYS
