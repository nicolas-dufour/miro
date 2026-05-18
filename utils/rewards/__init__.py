"""Reward-model backbones used during dataset preprocessing.

Vendored from the legacy ``cad.utils`` so miro has no runtime dependency on
``cad/``. These are only loaded by ``miro/data/preprocess_data.py``; the
inference path (``MiroPipeline``) and post-hoc evaluation
(``miro/eval/rewards``) both have their own scorer implementations.
"""
