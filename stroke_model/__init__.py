"""Reference-conditioned V2 stroke extraction and structural evaluation."""
from .utils import set_seed, get_device, ensure_dir, load_config


def __getattr__(name):
    if name == "ReferenceStrokeModel":
        from .reference_model import ReferenceStrokeModel
        return ReferenceStrokeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
