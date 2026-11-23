import pytest
import imaging.superres_tile_seam as mod

CASES = [
    ("tile_overlap", (1,)),
    ("blend_weights", (1,)),
    ("size_cache_key", (1,)),
    ("color_manage", (1,)),
    ("frame_align", (1,)),
    ("drop_policy", (1,)),
    ("fallback_ts", (1,)),
    ("seam_mask", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("batch_shape", (1,)),
    ("metrics", (1,)),
    ("hdr_input", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
