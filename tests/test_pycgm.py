import pytest
import numpy as np
from pycgm import CGM_Processor
from pycgm.calibration import FcropParameters, retrieve_first_order
from pathlib import Path
import json
try:
    import cupy as cp
    has_gpu = True
except ImportError:
    cp = None
    has_gpu = False

@pytest.fixture
def example_dir():
    return Path(__file__).parents[1] / 'examples'

@pytest.fixture
def sample_reference(example_dir):
    return np.load(example_dir / 'sample_reference.npy')

@pytest.fixture
def sample_interferogram(example_dir):
    return np.load(example_dir / 'sample_interferogram.npy')

@pytest.fixture
def sample_phase_map(example_dir):
    return np.load(example_dir / 'sample_phase_map.npy')

@pytest.fixture
def valid_first_orders(example_dir):
    with open(example_dir / 'first_order_positions.json') as f:
        return json.load(f)


def test_cpu(sample_reference, sample_interferogram, sample_phase_map):
    cgm = CGM_Processor(ref=sample_reference, gpu=False)
    assert cgm.xp == np

    out = cgm.process(sample_interferogram)
    assert np.allclose(out, sample_phase_map)

def test_gpu(sample_reference, sample_interferogram, sample_phase_map):
    if not has_gpu:
        pytest.skip("No GPU available")
    cgm = CGM_Processor(ref=sample_reference, gpu=True)
    assert cgm.xp == cp

    out = cgm.process(sample_interferogram)
    assert np.allclose(out, sample_phase_map)

def test_retrieve_first_order(sample_reference, valid_first_orders):
    fref = np.fft.fftshift(np.fft.fft2(sample_reference))
    crops = retrieve_first_order(np.abs(fref), gpu=False)
    assert [crops.Nx, crops.Ny] == valid_first_orders['reference_size']
    assert [crops.x, crops.y] in valid_first_orders['valid_positions']
    assert abs(crops.R - valid_first_orders['radius']) < 1
    
def test_circ_mask():
    """Test circular mask creation."""
    params = FcropParameters(256, 256, 30, 512, 512)
    mask = params.circ_mask
    
    # Check mask properties
    assert mask.shape == (512, 512)
    assert mask.dtype == bool
    
    # Center of mask should be True
    assert mask[256, 256]
    
    # Points far from center should be False
    assert not mask[0, 0]
    assert not mask[511, 511]

def test_rotate90():
    """Test 90-degree rotation functionality."""
    params = FcropParameters(400, 256, 30, 512, 512)
    rotated = params.rotate90()
    
    assert abs(rotated.x-256) < 5
    assert abs(rotated.y-400) < 5
    assert rotated.R == params.R
    assert rotated.Nx == params.Nx
    assert rotated.Ny == params.Ny