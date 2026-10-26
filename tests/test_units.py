import pytest

from aimd_mlff_methods.units import AMU_TO_EV_FS2_PER_A2


def test_amu_to_ev_fs2_reference():
    assert AMU_TO_EV_FS2_PER_A2 == pytest.approx(103.642691909, abs=1e-6)
