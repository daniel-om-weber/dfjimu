__version__ = "0.1.0"
_CYTHON_AVAILABLE = False

try:
    from ._cython.core import run_mekf_cython
    from ._cython.optimizer import build_system_cython, update_lin_points_cython
    from ._cython.lever_arms import estimate_lever_arms_cython
    _CYTHON_AVAILABLE = True
except ImportError:
    from ._python.core import run_mekf_cython
    from ._python.optimizer import build_system_cython, update_lin_points_cython
    from ._python.lever_arms import estimate_lever_arms_cython

from .mekf_acc import mekf_acc, MekfAcc
from .map_acc import map_acc, MapAcc
from .lever_arms import estimate_lever_arms

__all__ = ['mekf_acc', 'MekfAcc', 'map_acc', 'MapAcc', 'estimate_lever_arms', '_CYTHON_AVAILABLE']
