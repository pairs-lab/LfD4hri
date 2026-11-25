"""Frozen Oracle/X-VLA put-only qualitative collection, not SR training."""
import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType

# The actual quantitative SIM interpreter has the PINNED PyBullet build but
# no obsolete Gym. env_eval/__init__.py only registers unrelated Gym IDs.
# Bypass ONLY that registration side effect in this process; import all live
# Oracle/render/executor modules normally. Never substitute a different
# PyBullet build simply to make an unused Gym import succeed.
if 'env_eval' not in sys.modules and importlib.util.find_spec('gym') is None:
    package=ModuleType('env_eval')
    package.__path__=[str(Path(__file__).resolve().parents[2]/'env_eval')]
    package.__package__='env_eval'
    package.__spec__=ModuleSpec('env_eval',loader=None,is_package=True)
    sys.modules['env_eval']=package
