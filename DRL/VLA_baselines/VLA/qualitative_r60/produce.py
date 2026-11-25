"""Reuse the audited sequential renderer with a NEW output root only."""
from types import SimpleNamespace
from VLA.qualitative_r58 import produce
from .prepare import OUT

if __name__=='__main__':
    produce.OUT=OUT
    produce.main(SimpleNamespace(output=OUT/'production_v1'))
