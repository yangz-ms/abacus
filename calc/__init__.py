"""Abacus calculator package -- re-exports all public names."""
from calc.registry import REGISTRY

from calc.core import *
from calc.helpers import *
from calc.numtheory import *
from calc.algebra import *
from calc.inequalities import *
from calc.systems import *
from calc.polytools import *
from calc.polyineq import *
from calc.rational import *
from calc.radical import *
from calc.matrix import *
from calc.tex import *


def get_registry():
    """Return the ordered dict of all registered calculators."""
    return REGISTRY
