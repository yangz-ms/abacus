import math
import cmath

from calc.core import format_complex


def _to_int(value):
    if isinstance(value, complex):
        if value.imag == 0:
            value = value.real
        else:
            return value
    if isinstance(value, float) and math.isfinite(value):
        rounded = round(value)
        if abs(value - rounded) < 1e-12:
            return rounded
        if value == int(value):
            return int(value)
    return value

def _fmt_num(x):
    x = _to_int(x)
    if isinstance(x, complex):
        return format_complex(x)
    if isinstance(x, float):
        if math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)
    return str(x)

def _cbrt(z):
    z = complex(z)
    if z == 0:
        return complex(0)
    r = abs(z)
    theta = cmath.phase(z)
    return (r ** (1.0 / 3.0)) * cmath.exp(complex(0, theta / 3.0))

def _clean_root(r, eps=1e-9):
    if isinstance(r, complex):
        real = r.real
        imag = r.imag
        if abs(real) < eps:
            real = 0.0
        if abs(imag) < eps:
            imag = 0.0
        if abs(real - round(real)) < eps:
            real = round(real)
        if abs(imag - round(imag)) < eps:
            imag = round(imag)
        if imag == 0:
            return _to_int(real)
        return _to_int(complex(real, imag))
    else:
        if abs(r) < eps:
            return 0
        if abs(r - round(r)) < eps:
            return int(round(r))
        return _to_int(r)

def _unique_roots(roots, eps=1e-9):
    unique = []
    for r in roots:
        is_dup = False
        for u in unique:
            if isinstance(r, complex) or isinstance(u, complex):
                rc = complex(r) if not isinstance(r, complex) else r
                uc = complex(u) if not isinstance(u, complex) else u
                if abs(rc - uc) < eps:
                    is_dup = True
                    break
            else:
                if abs(r - u) < eps:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(r)
    return unique

def _sort_roots(roots):
    real_roots = []
    complex_roots = []
    for r in roots:
        if isinstance(r, complex):
            complex_roots.append(r)
        else:
            real_roots.append(r)
    real_roots.sort()
    complex_roots.sort(key=lambda v: (v.real, v.imag))
    return real_roots + complex_roots

def _format_solution(value):
    value = _to_int(value)
    if isinstance(value, complex):
        return format_complex(value)
    if isinstance(value, float):
        if math.isfinite(value) and value == int(value):
            return str(int(value))
        return str(value)
    return str(value)
