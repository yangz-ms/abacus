# Abacus
Simple calculator built incrementally, from basic arithmetic to symbolic algebra.

## Project Structure

The calculator is organized as a Python package under `calc/`, split into submodules along the inheritance chain. Each calculator auto-registers via `@register` in `calc/registry.py`, so adding a new level requires no changes to the UI or app.py.

```
calc/
  __init__.py        Re-exports all public names + get_registry()
  registry.py        @register decorator and REGISTRY dict
  core.py            calc1-7, Calculator3-7, format_complex, _format_result
  helpers.py         _to_int, _fmt_num, _cbrt, _clean_root, _format_solution
  numtheory.py       Calculator8-10, calc8-10, _prime_factors, _isprime
  matrix.py          Matrix, Calculator11, calc11, _clean_float
  algebra.py         Polynomial, Calculator12, calc12, _unwrap helpers
  systems.py         MultiPolynomial, Calculator13, calc13, solve_linear_system
  polytools.py       Radical, factor_polynomial, poly_divide, Calculator14, calc14
  inequalities.py    Interval, solve_inequality, Calculator15, calc15
```

Tests mirror the calc/ structure:

```
test/
  test_core.py           calc1-7 tests
  test_numtheory.py      calc8-10 tests
  test_matrix.py         calc11 tests
  test_algebra.py        calc12 tests
  test_systems.py        calc13 tests
  test_polytools.py      calc14 tests (Radical, factoring, division, etc.)
  test_inequalities.py   calc15 tests (inequalities)
  test_inheritance.py    cross-calculator inheritance, edge cases, performance
  test_backward_compat.py backward compatibility matrix (1140 checks)
```

## Versions

| Version | Description |
|---------|-------------|
| calc1   | Add and subtract |
| calc2   | Multiply and divide with operator precedence |
| calc3   | Recursive descent parser with parentheses and exponents |
| calc4   | Scientific notation and decimals |
| calc5   | Named constants (pi, e) |
| calc6   | Imaginary unit and complex numbers |
| calc7   | Math functions (sin, cos, tan, exp, ln, log, sqrt, abs, etc.) |
| calc8   | Number theory: gcd, lcm, factorial, modulo, prime factorization, rounding |
| calc9   | Combinatorics: permutations (P) and combinations (C) |
| calc10  | Extended trig (degree mode, sec/csc/cot), arbitrary-base log, polar/rect |
| calc11  | Matrices and vectors: arithmetic, determinant, inverse, dot/cross product |
| calc12  | Single-variable algebra and equation solving (linear, quadratic, cubic) |
| calc13  | Multi-variable linear equations (up to 3 variables) |
| calc14  | Polynomial tools: factoring, long division, completing the square, binomial expansion |
| calc15  | Inequalities: linear, quadratic, absolute value, compound |

## Examples

```
calc1("1+2+3")                       -> "6"
calc2("1+2*3-4")                     -> "3"
calc3("1+2*(3-4)")                   -> "-1"
calc4("1.5e3*2")                     -> "3000.0"
calc5("2*pi")                        -> "6.283185307179586"
calc6("e^(i*pi)")                    -> "-1"
calc7("sin(pi/2)")                   -> "1"

calc8("5!")                          -> "120"
calc8("gcd(12,8)")                   -> "4"
calc8("factor(60)")                  -> "2^2*3*5"
calc9("C(10,3)")                     -> "120"
calc10("sin(90d)")                   -> "1"
calc10("logb(2,1024)")               -> "10"

calc11("det([[1,2],[3,4]])")         -> "-2"
calc11("cross([1,0,0],[0,1,0])")     -> "[0,0,1]"

calc12("(x+1)*(x-1)")               -> "x^2-1"
calc12("x^2-5*x+6=0")              -> "x=2; x=3"

calc13("x+y=2; x-y=0")             -> "x=1; y=1"
calc13("x+y+z=6; x-y=0; x+z=4")   -> "x=2; y=2; z=2"

calc14("factor(x^3-6*x^2+11*x-6)") -> "(x-1)*(x-2)*(x-3)"
calc14("complsq(x^2+6*x+5)")       -> "(x+3)^2-4"
calc14("binom(x+1,3)")             -> "x^3+3*x^2+3*x+1"

calc15("x^2-4>0")                   -> "(-inf,-2) U (2,inf)"
calc15("abs(x-3)<5")                -> "(-2,8)"
```

## Web UI

A web interface with LaTeX-rendered input/output, powered by FastAPI and KaTeX.

```
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8000.

## Run tests

```
python test/test_core.py
python test/test_numtheory.py
python test/test_matrix.py
python test/test_algebra.py
python test/test_systems.py
python test/test_polytools.py
python test/test_inequalities.py
python test/test_inheritance.py
python test/test_backward_compat.py
python tex_converter.py
```
