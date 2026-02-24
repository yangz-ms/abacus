#!/usr/bin/env python
"""Backward compatibility tests: verify each calcN passes all earlier test cases.

KEY RULE: calcN must produce the same result as calcM for all M < N test cases.
For example, calc15("1+2+3") must return "6" (same as calc1).

This test suite:
1. Defines all test cases grouped by the level they were designed for.
2. For each calculator level N (2..15), runs ALL test cases from levels 1..N-1.
3. Computes the canonical expected value by running each expression through its
   native calculator, so we don't hard-code potentially stale expected values.
4. Reports pass/fail with clear error messages and a summary matrix.
"""

import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from calc import (
    calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9,
    calc10, calc11, calc12, calc13, calc14, calc15,
)

# ---------------------------------------------------------------------------
# Calculator functions indexed by level
# ---------------------------------------------------------------------------
CALC_FUNCTIONS = {
    1: calc1,   2: calc2,   3: calc3,   4: calc4,   5: calc5,
    6: calc6,   7: calc7,   8: calc8,   9: calc9,   10: calc10,
    11: calc11, 12: calc12, 13: calc13, 14: calc14, 15: calc15,
}

# ---------------------------------------------------------------------------
# Test cases grouped by the level they were DESIGNED for.
# Only normal-return test cases are included (exception tests are skipped).
# Format: list of expression strings.
# The canonical expected value is computed at runtime by calling the native
# calculator for that level.
# ---------------------------------------------------------------------------
TESTS = {
    # ---- calc1: addition and subtraction ----
    1: [
        "1+2+3",
        "123+456 - 789",
        "123-456",
    ],

    # ---- calc2: multiplication and division with precedence ----
    2: [
        "1+2+3",
        "123+456 - 789",
        "123-456",
        "1*2*3",
        "123+456*789",
        "1+2*3-4",
        "1+2*3-5/4",
        "1*2*3*4*5/6",
    ],

    # ---- calc3: parentheses and exponents (recursive descent) ----
    3: [
        "1+2+3",
        "123+456 - 789",
        "123-456",
        "1*2*3",
        "123+456*789",
        "1+2*3-4",
        "1+2*3-5/4",
        "1*2*3*4*5/6",
        "1+2*(3-4)",
        "(3^5+2)/(7*7)",
    ],

    # ---- calc4: scientific notation and decimals ----
    4: [
        "1+2+3",
        "1+2*3-4",
        "1e2",
        "1.5e3",
        "2.5e-3",
        "1.5E+3",
        "1e2+1.5e2",
        "1.5e3*2",
        "(1e2+1.5e2)*2e1",
        "1.5e3/1.5e2",
    ],

    # ---- calc5: named constants (pi, e) ----
    5: [
        "pi",
        "e",
        "2*pi",
        "pi+e",
        "e^2",
        "1e2*pi",
        "1+2",
    ],

    # ---- calc6: imaginary unit and complex numbers ----
    6: [
        "i",
        "i*i",
        "i^2",
        "1+i",
        "1-i",
        "(1+i)*2",
        "(1+i)*(1-i)",
        "(1+i)/(1-i)",
        "2+3*i",
        "e^(i*pi)",
        "1+2",
    ],

    # ---- calc7: math functions (sin, cos, tan, exp, ln, log, sqrt, abs, ...) ----
    7: [
        "sin(0)",
        "cos(0)",
        "sin(pi/2)",
        "tan(pi/4)",
        "sinh(0)",
        "cosh(0)",
        "exp(1)",
        "ln(1)",
        "log(100)",
        "sqrt(4)",
        "sqrt(0-1)",
        "abs(3+4*i)",
        "e^(i*pi)",
        "1+2",
    ],

    # ---- calc8: number theory (gcd, lcm, factorial, modulo, factor, etc.) ----
    8: [
        "gcd(12,8)",
        "lcm(4,6)",
        "5!",
        "17%3",
        "factor(60)",
        "floor(3.7)",
        "ceil(3.2)",
        "round(3.456)",
        "isprime(7)",
        "isprime(8)",
        "gcd(12,8)+lcm(4,6)",
        "3!+4!",
    ],

    # ---- calc9: combinatorics (C, P) ----
    9: [
        "C(10,3)",
        "P(5,2)",
        "C(52,5)",
        "C(10,3)+P(5,2)",
        "C(0,0)",
        "P(0,0)",
        "C(5,5)",
        "P(5,0)",
        # calc9 inherits calc8
        "5!",
        "gcd(12,8)",
    ],

    # ---- calc10: extended trig and logs (degree mode, sec, csc, cot, logb) ----
    10: [
        "sin(90d)",
        "cos(180d)",
        "tan(45d)",
        "sin(0d)",
        "cos(360d)",
        "sec(0)",
        "csc(pi/2)",
        "cot(pi/4)",
        "logb(2,8)",
        "logb(10,1000)",
        "logb(2,1024)",
        # calc10 inherits calc9
        "C(10,3)",
        "P(5,2)",
        "5!",
    ],

    # ---- calc11: matrix operations ----
    11: [
        "det([[1,2],[3,4]])",
        "det([[1,0,0],[0,1,0],[0,0,1]])",
        "trace([[1,2],[3,4]])",
        "dot([1,2,3],[4,5,6])",
        "cross([1,0,0],[0,1,0])",
        "[[1,2],[3,4]]+[[5,6],[7,8]]",
        "[[1,2],[3,4]]*[[1,0],[0,1]]",
        "[[1,2],[3,4]]*[[5,6],[7,8]]",
        "[[5,6],[7,8]]-[[1,2],[3,4]]",
        "inv([[2,1],[1,1]])",
        "2*[[1,2],[3,4]]",
        "[[2,4],[6,8]]/2",
        "trans([[1,2,3],[4,5,6]])",
        "rref([[1,2,3],[4,5,6]])",
        "[[1,1],[0,1]]^3",
    ],

    # ---- calc12: symbolic simplification and equation solving ----
    12: [
        # Simplification
        "x",
        "2*x+3*x",
        "x*x",
        "2+3",
        "x+1-1",
        "(x+1)*(x-1)",
        "3*x^2+2*x+1",
        # Linear equations
        "2*x=4",
        "x+1=3",
        "3*x+2=x+10",
        "x=5",
        "2*(x+1)=6",
        # More simplification
        "0*x",
        "1*x",
        "x+x+x",
        "x^2+x^2",
        "(x+1)^2",
        "x*0",
        "x-x",
        "x^3",
        "x*x*x",
        "(x+1)*(x+1)",
        "2*x*3",
        "x^2-x^2",
        "(x+1)*(x+2)",
        # More linear equations
        "5*x=0",
        "x/2=3",
        "10-x=3",
        "x+x=8",
        "3*x-1=2*x+4",
        # Quadratic equations
        "x^2=1",
        "x^2+2*x+1=0",
        "x^2-5*x+6=0",
        "x^2=4",
        "x^2-1=0",
        "x^2+1=0",
        "2*x^2-8=0",
        "x^2-3*x=0",
        "x^2-4*x+4=0",
        "x^2+4*x+4=0",
        "x^2-2*x-3=0",
        # Cubic equations
        "x^3-6*x^2+11*x-6=0",
        "x^3-1=0",
        "x^3=0",
        "x^3-3*x^2+3*x-1=0",
        "x^3+x^2-x-1=0",
        "x^3=8",
        "x^3+1=0",
        "x^3-x=0",
    ],

    # ---- calc13: multi-variable simplification and linear systems ----
    13: [
        # Single-variable (same as calc12)
        "2*x+3*x",
        "2*x=4",
        "x^2=1",
        "x^3-6*x^2+11*x-6=0",
        "2+3",
        # Multi-variable simplification
        "x+y+x",
        "3*x+2*y-x",
        "x+y-x-y",
        "2*x+3*y+x-y",
        # Two-variable linear systems
        "x+y=2; x-y=0",
        "2*x+3*y=7; x-y=1",
        "x+y=10; 2*x+y=15",
        "x+2*y=5; 3*x-y=1",
        "x=3; x+y=5",
        # Three-variable linear systems
        "x+y+z=6; x-y=0; x+z=4",
        "x+y+z=3; x+y-z=1; x-y+z=1",
        "x+y+z=10; x-y+z=4; x+y-z=2",
        "2*x+y-z=1; x+y+z=6; x-y+2*z=5",
    ],

    # ---- calc14: polynomial tools (factor, divpoly, complsq, binom, higher-degree eqs) ----
    14: [
        # Polynomial factoring
        "factor(x^2-5*x+6)",
        "factor(x^2-1)",
        "factor(x^3-6*x^2+11*x-6)",
        "factor(x^3-1)",
        "factor(2*x^2-8)",
        "factor(x^2+1)",
        "factor(x^4-1)",
        "factor(x^2-2*x+1)",
        # Polynomial division
        "divpoly(x^3-1,x-1)",
        "divpoly(x^2+3*x+2,x+1)",
        "divpoly(x^3+2*x+1,x+1)",
        # Completing the square
        "complsq(x^2+6*x+5)",
        "complsq(x^2-4*x+1)",
        "complsq(x^2+2*x+1)",
        "complsq(2*x^2+4*x+1)",
        # Binomial expansion
        "binom(x+1,3)",
        "binom(x+2,5)",
        "binom(x-1,4)",
        "binom(x+1,0)",
        "binom(x+1,1)",
        # Higher-degree equation solving
        "x^4-1=0",
        "x^4-5*x^2+4=0",
        # Pass-through
        "(x+1)^2",
        "2*x+3*x",
        "x^2-5*x+6=0",
        "x^3-6*x^2+11*x-6=0",
    ],

    # ---- calc15: inequalities ----
    15: [
        # Linear inequalities
        "2*x+3>7",
        "3-x>=1",
        "x<5",
        "2*x>=4",
        "x+1>0",
        "5*x<=10",
        # Quadratic inequalities
        "x^2-4>0",
        "x^2-4<0",
        "x^2-4<=0",
        "x^2-4>=0",
        "x^2+1<0",
        "x^2+1>0",
        "x^2-1<=0",
        # Cubic inequalities
        "x^3-x>0",
        "x^3-x<0",
        # Absolute value inequalities
        "abs(x-3)<5",
        "abs(x)>=2",
        "abs(x-2)<=5",
        "abs(x)<3",
        "abs(x)>0",
        # Compound inequalities
        "1<2*x+3<7",
        "0<=x<=5",
        # Constant inequalities
        "3>2",
        "1>2",
        # Fallthrough to parent calculator
        "2+3",
        "x+1",
        "2*x+3*x",
    ],
}


def compute_canonical_values():
    """Run each test case through its native calculator to get the canonical expected value.

    Returns a dict: {level: [(expr, canonical_result), ...]}
    Entries where the native calculator raises an exception are silently skipped.
    """
    canonical = {}
    for level, expressions in TESTS.items():
        calc_fn = CALC_FUNCTIONS[level]
        pairs = []
        for expr in expressions:
            try:
                result = calc_fn(expr)
                pairs.append((expr, result))
            except Exception:
                # Skip expressions that even the native calculator cannot handle
                pass
        canonical[level] = pairs
    return canonical


def run_backward_compat():
    """Run backward compatibility checks for every calculator level.

    For each level N (2..15), run ALL test cases from levels 1..N-1 and verify
    that calcN returns the same result as the native calculator for that level.
    """
    print("=" * 72)
    print("  BACKWARD COMPATIBILITY TEST SUITE")
    print("=" * 72)
    print()
    print("Computing canonical values from native calculators...")
    canonical = compute_canonical_values()

    # Count total test cases per native level
    for lvl in sorted(canonical):
        print(f"  Level {lvl:2d}: {len(canonical[lvl])} test case(s)")
    print()

    total_pass = 0
    total_fail = 0
    total_error = 0
    failures = []

    # Matrix: rows = calc level being tested, cols = source test level
    # Values: (pass_count, fail_count, error_count)
    matrix = {}

    print("-" * 72)
    print("  Running backward compatibility checks...")
    print("-" * 72)

    for level in range(2, 16):
        calc_fn = CALC_FUNCTIONS[level]
        level_pass = 0
        level_fail = 0
        level_error = 0
        matrix[level] = {}

        for test_level in range(1, level):
            if test_level not in canonical:
                continue
            src_pass = 0
            src_fail = 0
            src_error = 0

            for expr, expected in canonical[test_level]:
                try:
                    result = calc_fn(expr)
                    if result == expected:
                        src_pass += 1
                    else:
                        src_fail += 1
                        failures.append(
                            f"  calc{level} <- level {test_level}: "
                            f"{expr!r} expected {expected!r}, got {result!r}"
                        )
                except Exception as exc:
                    src_error += 1
                    failures.append(
                        f"  calc{level} <- level {test_level}: "
                        f"{expr!r} raised {type(exc).__name__}: {exc}"
                    )

            matrix[level][test_level] = (src_pass, src_fail, src_error)
            level_pass += src_pass
            level_fail += src_fail
            level_error += src_error

        status = "PASS" if (level_fail + level_error) == 0 else "FAIL"
        total_tested = level_pass + level_fail + level_error
        print(
            f"  calc{level:2d}: {level_pass:3d} pass, {level_fail:3d} fail, "
            f"{level_error:3d} error  (of {total_tested:3d})  [{status}]"
        )
        total_pass += level_pass
        total_fail += level_fail
        total_error += level_error

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"  TOTAL: {total_pass} pass, {total_fail} fail, {total_error} error")
    print("=" * 72)

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f)

    # -------------------------------------------------------------------
    # Compatibility matrix
    # -------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  COMPATIBILITY MATRIX")
    print("  Rows = calculator under test, Columns = source test level")
    print("  . = all pass  X = failure(s)  E = error(s)  - = not applicable")
    print("=" * 72)

    # Header row
    header = "         "
    for src in range(1, 16):
        header += f" {src:>3d}"
    print(header)
    print("         " + " ----" * 15)

    for level in range(1, 16):
        row = f"  calc{level:2d} |"
        for src in range(1, 16):
            if src >= level:
                row += "   -"
            elif level in matrix and src in matrix[level]:
                p, f_count, e = matrix[level][src]
                if e > 0:
                    row += "   E"
                elif f_count > 0:
                    row += "   X"
                else:
                    row += "   ."
            else:
                row += "   -"
        print(row)

    print()
    return total_fail + total_error


if __name__ == "__main__":
    fails = run_backward_compat()
    sys.exit(1 if fails > 0 else 0)
