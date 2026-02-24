import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc14, Radical, factor_polynomial, poly_divide, complete_square, binom_expand, Polynomial
from test_helper import test

if __name__ == '__main__':
    # --- Radical class ---
    r = Radical(0, 1, 48)  # sqrt(48)
    assert str(r) == "4*sqrt(3)", f"Expected 4*sqrt(3), got {str(r)}"
    print(f"  Radical: sqrt(48) = {r}")

    r2 = Radical(3, 2, 5)
    assert str(r2) == "3+2*sqrt(5)"

    r3 = Radical(0, 1, 1)
    assert str(r3) == "1"

    # Radical arithmetic
    a = Radical(1, 2, 3)
    b = Radical(2, 3, 3)
    assert str(a + b) == "3+5*sqrt(3)"
    assert str(a * b) == "20+7*sqrt(3)"
    assert str(a - b) == "-1-sqrt(3)"
    assert Radical(0, 1, 2) / Radical(0, 1, 2) == 1
    assert float(Radical(1, 1, 4)) == 3.0  # 1 + 1*sqrt(4) = 1+2 = 3
    print("  Radical: all arithmetic tests passed")

    # --- Polynomial factoring ---
    test("factor(x^2-5*x+6)", "(x-2)*(x-3)", calc14)
    test("factor(x^2-1)", "(x-1)*(x+1)", calc14)
    test("factor(x^3-6*x^2+11*x-6)", "(x-1)*(x-2)*(x-3)", calc14)
    test("factor(x^3-1)", "(x-1)*(x^2+x+1)", calc14)
    test("factor(2*x^2-8)", "2*(x-2)*(x+2)", calc14)
    test("factor(x^2+1)", "x^2+1", calc14)
    test("factor(x^4-1)", "(x-1)*(x+1)*(x^2+1)", calc14)
    test("factor(x^2-2*x+1)", "(x-1)*(x-1)", calc14)

    # --- Polynomial division ---
    test("divpoly(x^3-1,x-1)", "x^2+x+1", calc14)
    test("divpoly(x^2+3*x+2,x+1)", "x+2", calc14)
    test("divpoly(x^3+2*x+1,x+1)", "x^2-x+3 R -2", calc14)

    # --- Completing the square ---
    test("complsq(x^2+6*x+5)", "(x+3)^2-4", calc14)
    test("complsq(x^2-4*x+1)", "(x-2)^2-3", calc14)
    test("complsq(x^2+2*x+1)", "(x+1)^2", calc14)
    test("complsq(2*x^2+4*x+1)", "2*(x+1)^2-1", calc14)

    # --- Binomial expansion ---
    test("binom(x+1,3)", "x^3+3*x^2+3*x+1", calc14)
    test("binom(x+2,5)", "x^5+10*x^4+40*x^3+80*x^2+80*x+32", calc14)
    test("binom(x-1,4)", "x^4-4*x^3+6*x^2-4*x+1", calc14)
    test("binom(x+1,0)", "1", calc14)
    test("binom(x+1,1)", "x+1", calc14)

    # --- Higher-degree equation solving ---
    test("x^4-1=0", "x=-1; x=1; x=-i; x=i", calc14)
    test("x^4-5*x^2+4=0", "x=-2; x=-1; x=1; x=2", calc14)

    # --- Pass-through (simplification, lower-degree equations) ---
    test("(x+1)^2", "x^2+2*x+1", calc14)
    test("2*x+3*x", "5*x", calc14)
    test("x^2-5*x+6=0", "x=2; x=3", calc14)
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc14)
