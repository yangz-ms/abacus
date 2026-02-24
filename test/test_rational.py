import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc17
from calc.rational import RationalExpression, poly_gcd, Calculator17
from calc.algebra import Polynomial
from test_helper import test

if __name__ == '__main__':
    # =======================================================================
    # poly_gcd tests
    # =======================================================================
    print("--- poly_gcd ---")

    # GCD of (x^2-1) and (x+1) should be (x+1)
    a = Polynomial([-1, 0, 1])   # x^2 - 1
    b = Polynomial([1, 1])       # x + 1
    g = poly_gcd(a, b)
    assert str(g) == "x+1", f"Expected x+1, got {g}"
    print(f"  PASS gcd(x^2-1, x+1) = {g}")

    # GCD of (x^2-1) and (x-1) should be (x-1)
    b2 = Polynomial([-1, 1])     # x - 1
    g2 = poly_gcd(a, b2)
    assert str(g2) == "x-1", f"Expected x-1, got {g2}"
    print(f"  PASS gcd(x^2-1, x-1) = {g2}")

    # GCD of two coprime polynomials: (x+1) and (x-1)
    g3 = poly_gcd(Polynomial([1, 1]), Polynomial([-1, 1]))
    assert g3.degree() == 0, f"Expected degree 0, got {g3.degree()}"
    print(f"  PASS gcd(x+1, x-1) = {g3} (coprime)")

    # GCD of (x^3-x) and (x^2-1): both factor through (x^2-1)
    g4 = poly_gcd(Polynomial([0, -1, 0, 1]), Polynomial([-1, 0, 1]))
    assert g4.degree() == 2, f"Expected degree 2, got {g4.degree()}"
    print(f"  PASS gcd(x^3-x, x^2-1) = {g4}")

    # GCD of a polynomial with itself
    p = Polynomial([1, 2, 1])  # x^2+2x+1 = (x+1)^2
    g5 = poly_gcd(p, p)
    assert str(g5) == "x^2+2*x+1", f"Expected x^2+2*x+1, got {g5}"
    print(f"  PASS gcd(x^2+2x+1, x^2+2x+1) = {g5}")

    # GCD of zero and a polynomial
    g6 = poly_gcd(Polynomial([0]), Polynomial([1, 1]))
    assert str(g6) == "x+1", f"Expected x+1, got {g6}"
    print(f"  PASS gcd(0, x+1) = {g6}")

    # =======================================================================
    # RationalExpression class tests
    # =======================================================================
    print("\n--- RationalExpression class ---")

    # Basic construction and str
    r1 = RationalExpression(Polynomial([1, 1]), Polynomial([-1, 1]))
    print(f"  (x+1)/(x-1) = {r1}")
    assert str(r1) == "(x+1)/(x-1)", f"Expected (x+1)/(x-1), got {r1}"
    print("  PASS basic construction")

    # Simplification
    r2 = RationalExpression(Polynomial([-1, 0, 1]), Polynomial([1, 1]))
    r2s = r2.simplify()
    assert str(r2s) == "x-1", f"Expected x-1, got {r2s}"
    print(f"  PASS simplify (x^2-1)/(x+1) = {r2s}")

    # Addition
    ra = RationalExpression(Polynomial([1]), Polynomial([1, 1]))   # 1/(x+1)
    rb = RationalExpression(Polynomial([1]), Polynomial([-1, 1]))  # 1/(x-1)
    rc = ra + rb
    assert str(rc) == "(2*x)/(x^2-1)", f"Expected (2*x)/(x^2-1), got {rc}"
    print(f"  PASS 1/(x+1) + 1/(x-1) = {rc}")

    # Subtraction
    rd = ra - rb
    assert str(rd) == "-2/(x^2-1)", f"Expected -2/(x^2-1), got {rd}"
    print(f"  PASS 1/(x+1) - 1/(x-1) = {rd}")

    # Multiplication
    re = RationalExpression(Polynomial([0, 1]), Polynomial([1, 1]))   # x/(x+1)
    rf = RationalExpression(Polynomial([1, 1]), Polynomial([-1, 1]))  # (x+1)/(x-1)
    rg = re * rf
    assert str(rg) == "(x)/(x-1)", f"Expected (x)/(x-1), got {rg}"
    print(f"  PASS x/(x+1) * (x+1)/(x-1) = {rg}")

    # Division
    rh = re / rf
    assert "(x^2-x)" in str(rh) or "(x)" in str(rh), f"Unexpected: {rh}"
    print(f"  PASS x/(x+1) / ((x+1)/(x-1)) = {rh}")

    # Simplify to 1
    ri = RationalExpression(Polynomial([1, 1]), Polynomial([1, 1]))
    ris = ri.simplify()
    assert str(ris) == "1", f"Expected 1, got {ris}"
    print(f"  PASS simplify (x+1)/(x+1) = {ris}")

    # Denominator is 1
    rj = RationalExpression(Polynomial([1, 2, 1]), Polynomial([1]))
    assert str(rj) == "x^2+2*x+1", f"Expected x^2+2*x+1, got {rj}"
    print(f"  PASS display with denominator 1: {rj}")

    # =======================================================================
    # calc17 entry function tests - simplify
    # =======================================================================
    print("\n--- calc17 simplify ---")

    test("simplify(x^2-1,x+1)", "x-1", calc17)
    test("simplify(x^2-1,x-1)", "x+1", calc17)
    test("simplify(x^2+2*x+1,x+1)", "x+1", calc17)
    test("simplify(x^3-x,x^2-1)", "x", calc17)
    test("simplify(x^2-1,x^2-1)", "1", calc17)
    test("simplify(2*x+2,2)", "x+1", calc17)
    test("simplify(4*x^2-4,2*x+2)", "2*x-2", calc17)

    # =======================================================================
    # calc17 entry function tests - radd
    # =======================================================================
    print("\n--- calc17 radd ---")

    test("radd(1,x+1,1,x-1)", "(2*x)/(x^2-1)", calc17)
    test("radd(x,1,1,x)", "(x^2+1)/(x)", calc17)
    test("radd(1,x,1,x)", "2/(x)", calc17)

    # =======================================================================
    # calc17 entry function tests - rsub
    # =======================================================================
    print("\n--- calc17 rsub ---")

    test("rsub(1,x+1,1,x-1)", "-2/(x^2-1)", calc17)
    test("rsub(1,x-1,1,x+1)", "2/(x^2-1)", calc17)
    test("rsub(1,x,1,x)", "0", calc17)

    # =======================================================================
    # calc17 entry function tests - rmul
    # =======================================================================
    print("\n--- calc17 rmul ---")

    test("rmul(x,x+1,x+1,x-1)", "(x)/(x-1)", calc17)
    test("rmul(x+1,x-1,x-1,x+1)", "1", calc17)
    test("rmul(x,1,1,x)", "1", calc17)

    # =======================================================================
    # calc17 entry function tests - rdiv
    # =======================================================================
    print("\n--- calc17 rdiv ---")

    test("rdiv(x,x+1,x,x-1)", "(x-1)/(x+1)", calc17)
    test("rdiv(1,x+1,1,x-1)", "(x-1)/(x+1)", calc17)
    test("rdiv(x,1,1,x)", "x^2", calc17)

    # =======================================================================
    # Passthrough to calc16 and below
    # =======================================================================
    print("\n--- calc17 passthrough ---")

    test("2+3", "5", calc17)
    test("x^2-5*x+6=0", "x=2; x=3", calc17)
    test("(x+1)^2", "x^2+2*x+1", calc17)
    test("factor(x^2-1)", "(x-1)*(x+1)", calc17)
    test("x^2-4<0", "(-2,2)", calc17)

    # =======================================================================
    # Calculator17 inheritance
    # =======================================================================
    print("\n--- Calculator17 inheritance ---")

    from calc.polyineq import Calculator16
    assert issubclass(Calculator17, Calculator16), "Calculator17 should extend Calculator16"
    print("  PASS Calculator17 extends Calculator16")

    # =======================================================================
    # Registry check
    # =======================================================================
    print("\n--- Registry ---")

    from calc.registry import REGISTRY
    assert "calc17" in REGISTRY, "calc17 should be registered"
    entry = REGISTRY["calc17"]
    assert entry["short_desc"] == "Rational Expressions"
    assert entry["group"] == "solver"
    assert "simplify(x^2-1,x+1)" in entry["examples"]
    assert "zh" in entry["i18n"]
    assert entry["i18n"]["zh"] == "\u6709\u7406\u8868\u8fbe\u5f0f"
    print("  PASS calc17 registered with correct metadata")

    # =======================================================================
    # Error handling
    # =======================================================================
    print("\n--- Error handling ---")

    test("simplify(x,0)", None, calc17, exception=Exception())
    test("simplify(x)", None, calc17, exception=Exception())
    test("radd(1,x+1,1)", None, calc17, exception=Exception())
    test("rmul(1,x+1)", None, calc17, exception=Exception())

    print("\nAll calc17 tests passed!")
