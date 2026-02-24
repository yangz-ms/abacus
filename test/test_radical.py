import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc18, Calculator18, Calculator16
from calc.polytools import Radical
from test_helper import test

if __name__ == '__main__':
    # ================================================================
    # Calculator18 class hierarchy
    # ================================================================
    assert issubclass(Calculator18, Calculator16), "Calculator18 should extend Calculator16"
    print("  PASS Calculator18 extends Calculator16")

    # ================================================================
    # simplifyrad: simplify sqrt(n)
    # ================================================================
    test("simplifyrad(50)", "5*sqrt(2)", calc18)
    test("simplifyrad(48)", "4*sqrt(3)", calc18)
    test("simplifyrad(4)", "2", calc18)           # perfect square
    test("simplifyrad(7)", "sqrt(7)", calc18)      # already square-free
    test("simplifyrad(1)", "1", calc18)            # sqrt(1) = 1
    test("simplifyrad(72)", "6*sqrt(2)", calc18)
    test("simplifyrad(100)", "10", calc18)         # perfect square
    test("simplifyrad(12)", "2*sqrt(3)", calc18)
    test("simplifyrad(18)", "3*sqrt(2)", calc18)
    test("simplifyrad(75)", "5*sqrt(3)", calc18)
    test("simplifyrad(200)", "10*sqrt(2)", calc18)
    test("simplifyrad(9)", "3", calc18)            # perfect square
    test("simplifyrad(2)", "sqrt(2)", calc18)      # prime, square-free
    test("simplifyrad(98)", "7*sqrt(2)", calc18)

    # ================================================================
    # rationalize: a / (b * sqrt(n))
    # ================================================================
    test("rationalize(1,1,2)", "(1/2)*sqrt(2)", calc18)     # 1/sqrt(2)
    test("rationalize(2,1,3)", "(2/3)*sqrt(3)", calc18)     # 2/sqrt(3)
    test("rationalize(1,2,5)", "(1/10)*sqrt(5)", calc18)    # 1/(2*sqrt(5))
    test("rationalize(6,3,2)", "sqrt(2)", calc18)           # 6/(3*sqrt(2)) = 2/sqrt(2) = sqrt(2)
    test("rationalize(3,1,3)", "sqrt(3)", calc18)           # 3/sqrt(3) = sqrt(3)
    test("rationalize(4,2,2)", "sqrt(2)", calc18)           # 4/(2*sqrt(2)) = 2/sqrt(2) = sqrt(2)
    test("rationalize(10,5,2)", "sqrt(2)", calc18)          # 10/(5*sqrt(2)) = 2/sqrt(2) = sqrt(2)

    # ================================================================
    # addrad: (a1 + b1*sqrt(n1)) + (a2 + b2*sqrt(n2))
    # ================================================================
    test("addrad(0,1,2,0,1,8)", "3*sqrt(2)", calc18)       # sqrt(2)+sqrt(8) = sqrt(2)+2*sqrt(2)
    test("addrad(1,2,3,2,3,3)", "3+5*sqrt(3)", calc18)
    test("addrad(0,1,12,0,1,27)", "5*sqrt(3)", calc18)     # 2*sqrt(3)+3*sqrt(3)
    test("addrad(3,0,0,0,2,5)", "3+2*sqrt(5)", calc18)     # rational + radical
    test("addrad(0,2,5,0,3,5)", "5*sqrt(5)", calc18)       # 2*sqrt(5)+3*sqrt(5)
    test("addrad(5,0,0,3,0,0)", "8", calc18)               # two rationals: 5+3
    test("addrad(0,1,18,0,1,2)", "4*sqrt(2)", calc18)      # sqrt(18)+sqrt(2) = 3*sqrt(2)+sqrt(2)
    test("addrad(0,1,50,0,1,32)", "9*sqrt(2)", calc18)     # 5*sqrt(2)+4*sqrt(2)

    # ================================================================
    # mulrad: (a1 + b1*sqrt(n1)) * (a2 + b2*sqrt(n2))
    # ================================================================
    test("mulrad(1,1,2,1,-1,2)", "-1", calc18)             # conjugate: (1+sqrt(2))*(1-sqrt(2)) = -1
    test("mulrad(0,1,3,0,1,3)", "3", calc18)               # sqrt(3)*sqrt(3) = 3
    test("mulrad(1,2,3,2,3,3)", "20+7*sqrt(3)", calc18)
    test("mulrad(0,2,5,0,3,5)", "30", calc18)              # 2*sqrt(5)*3*sqrt(5) = 30
    test("mulrad(2,0,0,0,3,7)", "6*sqrt(7)", calc18)       # 2 * 3*sqrt(7)
    test("mulrad(0,1,2,0,1,2)", "2", calc18)               # sqrt(2)*sqrt(2) = 2
    test("mulrad(3,1,5,3,-1,5)", "4", calc18)              # (3+sqrt(5))*(3-sqrt(5)) = 9-5 = 4
    test("mulrad(1,1,3,1,1,3)", "4+2*sqrt(3)", calc18)     # (1+sqrt(3))^2 = 1+2*sqrt(3)+3

    # ================================================================
    # Error cases
    # ================================================================

    # addrad with incompatible radicands (after simplification)
    test("addrad(0,1,2,0,1,3)", None, calc18,
         exception=ValueError("Cannot add radicals with different radicands"))

    # mulrad with incompatible radicands
    test("mulrad(0,1,2,0,1,3)", None, calc18,
         exception=ValueError("Cannot multiply radicals with different radicands"))

    # rationalize with b=0
    test("rationalize(1,0,2)", None, calc18, exception=Exception("Denominator coefficient b cannot be zero"))

    # simplifyrad with negative
    test("simplifyrad(-4)", None, calc18, exception=Exception("Cannot simplify square root of a negative number"))

    # Wrong number of args
    test("rationalize(1,2)", None, calc18,
         exception=Exception("rationalize requires exactly 3 arguments: a, b, n"))
    test("addrad(1,2,3)", None, calc18,
         exception=Exception("addrad requires exactly 6 arguments: a1, b1, n1, a2, b2, n2"))
    test("mulrad(1,2,3)", None, calc18,
         exception=Exception("mulrad requires exactly 6 arguments: a1, b1, n1, a2, b2, n2"))

    # ================================================================
    # Fallthrough to calc16 and below
    # ================================================================
    test("2+3", "5", calc18)
    test("x^2-4<0", "(-2,2)", calc18)
    test("factor(x^2-1)", "(x-1)*(x+1)", calc18)
    test("x^2-5*x+6=0", "x=2; x=3", calc18)
    test("2*x+3*x", "5*x", calc18)

    # ================================================================
    # Registry check
    # ================================================================
    from calc.registry import REGISTRY
    assert "calc18" in REGISTRY, "calc18 should be registered"
    meta = REGISTRY["calc18"]
    assert meta["short_desc"] == "Radical Expressions"
    assert meta["group"] == "solver"
    assert "simplifyrad(50)" in meta["examples"]
    assert "zh" in meta["i18n"]
    print("  PASS calc18 registry metadata is correct")
