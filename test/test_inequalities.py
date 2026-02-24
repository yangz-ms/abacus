import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc13, calc16
from test_helper import test

if __name__ == '__main__':
    # ================================================================
    # calc13: Linear Inequalities
    # ================================================================

    # Linear inequalities
    test("2*x+3>7", "(2,inf)", calc13)
    test("3-x>=1", "(-inf,2]", calc13)
    test("x<5", "(-inf,5)", calc13)
    test("2*x>=4", "[2,inf)", calc13)
    test("x+1>0", "(-1,inf)", calc13)
    test("5*x<=10", "(-inf,2]", calc13)

    # Absolute value inequalities
    test("abs(x-3)<5", "(-2,8)", calc13)
    test("abs(x)>=2", "(-inf,-2] U [2,inf)", calc13)
    test("abs(x-2)<=5", "[-3,7]", calc13)
    test("abs(x)<3", "(-3,3)", calc13)
    test("abs(x)>0", "(-inf,0) U (0,inf)", calc13)

    # Compound linear inequalities
    test("1<2*x+3<7", "(-1,2)", calc13)
    test("0<=x<=5", "[0,5]", calc13)

    # Constant inequalities
    test("3>2", "(-inf,inf)", calc13)
    test("1>2", "no solution", calc13)

    # Fallthrough to parent calculator (calc12)
    test("2+3", "5", calc13)
    test("x+1", "x+1", calc13)
    test("2*x+3*x", "5*x", calc13)

    # ================================================================
    # calc16: Polynomial Inequalities
    # ================================================================

    # Quadratic inequalities
    test("x^2-4>0", "(-inf,-2) U (2,inf)", calc16)
    test("x^2-4<0", "(-2,2)", calc16)
    test("x^2-4<=0", "[-2,2]", calc16)
    test("x^2-4>=0", "(-inf,-2] U [2,inf)", calc16)
    test("x^2+1<0", "no solution", calc16)
    test("x^2+1>0", "(-inf,inf)", calc16)
    test("x^2-1<=0", "[-1,1]", calc16)

    # Cubic inequalities
    test("x^3-x>0", "(-1,0) U (1,inf)", calc16)
    test("x^3-x<0", "(-inf,-1) U (0,1)", calc16)

    # Fallthrough to parent calculator (calc15)
    test("2+3", "5", calc16)
    test("x+1", "x+1", calc16)
    test("2*x+3*x", "5*x", calc16)
