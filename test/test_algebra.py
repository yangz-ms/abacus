import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc12
from test_helper import test

if __name__ == '__main__':
    # calc12: Simplification tests
    test("x", "x", calc12)
    test("2*x+3*x", "5*x", calc12)
    test("x*x", "x^2", calc12)
    test("2+3", "5", calc12)
    test("x+1-1", "x", calc12)
    test("(x+1)*(x-1)", "x^2-1", calc12)
    test("3*x^2+2*x+1", "3*x^2+2*x+1", calc12)

    # calc12: Linear equation tests
    test("2*x=4", "x=2", calc12)
    test("x+1=3", "x=2", calc12)
    test("3*x+2=x+10", "x=4", calc12)
    test("x=5", "x=5", calc12)
    test("2*(x+1)=6", "x=2", calc12)

    # calc12: More simplification tests
    test("0*x", "0", calc12)
    test("1*x", "x", calc12)
    test("x+x+x", "3*x", calc12)
    test("x^2+x^2", "2*x^2", calc12)
    test("(x+1)^2", "x^2+2*x+1", calc12)
    test("x*0", "0", calc12)
    test("x-x", "0", calc12)
    test("x^3", "x^3", calc12)
    test("x*x*x", "x^3", calc12)
    test("(x+1)*(x+1)", "x^2+2*x+1", calc12)
    test("2*x*3", "6*x", calc12)
    test("x^2-x^2", "0", calc12)
    test("(x+1)*(x+2)", "x^2+3*x+2", calc12)

    # calc12: More linear equation tests
    test("5*x=0", "x=0", calc12)
    test("x/2=3", "x=6", calc12)
    test("10-x=3", "x=7", calc12)
    test("x+x=8", "x=4", calc12)
    test("3*x-1=2*x+4", "x=5", calc12)

    # calc12: Quadratic equation tests
    test("x^2=1", "x=-1; x=1", calc12)
    test("x^2+2*x+1=0", "x=-1", calc12)
    test("x^2-5*x+6=0", "x=2; x=3", calc12)
    test("x^2=4", "x=-2; x=2", calc12)
    test("x^2-1=0", "x=-1; x=1", calc12)
    test("x^2+1=0", "x=-i; x=i", calc12)
    test("2*x^2-8=0", "x=-2; x=2", calc12)
    test("x^2-3*x=0", "x=0; x=3", calc12)
    test("x^2-4*x+4=0", "x=2", calc12)
    test("x^2+4*x+4=0", "x=-2", calc12)
    test("x^2-2*x-3=0", "x=-1; x=3", calc12)

    # calc12: Cubic equation tests
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc12)
    test("x^3-1=0", "x=1; x=-0.5-0.866025403784439i; x=-0.5+0.866025403784439i", calc12)
    test("x^3=0", "x=0", calc12)
    test("x^3-3*x^2+3*x-1=0", "x=1", calc12)
    test("x^3+x^2-x-1=0", "x=-1; x=1", calc12)
    test("x^3=8", "x=2; x=-1-1.732050807568877i; x=-1+1.732050807568877i", calc12)
    test("x^3+1=0", "x=-1; x=0.5-0.866025403784439i; x=0.5+0.866025403784439i", calc12)
    test("x^3-x=0", "x=-1; x=0; x=1", calc12)
