import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc14
from test_helper import test

if __name__ == '__main__':
    # calc14: single-variable (same as calc12)
    test("2*x+3*x", "5*x", calc14)
    test("2*x=4", "x=2", calc14)
    test("x^2=1", "x=-1; x=1", calc14)
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc14)
    test("2+3", "5", calc14)

    # calc14: multi-variable simplification
    test("x+y+x", "2*x+y", calc14)
    test("3*x+2*y-x", "2*x+2*y", calc14)
    test("x+y-x-y", "0", calc14)
    test("2*x+3*y+x-y", "3*x+2*y", calc14)

    # calc14: two-variable linear systems
    test("x+y=2; x-y=0", "x=1; y=1", calc14)
    test("2*x+3*y=7; x-y=1", "x=2; y=1", calc14)
    test("x+y=10; 2*x+y=15", "x=5; y=5", calc14)
    test("x+2*y=5; 3*x-y=1", "x=1; y=2", calc14)
    test("x=3; x+y=5", "x=3; y=2", calc14)

    # calc14: three-variable linear systems
    test("x+y+z=6; x-y=0; x+z=4", "x=2; y=2; z=2", calc14)
    test("x+y+z=3; x+y-z=1; x-y+z=1", "x=1; y=1; z=1", calc14)
    test("x+y+z=10; x-y+z=4; x+y-z=2", "x=3; y=3; z=4", calc14)
    test("2*x+y-z=1; x+y+z=6; x-y+2*z=5", "x=1; y=2; z=3", calc14)
