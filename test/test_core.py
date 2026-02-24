import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc1, calc2, calc3, calc4, calc5
from test_helper import test

if __name__ == '__main__':
    # calc1: add and subtract
    test("1+2+3", "6", calc1)
    test("123+456 - 789", "-210", calc1)
    test("123-456", "-333", calc1)

    # calc2: add multiply and divide
    test("1+2+3", "6", calc2)
    test("123+456 - 789", "-210", calc2)
    test("123-456", "-333", calc2)
    test("1*2*3", "6", calc2)
    test("123+456*789", "359907", calc2)
    test("1+2*3-4", "3", calc2)
    test("1+2*3-5/4", "23/4", calc2)
    test("1*2*3*4*5/6", "20", calc2)

    # calc3: decimal arithmetic (recursive descent parser, NO exponents)
    test("1+2+3", "6", calc3)
    test("123+456 - 789", "-210", calc3)
    test("123-456", "-333", calc3)
    test("1*2*3", "6", calc3)
    test("123+456*789", "359907", calc3)
    test("1+2*3-4", "3", calc3)
    test("1+2*3-5/4", "23/4", calc3)
    test("1*2*3*4*5/6", "20", calc3)
    test("1+2*(3-4)", "-1", calc3)
    test("1.5+2.3", "3.8", calc3)
    test("3.14*2", "6.28", calc3)
    test("10/3", "10/3", calc3)
    test("1.5*2", "3", calc3)
    test("", "", calc3, Exception())

    # calc4: parentheses and exponents (adds ^ to calc3)
    test("1+2+3", "6", calc4)
    test("1+2*3-4", "3", calc4)
    test("1+2*(3-4)", "-1", calc4)
    test("(3^5+2)/(7*7)", "5", calc4)
    test("2^10", "1024", calc4)
    test("2^0", "1", calc4)
    test("1.5+2.3", "3.8", calc4)
    test("3.14*2", "6.28", calc4)
    test("1**2", "", calc4, Exception())
    test("", "", calc4, Exception())

    # calc5: exponents and sqrt
    test("sqrt(4)", "2", calc5)
    test("sqrt(9)", "3", calc5)
    test("2^10+sqrt(9)", "1027", calc5)
    test("sqrt(2)", "1.4142135623730951", calc5)
    test("sqrt(0-1)", "i", calc5)
    test("1+2", "3", calc5)
    test("2^10", "1024", calc5)
    test("1.5+2.3", "3.8", calc5)
    test("foo", "", calc5, Exception())
