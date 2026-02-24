import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc8, calc9, calc10
from test_helper import test

if __name__ == '__main__':
    # calc8: number theory
    test("gcd(12,8)", "4", calc8)
    test("lcm(4,6)", "12", calc8)
    test("5!", "120", calc8)
    test("17%3", "2", calc8)
    test("factor(60)", "2^2*3*5", calc8)
    test("floor(3.7)", "3", calc8)
    test("ceil(3.2)", "4", calc8)
    test("round(3.456)", "3", calc8)
    test("isprime(7)", "1", calc8)
    test("isprime(8)", "0", calc8)
    test("gcd(12,8)+lcm(4,6)", "16", calc8)
    test("3!+4!", "30", calc8)

    # calc9: combinatorics
    test("C(10,3)", "120", calc9)
    test("P(5,2)", "20", calc9)
    test("C(52,5)", "2598960", calc9)
    test("C(10,3)+P(5,2)", "140", calc9)
    test("C(0,0)", "1", calc9)
    test("P(0,0)", "1", calc9)
    test("C(5,5)", "1", calc9)
    test("P(5,0)", "1", calc9)
    # calc9 inherits calc8
    test("5!", "120", calc9)
    test("gcd(12,8)", "4", calc9)

    # calc10: extended trig and logs
    test("sin(90d)", "1", calc10)
    test("cos(180d)", "-1", calc10)
    test("tan(45d)", "1", calc10)
    test("sin(0d)", "0", calc10)
    test("cos(360d)", "1", calc10)
    test("sec(0)", "1", calc10)
    test("csc(pi/2)", "1", calc10)
    test("cot(pi/4)", "1", calc10)
    test("logb(2,8)", "3", calc10)
    test("logb(10,1000)", "3", calc10)
    test("logb(2,1024)", "10", calc10)
    # calc10 inherits calc9
    test("C(10,3)", "120", calc10)
    test("P(5,2)", "20", calc10)
    test("5!", "120", calc10)
