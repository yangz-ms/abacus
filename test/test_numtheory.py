import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc6, calc7, calc8, calc9, calc10, calc11
from test_helper import test

if __name__ == '__main__':
    # ================================================================
    # calc6: GCD, LCM, Primes, Modulo
    # ================================================================
    test("gcd(12,8)", "4", calc6)
    test("lcm(4,6)", "12", calc6)
    test("17%3", "2", calc6)
    test("factor(60)", "2^2*3*5", calc6)
    test("floor(3.7)", "3", calc6)
    test("ceil(3.2)", "4", calc6)
    test("round(3.456)", "3", calc6)
    test("isprime(7)", "1", calc6)
    test("isprime(8)", "0", calc6)
    test("gcd(12,8)+lcm(4,6)", "16", calc6)
    # calc6 inherits calc5 (exponents and sqrt)
    test("sqrt(4)", "2", calc6)
    test("2^10", "1024", calc6)
    test("1+2", "3", calc6)

    # ================================================================
    # calc7: Pi, e & Scientific Notation
    # ================================================================
    test("pi", "3.141592653589793", calc7)
    test("e", "2.718281828459045", calc7)
    test("2*pi", "6.283185307179586", calc7)
    test("e^2", "7.3890560989306495", calc7)
    test("1e2", "100", calc7)
    test("1.5e3", "1500", calc7)
    test("2.5e-3", "0.0025", calc7)
    test("1e2*pi", "314.1592653589793", calc7)
    # calc7 inherits calc6 (number theory)
    test("gcd(12,8)", "4", calc7)
    test("lcm(4,6)", "12", calc7)
    test("factor(60)", "2^2*3*5", calc7)
    test("1+2", "3", calc7)

    # ================================================================
    # calc8: Basic Trig + Degree Mode
    # ================================================================
    test("sin(90d)", "1", calc8)
    test("cos(180d)", "-1", calc8)
    test("tan(45d)", "1", calc8)
    test("sin(0d)", "0", calc8)
    test("cos(360d)", "1", calc8)
    test("sin(pi/2)", "1", calc8)
    test("cos(0)", "1", calc8)
    test("tan(pi/4)", "1", calc8)
    test("sin(0)", "0", calc8)
    # calc8 inherits calc7 (pi, e, sci notation)
    test("pi", "3.141592653589793", calc8)
    test("e", "2.718281828459045", calc8)
    test("1e2", "100", calc8)
    # calc8 inherits calc6 (number theory)
    test("gcd(12,8)", "4", calc8)
    test("1+2", "3", calc8)

    # ================================================================
    # calc9: Exp & Log Functions
    # ================================================================
    test("exp(1)", "2.718281828459045", calc9)
    test("ln(1)", "0", calc9)
    test("log(100)", "2", calc9)
    test("logb(2,8)", "3", calc9)
    test("logb(10,1000)", "3", calc9)
    test("logb(2,1024)", "10", calc9)
    # calc9 inherits calc8 (basic trig)
    test("sin(90d)", "1", calc9)
    test("cos(180d)", "-1", calc9)
    test("tan(45d)", "1", calc9)
    # calc9 inherits calc7 (pi, e)
    test("pi", "3.141592653589793", calc9)
    # calc9 inherits calc6 (number theory)
    test("gcd(12,8)", "4", calc9)
    test("1+2", "3", calc9)

    # ================================================================
    # calc10: Factorial & Combinatorics
    # ================================================================
    test("5!", "120", calc10)
    test("3!+4!", "30", calc10)
    test("C(10,3)", "120", calc10)
    test("P(5,2)", "20", calc10)
    test("C(52,5)", "2598960", calc10)
    test("C(10,3)+P(5,2)", "140", calc10)
    test("C(0,0)", "1", calc10)
    test("P(0,0)", "1", calc10)
    test("C(5,5)", "1", calc10)
    test("P(5,0)", "1", calc10)
    # calc10 inherits calc9 (exp & log)
    test("exp(1)", "2.718281828459045", calc10)
    test("ln(1)", "0", calc10)
    test("log(100)", "2", calc10)
    # calc10 inherits calc8 (trig)
    test("sin(90d)", "1", calc10)
    test("cos(180d)", "-1", calc10)
    # calc10 inherits calc6 (number theory)
    test("gcd(12,8)", "4", calc10)
    test("1+2", "3", calc10)

    # ================================================================
    # calc11: Complex Numbers
    # ================================================================
    test("i", "i", calc11)
    test("i*i", "-1", calc11)
    test("i^2", "-1", calc11)
    test("1+i", "1+i", calc11)
    test("1-i", "1-i", calc11)
    test("(1+i)*2", "2+2i", calc11)
    test("(1+i)*(1-i)", "2", calc11)
    test("(1+i)/(1-i)", "i", calc11)
    test("2+3*i", "2+3i", calc11)
    test("e^(i*pi)", "-1", calc11)
    test("abs(3+4*i)", "5", calc11)
    # calc11 inherits calc10 (factorial & combinatorics)
    test("5!", "120", calc11)
    test("C(10,3)", "120", calc11)
    test("P(5,2)", "20", calc11)
    # calc11 inherits calc9 (exp & log)
    test("exp(1)", "2.718281828459045", calc11)
    test("ln(1)", "0", calc11)
    # calc11 inherits calc8 (trig)
    test("sin(90d)", "1", calc11)
    test("cos(180d)", "-1", calc11)
    # calc11 inherits calc7 (pi, e)
    test("pi", "3.141592653589793", calc11)
    test("e", "2.718281828459045", calc11)
    # calc11 inherits calc6 (number theory)
    test("gcd(12,8)", "4", calc11)
    test("1+2", "3", calc11)
