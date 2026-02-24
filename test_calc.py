from calc import calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9, calc10, calc11, calc12, calc13
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
    test("1+2*3-5/4", "5.75", calc2)
    test("1*2*3*4*5/6", "20", calc2)

    # calc3: recursive descent parser with parentheses and exponents
    test("1+2+3", "6", calc3)
    test("123+456 - 789", "-210", calc3)
    test("123-456", "-333", calc3)
    test("1*2*3", "6", calc3)
    test("123+456*789", "359907", calc3)
    test("1+2*3-4", "3", calc3)
    test("1+2*3-5/4", "5.75", calc3)
    test("1*2*3*4*5/6", "20", calc3)
    test("1+2*(3-4)", "-1", calc3)
    test("(3^5+2)/(7*7)", "5", calc3)
    test("1**2", "", calc3, Exception())
    test("", "", calc3, Exception())

    # calc4: scientific notation and decimals
    test("1+2+3", "6", calc4)
    test("1+2*3-4", "3", calc4)
    test("1e2", "100", calc4)
    test("1.5e3", "1500", calc4)
    test("2.5e-3", "0.0025", calc4)
    test("1.5E+3", "1500", calc4)
    test("1e2+1.5e2", "250", calc4)
    test("1.5e3*2", "3000", calc4)
    test("(1e2+1.5e2)*2e1", "5000", calc4)
    test("1.5e3/1.5e2", "10", calc4)
    test("", "", calc4, Exception())

    # calc5: named constants (pi, e)
    test("pi", "3.141592653589793", calc5)
    test("e", "2.718281828459045", calc5)
    test("2*pi", "6.283185307179586", calc5)
    test("pi+e", "5.859874482048838", calc5)
    test("e^2", "7.3890560989306495", calc5)
    test("1e2*pi", "314.1592653589793", calc5)
    test("1+2", "3", calc5)
    test("foo", "", calc5, Exception())

    # calc6: imaginary unit and complex numbers
    test("i", "i", calc6)
    test("i*i", "-1", calc6)
    test("i^2", "-1", calc6)
    test("1+i", "1+i", calc6)
    test("1-i", "1-i", calc6)
    test("(1+i)*2", "2+2i", calc6)
    test("(1+i)*(1-i)", "2", calc6)
    test("(1+i)/(1-i)", "i", calc6)
    test("2+3*i", "2+3i", calc6)
    test("e^(i*pi)", "-1", calc6)
    test("1+2", "3", calc6)

    # calc7: math functions (sin, cos, tan, exp, ln, log, sqrt, abs, etc.)
    test("sin(0)", "0", calc7)
    test("cos(0)", "1", calc7)
    test("sin(pi/2)", "1", calc7)
    test("tan(pi/4)", "1", calc7)
    test("sinh(0)", "0", calc7)
    test("cosh(0)", "1", calc7)
    test("exp(1)", "2.718281828459045", calc7)
    test("ln(1)", "0", calc7)
    test("log(100)", "2", calc7)
    test("sqrt(4)", "2", calc7)
    test("sqrt(0-1)", "i", calc7)
    test("abs(3+4*i)", "5", calc7)
    test("e^(i*pi)", "-1", calc7)
    test("1+2", "3", calc7)

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

    # calc13: single-variable (same as calc12)
    test("2*x+3*x", "5*x", calc13)
    test("2*x=4", "x=2", calc13)
    test("x^2=1", "x=-1; x=1", calc13)
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc13)
    test("2+3", "5", calc13)

    # calc13: multi-variable simplification
    test("x+y+x", "2*x+y", calc13)
    test("3*x+2*y-x", "2*x+2*y", calc13)
    test("x+y-x-y", "0", calc13)
    test("2*x+3*y+x-y", "3*x+2*y", calc13)

    # calc13: two-variable linear systems
    test("x+y=2; x-y=0", "x=1; y=1", calc13)
    test("2*x+3*y=7; x-y=1", "x=2; y=1", calc13)
    test("x+y=10; 2*x+y=15", "x=5; y=5", calc13)
    test("x+2*y=5; 3*x-y=1", "x=1; y=2", calc13)
    test("x=3; x+y=5", "x=3; y=2", calc13)

    # calc13: three-variable linear systems
    test("x+y+z=6; x-y=0; x+z=4", "x=2; y=2; z=2", calc13)
    test("x+y+z=3; x+y-z=1; x-y+z=1", "x=1; y=1; z=1", calc13)
    test("x+y+z=10; x-y+z=4; x+y-z=2", "x=3; y=3; z=4", calc13)
    test("2*x+y-z=1; x+y+z=6; x-y+2*z=5", "x=1; y=2; z=3", calc13)

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

    # calc11: Matrix operations
    test("det([[1,2],[3,4]])", "-2", calc11)
    test("det([[1,0,0],[0,1,0],[0,0,1]])", "1", calc11)
    test("trace([[1,2],[3,4]])", "5", calc11)
    test("dot([1,2,3],[4,5,6])", "32", calc11)
    test("cross([1,0,0],[0,1,0])", "[0,0,1]", calc11)

    # calc11: Matrix arithmetic
    test("[[1,2],[3,4]]+[[5,6],[7,8]]", "[[6,8],[10,12]]", calc11)
    test("[[1,2],[3,4]]*[[1,0],[0,1]]", "[[1,2],[3,4]]", calc11)
    test("[[1,2],[3,4]]*[[5,6],[7,8]]", "[[19,22],[43,50]]", calc11)
    test("[[5,6],[7,8]]-[[1,2],[3,4]]", "[[4,4],[4,4]]", calc11)

    # calc11: Inverse
    test("inv([[2,1],[1,1]])", "[[1,-1],[-1,2]]", calc11)

    # calc11: Scalar operations
    test("2*[[1,2],[3,4]]", "[[2,4],[6,8]]", calc11)
    test("[[2,4],[6,8]]/2", "[[1,2],[3,4]]", calc11)

    # calc11: Transpose and RREF
    test("trans([[1,2,3],[4,5,6]])", "[[1,4],[2,5],[3,6]]", calc11)
    test("rref([[1,2,3],[4,5,6]])", "[[1,0,-1],[0,1,2]]", calc11)

    # calc11: Matrix power
    test("[[1,1],[0,1]]^3", "[[1,3],[0,1]]", calc11)
