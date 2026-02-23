from calc import calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9
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
    test("1*2*3*4*5/6", "20.0", calc2)

    # calc3: recursive descent parser with parentheses and exponents
    test("1+2+3", "6", calc3)
    test("123+456 - 789", "-210", calc3)
    test("123-456", "-333", calc3)
    test("1*2*3", "6", calc3)
    test("123+456*789", "359907", calc3)
    test("1+2*3-4", "3", calc3)
    test("1+2*3-5/4", "5.75", calc3)
    test("1*2*3*4*5/6", "20.0", calc3)
    test("1+2*(3-4)", "-1", calc3)
    test("(3^5+2)/(7*7)", "5.0", calc3)
    test("1**2", "", calc3, Exception())
    test("", "", calc3, Exception())

    # calc4: scientific notation and decimals
    test("1+2+3", "6", calc4)
    test("1+2*3-4", "3", calc4)
    test("1e2", "100.0", calc4)
    test("1.5e3", "1500.0", calc4)
    test("2.5e-3", "0.0025", calc4)
    test("1.5E+3", "1500.0", calc4)
    test("1e2+1.5e2", "250.0", calc4)
    test("1.5e3*2", "3000.0", calc4)
    test("(1e2+1.5e2)*2e1", "5000.0", calc4)
    test("1.5e3/1.5e2", "10.0", calc4)
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

    # calc8: Simplification tests
    test("x", "x", calc8)
    test("2*x+3*x", "5*x", calc8)
    test("x*x", "x^2", calc8)
    test("2+3", "5", calc8)
    test("x+1-1", "x", calc8)
    test("(x+1)*(x-1)", "x^2-1", calc8)
    test("3*x^2+2*x+1", "3*x^2+2*x+1", calc8)

    # calc8: Linear equation tests
    test("2*x=4", "x=2", calc8)
    test("x+1=3", "x=2", calc8)
    test("3*x+2=x+10", "x=4", calc8)
    test("x=5", "x=5", calc8)
    test("2*(x+1)=6", "x=2", calc8)

    # calc8: More simplification tests
    test("0*x", "0", calc8)
    test("1*x", "x", calc8)
    test("x+x+x", "3*x", calc8)
    test("x^2+x^2", "2*x^2", calc8)
    test("(x+1)^2", "x^2+2*x+1", calc8)
    test("x*0", "0", calc8)
    test("x-x", "0", calc8)
    test("x^3", "x^3", calc8)
    test("x*x*x", "x^3", calc8)
    test("(x+1)*(x+1)", "x^2+2*x+1", calc8)
    test("2*x*3", "6*x", calc8)
    test("x^2-x^2", "0", calc8)
    test("(x+1)*(x+2)", "x^2+3*x+2", calc8)

    # calc8: More linear equation tests
    test("5*x=0", "x=0", calc8)
    test("x/2=3", "x=6", calc8)
    test("10-x=3", "x=7", calc8)
    test("x+x=8", "x=4", calc8)
    test("3*x-1=2*x+4", "x=5", calc8)

    # calc8: Quadratic equation tests
    test("x^2=1", "x=-1; x=1", calc8)
    test("x^2+2*x+1=0", "x=-1", calc8)
    test("x^2-5*x+6=0", "x=2; x=3", calc8)
    test("x^2=4", "x=-2; x=2", calc8)
    test("x^2-1=0", "x=-1; x=1", calc8)
    test("x^2+1=0", "x=-i; x=i", calc8)
    test("2*x^2-8=0", "x=-2; x=2", calc8)
    test("x^2-3*x=0", "x=0; x=3", calc8)
    test("x^2-4*x+4=0", "x=2", calc8)
    test("x^2+4*x+4=0", "x=-2", calc8)
    test("x^2-2*x-3=0", "x=-1; x=3", calc8)

    # calc8: Cubic equation tests
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc8)
    test("x^3-1=0", "x=1; x=-0.5-0.866025403784439i; x=-0.5+0.866025403784439i", calc8)
    test("x^3=0", "x=0", calc8)
    test("x^3-3*x^2+3*x-1=0", "x=1", calc8)
    test("x^3+x^2-x-1=0", "x=-1; x=1", calc8)
    test("x^3=8", "x=2; x=-1-1.732050807568877i; x=-1+1.732050807568877i", calc8)
    test("x^3+1=0", "x=-1; x=0.5-0.866025403784439i; x=0.5+0.866025403784439i", calc8)
    test("x^3-x=0", "x=-1; x=0; x=1", calc8)

    # calc9: single-variable (same as calc8)
    test("2*x+3*x", "5*x", calc9)
    test("2*x=4", "x=2", calc9)
    test("x^2=1", "x=-1; x=1", calc9)
    test("x^3-6*x^2+11*x-6=0", "x=1; x=2; x=3", calc9)
    test("2+3", "5", calc9)

    # calc9: multi-variable simplification
    test("x+y+x", "2*x+y", calc9)
    test("3*x+2*y-x", "2*x+2*y", calc9)
    test("x+y-x-y", "0", calc9)
    test("2*x+3*y+x-y", "3*x+2*y", calc9)

    # calc9: two-variable linear systems
    test("x+y=2; x-y=0", "x=1; y=1", calc9)
    test("2*x+3*y=7; x-y=1", "x=2; y=1", calc9)
    test("x+y=10; 2*x+y=15", "x=5; y=5", calc9)
    test("x+2*y=5; 3*x-y=1", "x=1; y=2", calc9)
    test("x=3; x+y=5", "x=3; y=2", calc9)

    # calc9: three-variable linear systems
    test("x+y+z=6; x-y=0; x+z=4", "x=2; y=2; z=2", calc9)
    test("x+y+z=3; x+y-z=1; x-y+z=1", "x=1; y=1; z=1", calc9)
    test("x+y+z=10; x-y+z=4; x+y-z=2", "x=3; y=3; z=4", calc9)
    test("2*x+y-z=1; x+y+z=6; x-y+2*z=5", "x=1; y=2; z=3", calc9)
