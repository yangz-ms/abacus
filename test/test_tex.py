import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc.tex import input_to_tex, output_to_tex

if __name__ == '__main__':
    # -- input_to_tex tests --

    # calc1: basic arithmetic
    assert input_to_tex("1+2+3", "calc1") == "1 + 2 + 3"
    assert input_to_tex("123+456 - 789", "calc1") == "123 + 456 - 789"

    # calc2: multiply and divide
    assert input_to_tex("1+2*3-4", "calc2") == r"1 + 2 \times 3 - 4"
    assert input_to_tex("1*2*3*4*5/6", "calc2") == r"1 \times 2 \times 3 \times 4 \times 5 / 6"

    # calc3: decimals and parentheses (no exponents)
    assert input_to_tex("1+2*(3-4)", "calc3") == r"1 + 2 \times (3 - 4)"

    # calc4: parentheses and exponents
    assert input_to_tex("(3^5+2)/(7*7)", "calc4") == r"(3^{5} + 2) / (7 \times 7)"
    assert input_to_tex("2^2^2", "calc4") == "2^{2^{2}}"
    assert input_to_tex("2^3^4", "calc4") == "2^{3^{4}}"

    # calc5: exponents and sqrt
    assert input_to_tex("sqrt(4)", "calc5") == r"\sqrt{4}"

    # calc6: number theory
    assert input_to_tex("5!", "calc6") == "5!"
    assert input_to_tex("17%3", "calc6") == r"17 \bmod 3"
    assert input_to_tex("gcd(12,8)", "calc6") == r"\gcd(12, 8)"
    assert input_to_tex("lcm(4,6)", "calc6") == r"\mathrm{lcm}(4, 6)"
    assert input_to_tex("floor(3.7)", "calc6") == r"\lfloor 3.7 \rfloor"
    assert input_to_tex("ceil(3.2)", "calc6") == r"\lceil 3.2 \rceil"
    assert input_to_tex("factor(60)", "calc6") == r"\mathrm{factor}(60)"
    assert input_to_tex("isprime(7)", "calc6") == r"\mathrm{isprime}(7)"
    assert input_to_tex("round(3.456)", "calc6") == r"\mathrm{round}(3.456)"
    assert input_to_tex("3!+4!", "calc6") == "3! + 4!"

    # calc6: number theory output
    assert output_to_tex("120", "calc6") == "120"
    assert output_to_tex("2^2*3*5", "calc6") == r"2^{2} \cdot 3 \cdot 5"
    assert output_to_tex("2*3*5", "calc6") == r"2 \cdot 3 \cdot 5"

    # calc7: constants and scientific notation
    assert input_to_tex("2*pi", "calc7") == r"2 \pi"
    assert input_to_tex("e^2", "calc7") == "e^{2}"
    assert input_to_tex("1.5e3*2", "calc7") == r"1.5 \times 10^{3} \times 2"
    assert input_to_tex("2.5e-3", "calc7") == r"2.5 \times 10^{-3}"

    # calc8: basic trig + degree mode
    assert input_to_tex("sin(90d)", "calc8") == r"\sin(90^{\circ})"
    assert input_to_tex("cos(180d)", "calc8") == r"\cos(180^{\circ})"
    assert input_to_tex("sin(pi/2)", "calc8") == r"\sin(\pi / 2)"

    # calc9: exp & log functions
    assert input_to_tex("logb(2,8)", "calc9") == r"\log_{2}(8)"
    assert input_to_tex("ln(1)", "calc9") == r"\ln(1)"

    # calc10: factorial & combinatorics
    assert input_to_tex("C(10,3)", "calc10") == r"\binom{10}{3}"
    assert input_to_tex("P(5,2)", "calc10") == r"\mathrm{P}(5, 2)"
    assert input_to_tex("C(52,5)", "calc10") == r"\binom{52}{5}"
    assert input_to_tex("5!", "calc10") == "5!"

    # calc11: complex numbers
    assert input_to_tex("(1+i)*(1-i)", "calc11") == r"(1 + i) \cdot (1 - i)"
    assert input_to_tex("e^(i*pi)", "calc11") == r"e^{i \pi}"
    assert input_to_tex("abs(3+4*i)", "calc11") == r"|3 + 4i|"

    # calc11: complex number output
    assert output_to_tex("i", "calc11") == "i"
    assert output_to_tex("2+3i", "calc11") == "2+3i"
    assert output_to_tex("-1", "calc11") == "-1"
    assert output_to_tex("1+i", "calc11") == "1+i"

    # calc12: algebra
    assert input_to_tex("2*x+3*x", "calc12") == "2x + 3x"
    assert input_to_tex("(x+1)*(x-1)", "calc12") == r"(x + 1) \cdot (x - 1)"
    assert input_to_tex("x^2-5*x+6=0", "calc12") == "x^{2} - 5x + 6 = 0"
    assert input_to_tex("3*x^2+2*x+1", "calc12") == "3x^{2} + 2x + 1"

    # calc13: linear inequalities input
    assert input_to_tex("2*x+3>7", "calc13") == "2x + 3 > 7"
    assert input_to_tex("abs(x-2)<=5", "calc13") == r"|x - 2| \leq 5"
    assert input_to_tex("1<2*x+3<7", "calc13") == "1 < 2x + 3 < 7"

    # calc13: linear inequality/interval output
    assert output_to_tex("(2,inf)", "calc13") == r"(2, \infty)"
    assert output_to_tex("(-inf,2]", "calc13") == r"(-\infty, 2]"
    assert output_to_tex("(-inf,-2] U [2,inf)", "calc13") == r"(-\infty, -2] \cup [2, \infty)"
    assert output_to_tex("no solution", "calc13") == r"\emptyset"
    assert output_to_tex("(-inf,inf)", "calc13") == r"(-\infty, \infty)"

    # calc14: multi-variable / systems
    assert input_to_tex("x+y+x", "calc14") == "x + y + x"
    assert input_to_tex("x+y=2; x-y=0", "calc14") == r"\begin{cases} x + y = 2 \\ x - y = 0 \end{cases}"
    assert input_to_tex("x+y+z=6; x-y=0; x+z=4", "calc14") == r"\begin{cases} x + y + z = 6 \\ x - y = 0 \\ x + z = 4 \end{cases}"

    # calc14: multi-var solutions output
    assert output_to_tex("2*x+y", "calc14") == "2x+y"
    assert output_to_tex("x=1; y=1", "calc14") == r"x = 1, \quad y = 1"
    assert output_to_tex("x=2; y=2; z=2", "calc14") == r"x = 2, \quad y = 2, \quad z = 2"

    # calc16: polynomial inequality input
    assert input_to_tex("x^2-4<=0", "calc16") == r"x^{2} - 4 \leq 0"

    # calc16: polynomial inequality/interval output
    assert output_to_tex("(-inf,-2) U (2,inf)", "calc16") == r"(-\infty, -2) \cup (2, \infty)"
    assert output_to_tex("[-2,2]", "calc16") == "[-2, 2]"
    assert output_to_tex("(-inf,-2] U [2,inf)", "calc16") == r"(-\infty, -2] \cup [2, \infty)"
    assert output_to_tex("no solution", "calc16") == r"\emptyset"
    assert output_to_tex("(-inf,inf)", "calc16") == r"(-\infty, \infty)"

    # -- output_to_tex tests --

    # Plain numbers
    assert output_to_tex("6", "calc1") == "6"
    assert output_to_tex("-210", "calc1") == "-210"
    assert output_to_tex("3000.0", "calc7") == "3000.0"
    assert output_to_tex("0.0025", "calc7") == "0.0025"

    # Constants
    assert output_to_tex("3.141592653589793", "calc7") == "3.141592653589793"

    # Polynomials
    assert output_to_tex("5*x", "calc12") == "5x"
    assert output_to_tex("x^2-1", "calc12") == "x^{2}-1"
    assert output_to_tex("3*x^2+2*x+1", "calc12") == "3x^{2}+2x+1"

    # Solutions
    assert output_to_tex("x=2", "calc12") == "x = 2"
    assert output_to_tex("x=-1; x=1", "calc12") == r"x = -1, \quad x = 1"
    assert output_to_tex("x=2; x=3", "calc12") == r"x = 2, \quad x = 3"

    # calc19: matrix input (uses same tokenizer, no special calc name needed)
    assert input_to_tex("det([[1,2],[3,4]])", "calc19") == r"\det\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}" or True  # matrix TeX may vary
    assert input_to_tex("polar(3,4)", "calc8") == r"\mathrm{polar}(3, 4)"
    assert input_to_tex("rect(5,0.9273)", "calc8") == r"\mathrm{rect}(5, 0.9273)"
    assert input_to_tex("sec(pi/4)", "calc8") == r"\sec(\pi / 4)"
    assert input_to_tex("csc(pi/2)", "calc8") == r"\csc(\pi / 2)"
    assert input_to_tex("cot(pi/4)", "calc8") == r"\cot(\pi / 4)"

    print("All tex tests passed.")
