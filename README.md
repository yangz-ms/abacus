# Abacus
Simple calculator built incrementally, from basic arithmetic to symbolic algebra.

## Versions

| Version | Description |
|---------|-------------|
| calc1   | Add and subtract |
| calc2   | Multiply and divide with operator precedence |
| calc3   | Decimal arithmetic with recursive descent parser and parentheses |
| calc4   | Parentheses and exponents (full PEMDAS) |
| calc5   | Square root function |
| calc6   | GCD, LCM, primes, modulo, floor, ceil, rounding |
| calc7   | Named constants (pi, e) and scientific notation |
| calc8   | Basic trigonometry (sin, cos, tan) with degree mode |
| calc9   | Exponential and logarithmic functions (exp, ln, log, logb) |
| calc10  | Factorial and combinatorics (n!, C, P) |
| calc11  | Complex numbers (imaginary unit i) |
| calc12  | Single-variable algebra and equation solving (linear, quadratic, cubic) |
| calc13  | Linear and absolute-value inequalities |
| calc14  | Multi-variable linear equation systems |
| calc15  | Polynomial tools: factoring, long division, completing the square, binomial expansion |
| calc16  | Polynomial inequalities (quadratic, cubic) |
| calc17  | Rational expressions: simplify, add, subtract, multiply, divide polynomial fractions |
| calc18  | Radical expressions: simplify radicals, rationalize denominators, radical arithmetic |
| calc19  | Matrices and vectors: arithmetic, determinant, inverse, dot/cross product |

## Examples

```
calc1("1+2+3")                          -> "6"
calc2("1*2*3*4*5/6")                    -> "20"
calc3("10/3")                           -> "10/3"
calc3("1.5+2.3")                        -> "3.8"
calc4("(3^5+2)/(7*7)")                  -> "5"
calc5("sqrt(4)")                        -> "2"

calc6("gcd(12,8)")                      -> "4"
calc6("factor(60)")                     -> "2^2*3*5"
calc7("2*pi")                           -> "6.283185307179586"
calc7("1.5e3*2")                        -> "3000"
calc8("sin(90d)")                       -> "1"
calc9("logb(2,1024)")                   -> "10"
calc10("5!")                            -> "120"
calc10("C(10,3)")                       -> "120"
calc11("e^(i*pi)")                      -> "-1"

calc12("(x+1)*(x-1)")                  -> "x^2-1"
calc12("x^2-5*x+6=0")                  -> "x=2; x=3"
calc13("2*x+3>7")                       -> "(2,inf)"
calc13("abs(x-3)<5")                    -> "(-2,8)"
calc14("x+y=2; x-y=0")                 -> "x=1; y=1"
calc15("factor(x^3-6*x^2+11*x-6)")     -> "(x-1)*(x-2)*(x-3)"
calc15("binom(x+1,3)")                 -> "x^3+3*x^2+3*x+1"
calc16("x^2-4>0")                       -> "(-inf,-2) U (2,inf)"
calc17("simplify(x^2-1,x+1)")          -> "x-1"
calc17("radd(1,x+1,1,x-1)")            -> "(2*x)/(x^2-1)"
calc18("simplifyrad(50)")               -> "5*sqrt(2)"
calc18("rationalize(1,1,2)")            -> "sqrt(2)/2"
calc19("det([[1,2],[3,4]])")            -> "-2"
calc19("cross([1,0,0],[0,1,0])")        -> "[0,0,1]"
```

## Web UI

A web interface with LaTeX-rendered input/output, powered by FastAPI and KaTeX.

```
pip install -r requirements.txt
python app.py
```

Then open http://localhost:8000.

## Run tests

```
python test/test_core.py
python test/test_numtheory.py
python test/test_matrix.py
python test/test_algebra.py
python test/test_systems.py
python test/test_polytools.py
python test/test_inequalities.py
python test/test_rational.py
python test/test_radical.py
python test/test_tex.py
python test/test_inheritance.py
python test/test_backward_compat.py
```
