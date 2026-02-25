# Abacus
Simple calculator built incrementally, from basic arithmetic to symbolic algebra.

## Versions

Each calculator level inherits all capabilities from every level below it.

| Level | Name | Category | Description | Grade |
|-------|------|----------|-------------|-------|
| calc1 | Add & Subtract | expression | Addition and subtraction of integers | K-2 |
| calc2 | Multiply & Divide | expression | Four operations with operator precedence; exact Fraction for integer division | 3-4 |
| calc3 | Decimals | expression | Decimal arithmetic with recursive descent parser and parentheses | 4-5 |
| calc4 | Parentheses & Exponents | expression | Adds `^` operator for full PEMDAS | 5-6 |
| calc5 | Exponents & Sqrt | expression | Adds `sqrt()` function | 6-8 |
| calc6 | Number Theory | expression | GCD, LCM, primes, modulo (`%`), floor, ceil, isprime, factor | 6 |
| calc7 | Pi, e & Sci. Notation | expression | Named constants (`pi`, `e`) and scientific notation (`1.5e3`) | 6-8 |
| calc8 | Basic Trig | expression | sin, cos, tan, inverse trig, hyperbolic trig, degree mode (`90d`) | 9-10 |
| calc9 | Exp & Log | expression | exp, ln, log, logb (arbitrary-base logarithm) | 10-11 |
| calc10 | Factorial & Combinatorics | expression | Factorial (`n!`), combinations (`C(n,r)`), permutations (`P(n,r)`) | 10-11 |
| calc11 | Complex Numbers | expression | Imaginary unit `i`, complex arithmetic, `abs()` | 10-11 |
| calc12 | Algebra & Equations | solver | Single-variable simplification and equation solving (linear, quadratic, cubic) | 7-9 |
| calc13 | Linear Inequalities | solver | Linear and absolute-value inequalities; interval notation output | 8-9 |
| calc14 | Linear Systems | solver | Multi-variable linear equation systems (semicolon-separated) | 8-10 |
| calc15 | Polynomial Tools | solver | factor, divpoly, complsq, binom; higher-degree equation solving (Durand-Kerner) | 9-12 |
| calc16 | Polynomial Inequalities | solver | Quadratic and cubic inequalities | 10 |
| calc17 | Rational Expressions | solver | Simplify, add, subtract, multiply, divide polynomial fractions | 10 |
| calc18 | Radical Expressions | solver | Simplify radicals, rationalize denominators, radical arithmetic | 10 |
| calc19 | Vectors & Matrices | expression | Matrix arithmetic, determinant, inverse, transpose, dot/cross product | 11-12 |

## Examples

```
calc1("1+2+3")                       -> "6"
calc2("1*2*3*4*5/6")                 -> "20"
calc3("10/3")                        -> "10/3"
calc3("1.5+2.3")                     -> "3.8"
calc4("(3^5+2)/(7*7)")               -> "5"
calc5("sqrt(4)")                     -> "2"

calc6("gcd(12,8)")                   -> "4"
calc6("factor(60)")                  -> "2^2*3*5"
calc7("2*pi")                        -> "6.283185307179586"
calc7("1.5e3*2")                     -> "3000"
calc8("sin(90d)")                    -> "1"
calc9("logb(2,1024)")                -> "10"
calc10("5!")                         -> "120"
calc10("C(10,3)")                    -> "120"
calc11("e^(i*pi)")                   -> "-1"

calc12("(x+1)*(x-1)")               -> "x^2-1"
calc12("x^2-5*x+6=0")               -> "x=2; x=3"
calc13("2*x+3>7")                    -> "(2,inf)"
calc13("abs(x-3)<5")                 -> "(-2,8)"
calc14("x+y=2; x-y=0")              -> "x=1; y=1"
calc15("factor(x^3-6*x^2+11*x-6)")  -> "(x-1)*(x-2)*(x-3)"
calc15("binom(x+1,3)")              -> "x^3+3*x^2+3*x+1"
calc16("x^2-4>0")                    -> "(-inf,-2) U (2,inf)"
calc17("simplify(x^2-1,x+1)")       -> "x-1"
calc17("radd(1,x+1,1,x-1)")         -> "(2*x)/(x^2-1)"
calc18("simplifyrad(50)")            -> "5*sqrt(2)"
calc18("rationalize(1,1,2)")         -> "sqrt(2)/2"
calc19("det([[1,2],[3,4]])")         -> "-2"
calc19("cross([1,0,0],[0,1,0])")     -> "[0,0,1]"
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
