# calc
Simple calculator built incrementally, from basic arithmetic to symbolic algebra.

## Versions

| Version | Description |
|---------|-------------|
| calc    | Add and subtract |
| calc2   | Multiply and divide with operator precedence |
| calc3   | Recursive descent parser with parentheses and exponents |
| calc4   | Scientific notation and decimals |
| calc5   | Named constants (pi, e) |
| calc6   | Imaginary unit and complex numbers |
| calc7   | Math functions (sin, cos, tan, exp, ln, log, sqrt, abs, etc.) |
| calc8   | Single-variable algebra and equation solving (linear, quadratic, cubic) |
| calc9   | Multi-variable linear equations (up to 3 variables) |

## Examples

```
calc("1+2+3")                       → "6"
calc("123+456 - 789")               → "-210"

calc2("1+2*3-4")                    → "3"
calc2("1*2*3*4*5/6")                → "20.0"

calc3("1+2*(3-4)")                  → "-1"
calc3("(3^5+2)/(7*7)")              → "5.0"

calc4("1.5e3*2")                    → "3000.0"
calc4("2.5e-3")                     → "0.0025"

calc5("2*pi")                       → "6.283185307179586"
calc5("e^2")                        → "7.3890560989306495"

calc6("(1+i)*(1-i)")                → "2"
calc6("e^(i*pi)")                   → "-1"

calc7("sin(pi/2)")                  → "1"
calc7("sqrt(4)")                    → "2"
calc7("ln(1)")                      → "0"

calc8("2*x+3*x")                    → "5*x"
calc8("(x+1)*(x-1)")               → "x^2-1"
calc8("x^2-5*x+6=0")               → "x=2; x=3"
calc8("x^3-6*x^2+11*x-6=0")        → "x=1; x=2; x=3"

calc9("x+y+x")                     → "2*x+y"
calc9("x+y=2; x-y=0")              → "x=1; y=1"
calc9("x+y+z=6; x-y=0; x+z=4")    → "x=2; y=2; z=2"
```

## Run tests

```
python -X utf8 calc.py
```
