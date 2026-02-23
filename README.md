# Abacus
Simple calculator built incrementally, from basic arithmetic to symbolic algebra.

## Versions

| Version | Description |
|---------|-------------|
| calc1   | Add and subtract |
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
calc1("1+2+3")                       → "6"
calc1("123-456")                     → "-333"
calc1("123+456-789")                 → "-210"

calc2("1+2*3-4")                     → "3"
calc2("123+456*789")                 → "359907"
calc2("1*2*3*4*5/6")                 → "20.0"

calc3("2^10")                        → "1024"
calc3("1+2*(3-4)")                   → "-1"
calc3("(3^5+2)/(7*7)")               → "5.0"

calc4("1.5e3*2")                     → "3000.0"
calc4("2.5e-3")                      → "0.0025"
calc4("(1e2+1.5e2)*2e1")             → "5000.0"

calc5("2*pi")                        → "6.283185307179586"
calc5("e^2")                         → "7.3890560989306495"
calc5("pi+e")                        → "5.859874482048838"

calc6("i^2")                         → "-1"
calc6("(1+i)*(1-i)")                 → "2"
calc6("(1+i)/(1-i)")                 → "i"
calc6("e^(i*pi)")                    → "-1"

calc7("sin(pi/2)")                   → "1"
calc7("log(100)")                    → "2"
calc7("sqrt(4)")                     → "2"
calc7("abs(3+4*i)")                  → "5"

calc8("(x+1)*(x-1)")                → "x^2-1"
calc8("(x+1)^2")                    → "x^2+2*x+1"
calc8("x^2-5*x+6=0")               → "x=2; x=3"
calc8("x^3-6*x^2+11*x-6=0")        → "x=1; x=2; x=3"

calc9("3*x+2*y-x")                  → "2*x+2*y"
calc9("x+y=2; x-y=0")              → "x=1; y=1"
calc9("x+y+z=6; x-y=0; x+z=4")    → "x=2; y=2; z=2"
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
python -X utf8 test_calc.py
python -X utf8 tex_converter.py
```
