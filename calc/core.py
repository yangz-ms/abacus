import math
import cmath
import fractions
from fractions import Fraction
from functools import reduce

from calc.registry import register

@register("calc1", description="Basic addition and subtraction",
          short_desc="Add & Subtract", group="expression",
          examples=["1+2+3", "123-456", "123+456-789"],
          i18n={"zh": "加减运算", "hi": "जोड़ और घटाव", "es": "Suma y Resta", "fr": "Addition et Soustraction", "ar": "الجمع والطرح", "pt": "Adição e Subtração", "ru": "Сложение и вычитание", "ja": "加減算", "de": "Addition und Subtraktion"})
def calc1(expression):
    '''Evaluate addition and subtraction of integers.'''
    result = 0
    op = '+'
    current = 0
    for c in expression+'\uffff':
        if c >= '0' and c <= '9':
            # accumulate current number
            current = current * 10 + int(c)
        elif c == '+' or c == '-' or c == '\uffff':
            # start next operator, calculate current result
            if op == '+':
                result += current
            elif op == '-':
                result -= current
            current = 0
            op = c
        elif c != ' ':
            raise Exception(f"Invalid character '{c}'")
    return str(result)

@register("calc2", description="Add, subtract, multiply, and divide",
          short_desc="Multiply & Divide", group="expression",
          examples=["1+2*3-4", "123+456*789", "1*2*3*4*5/6"],
          i18n={"zh": "四则运算", "hi": "गुणा और भाग", "es": "Multiplicar y Dividir", "fr": "Multiplication et Division", "ar": "الضرب والقسمة", "pt": "Multiplicação e Divisão", "ru": "Умножение и деление", "ja": "四則演算", "de": "Multiplikation und Division"})
def calc2(expression):
    '''Evaluate addition, subtraction, multiplication, and division with operator precedence.'''
    result = 0
    result2 = 0
    op = '+'
    current = 0
    for c in expression+'\uffff':
        if c >= '0' and c <= '9':
            # accumulate current number
            current = current * 10 + int(c)
        elif c == '+' or c == '-' or c == '*' or c == '/' or c == '\uffff':
            # start next operator, calculate current result
            if op == '+':
                result += result2
                result2 = current
            elif op == '-':
                result += result2
                result2 = -current
            elif op == '*':
                result2 *= current
            elif op == '/':
                if isinstance(result2, (int, Fraction)) and isinstance(current, (int, Fraction)):
                    result2 = Fraction(result2, current)
                else:
                    result2 /= current
            current = 0
            op = c
        elif c != ' ':
            raise Exception(f"Invalid character '{c}'")

    return _format_result(result+result2)


class Calculator3:
    """Decimal arithmetic: recursive descent parser with +, -, *, /, parentheses.
    Supports integers and decimal (float) numbers. No exponents, no scientific notation.
    Uses Fraction for int/int division."""
    exp = None
    idx = 0

    def __init__(self, expression):
        self.exp = []
        self.idx = 0
        i = 0
        while i < len(expression):
            c = expression[i]
            if (c >= '0' and c <= '9') or c == '.':
                j = i
                while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                    j += 1
                if j < len(expression) and expression[j] == '.':
                    j += 1
                    while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                        j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '(', ')'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def PeekNextToken(self):
        if self.idx >= len(self.exp):
            return None
        return self.exp[self.idx]

    def PopNextToken(self):
        if self.idx >= len(self.exp):
            return None
        result = self.exp[self.idx]
        self.idx += 1
        return result

    def Expr(self):
        return self.Sum()

    def Parse(self):
        result = self.Expr()
        if self.PeekNextToken() is not None:
            raise Exception(f"Unexpected token {self.PeekNextToken()}")
        return result

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result

    def Product(self):
        result = self.Value()
        next = self.PeekNextToken()
        while next == "*" or next == "/":
            next = self.PopNextToken()
            nextResult = self.Value()
            if next == "*":
                result *= nextResult
            elif next == "/":
                if isinstance(result, (int, fractions.Fraction)) and isinstance(nextResult, (int, fractions.Fraction)):
                    result = fractions.Fraction(result) / fractions.Fraction(nextResult)
                else:
                    result /= nextResult
            next = self.PeekNextToken()
        return result

    def Sum(self):
        result = self.Product()
        next = self.PeekNextToken()
        while next == "+" or next == "-":
            next = self.PopNextToken()
            nextResult = self.Product()
            if next == "+":
                result += nextResult
            elif next == "-":
                result -= nextResult
            next = self.PeekNextToken()
        return result


def _format_result(value):
    if isinstance(value, fractions.Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return str(int(value))
    return str(value)


@register("calc3", description="Decimal arithmetic",
          short_desc="Decimals", group="expression",
          examples=["1.5+2.3", "3.14*2", "10/3"],
          i18n={"zh": "小数运算", "hi": "दशमलव", "es": "Decimales", "fr": "Décimales", "ar": "الكسور العشرية", "pt": "Decimais", "ru": "Десятичные", "ja": "小数", "de": "Dezimalzahlen"})
def calc3(expression):
    '''
    Decimal arithmetic with recursive descent parser.
    Use the following grammar:
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Value (('*' / '/') Value)*
    Value   <- Number / '(' Expr ')'
    Number  <- [0-9]+ ('.' [0-9]+)?
    '''
    calculator = Calculator3(expression)
    return _format_result(calculator.Parse())


class Calculator4(Calculator3):
    """Extends Calculator3 with ^ (exponent) operator and parentheses for PEMDAS."""

    def __init__(self, expression):
        self.exp = []
        self.idx = 0
        i = 0
        while i < len(expression):
            c = expression[i]
            if (c >= '0' and c <= '9') or c == '.':
                j = i
                while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                    j += 1
                if j < len(expression) and expression[j] == '.':
                    j += 1
                    while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                        j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def Power(self):
        result = self.Value()
        next = self.PeekNextToken()
        if next == "^":
            next = self.PopNextToken()
            nextResult = self.Power()
            result = pow(result, nextResult)
        return result

    def Product(self):
        result = self.Power()
        next = self.PeekNextToken()
        while next == "*" or next == "/":
            next = self.PopNextToken()
            nextResult = self.Power()
            if next == "*":
                result *= nextResult
            elif next == "/":
                if isinstance(result, (int, fractions.Fraction)) and isinstance(nextResult, (int, fractions.Fraction)):
                    result = fractions.Fraction(result) / fractions.Fraction(nextResult)
                else:
                    result /= nextResult
            next = self.PeekNextToken()
        return result


@register("calc4", description="Parentheses and exponents (order of operations)",
          short_desc="Parentheses & Exponents", group="expression",
          examples=["2^10", "1+2*(3-4)", "(3^5+2)/(7*7)"],
          i18n={"zh": "括号与幂", "hi": "कोष्ठक और घातांक", "es": "Paréntesis y Exponentes", "fr": "Parenthèses et Exposants", "ar": "الأقواس والأُسُس", "pt": "Parênteses e Expoentes", "ru": "Скобки и степени", "ja": "括弧と累乗", "de": "Klammern und Exponenten"})
def calc4(expression):
    '''
    Extends calc3 to support exponents and full PEMDAS.
    Use the following grammar:
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- Number / '(' Expr ')'
    Number  <- [0-9]+ ('.' [0-9]+)?
    '''
    calculator = Calculator4(expression)
    return _format_result(calculator.Parse())


class Calculator5(Calculator4):
    """Extends Calculator4 with sqrt function and alpha identifier parsing."""

    FUNCTIONS = {
        'sqrt': cmath.sqrt,
    }

    def __init__(self, expression):
        self.exp = []
        self.idx = 0
        i = 0
        while i < len(expression):
            c = expression[i]
            if (c >= '0' and c <= '9') or c == '.':
                j = i
                while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                    j += 1
                if j < len(expression) and expression[j] == '.':
                    j += 1
                    while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                        j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        elif next is not None and next[0].isalpha():
            next = self.PopNextToken()
            if next in self.FUNCTIONS and self.PeekNextToken() == "(":
                func = self.FUNCTIONS[next]
                self.PopNextToken()  # consume '('
                arg = self.Expr()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
                result = func(arg)
                # Convert complex result to real if imaginary part is zero
                if isinstance(result, complex) and result.imag == 0:
                    result = result.real
            else:
                raise Exception(f"Unknown identifier '{next}'")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result


@register("calc5", description="Exponents and square root",
          short_desc="Exponents & Sqrt", group="expression",
          examples=["sqrt(4)", "2^10+sqrt(9)", "sqrt(2)"],
          i18n={"zh": "指数与平方根", "hi": "घातांक और वर्गमूल", "es": "Exponentes y Raíces", "fr": "Exposants et Racines", "ar": "الأسس والجذور", "pt": "Expoentes e Raízes", "ru": "Степени и корни", "ja": "指数と平方根", "de": "Exponenten und Wurzeln"})
def calc5(expression):
    '''
    Extends calc4 to support sqrt function.
    Use the following grammar:
    Expr     <- Sum
    Sum      <- Product (('+' / '-') Product)*
    Product  <- Power (('*' / '/') Power)*
    Power    <- Value ('^' Power)?
    Value    <- Function '(' Expr ')' / Number / '(' Expr ')'
    Number   <- [0-9]+ ('.' [0-9]+)?
    Function <- 'sqrt'
    '''
    calculator = Calculator5(expression)
    result = calculator.Parse()
    if isinstance(result, complex):
        return format_complex(result)
    return _format_result(result)


class Calculator6(Calculator5):
    """Extends Calculator5 with scientific notation, named constants (pi, e), and imaginary unit i."""

    CONSTANTS = {
        'pi': math.pi,
        'e':  math.e,
        'i':  complex(0, 1),
    }

    FUNCTIONS = {
        **Calculator5.FUNCTIONS,
    }

    def __init__(self, expression):
        self.exp = []
        self.idx = 0
        i = 0
        while i < len(expression):
            c = expression[i]
            if (c >= '0' and c <= '9') or c == '.':
                j = i
                while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                    j += 1
                if j < len(expression) and expression[j] == '.':
                    j += 1
                    while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                        j += 1
                # only consume e/E as scientific notation if followed by digit or sign+digit
                if j < len(expression) and expression[j] in ('e', 'E'):
                    k = j + 1
                    if k < len(expression) and expression[k] in ('+', '-'):
                        k += 1
                    if k < len(expression) and expression[k] >= '0' and expression[k] <= '9':
                        j = k
                        while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                            j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('+', '-', '*', '/', '^', '(', ')', ','):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        elif next is not None and next[0].isalpha():
            next = self.PopNextToken()
            if next in self.FUNCTIONS and self.PeekNextToken() == "(":
                func = self.FUNCTIONS[next]
                self.PopNextToken()  # consume '('
                arg = self.Expr()
                closing = self.PopNextToken()
                if closing != ")":
                    raise Exception(f"Invalid token {closing}")
                result = func(arg)
                if isinstance(result, complex) and result.imag == 0:
                    result = result.real
            elif next in self.CONSTANTS:
                result = self.CONSTANTS[next]
            else:
                raise Exception(f"Unknown identifier '{next}'")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next or 'e' in next or 'E' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result


def format_complex(value):
    def fmt_num(x):
        if isinstance(x, fractions.Fraction):
            if x.denominator == 1:
                return str(x.numerator)
            return f"{x.numerator}/{x.denominator}"
        if isinstance(x, float) and math.isfinite(x):
            rounded = round(x)
            if abs(x - rounded) < 1e-12:
                return str(rounded)
            if x == int(x):
                return str(int(x))
        return str(x)

    if isinstance(value, fractions.Fraction):
        return _format_result(value)
    if not isinstance(value, complex):
        return fmt_num(value)

    real = value.real
    imag = value.imag
    # Clean up floating-point noise: if very close to integer, snap
    if abs(real - round(real)) < 1e-12:
        real = round(real)
    if abs(imag - round(imag)) < 1e-12:
        imag = round(imag)

    if imag == 0:
        return fmt_num(real)
    if real == 0:
        if imag == 1:
            return 'i'
        elif imag == -1:
            return '-i'
        return f"{fmt_num(imag)}i"
    if imag == 1:
        return f"{fmt_num(real)}+i"
    elif imag == -1:
        return f"{fmt_num(real)}-i"
    elif imag > 0:
        return f"{fmt_num(real)}+{fmt_num(imag)}i"
    else:
        return f"{fmt_num(real)}{fmt_num(imag)}i"


@register("calc6", description="Complex numbers (imaginary unit i)",
          short_desc="Complex Numbers", group="expression",
          examples=["i^2", "(1+i)*(1-i)", "(1+i)/(1-i)", "e^(i*pi)"],
          i18n={"zh": "复数运算", "hi": "सम्मिश्र संख्याएँ", "es": "Números Complejos", "fr": "Nombres Complexes", "ar": "الأعداد المركبة", "pt": "Números Complexos", "ru": "Комплексные числа", "ja": "複素数", "de": "Komplexe Zahlen"})
def calc6(expression):
    '''
    Extends calc5 to support scientific notation, named constants (pi, e),
    imaginary unit i, and complex numbers.
    Use the following grammar:
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- Function '(' Expr ')' / Constant / Number / '(' Expr ')'
    Number  <- [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant <- 'pi' / 'e' / 'i'
    Function <- 'sqrt'
    '''
    calculator = Calculator6(expression)
    return format_complex(calculator.Parse())


class Calculator7(Calculator6):
    FUNCTIONS = {
        **Calculator6.FUNCTIONS,
        'sin':   cmath.sin,
        'cos':   cmath.cos,
        'tan':   cmath.tan,
        'asin':  cmath.asin,
        'acos':  cmath.acos,
        'atan':  cmath.atan,
        'sinh':  cmath.sinh,
        'cosh':  cmath.cosh,
        'tanh':  cmath.tanh,
        'exp':   cmath.exp,
        'ln':    cmath.log,
        'log':   cmath.log10,
        'abs':   abs,
    }
    MULTI_FUNCTIONS = {}

    def _parse_function_args(self):
        args = [self.Expr()]
        while self.PeekNextToken() == ",":
            self.PopNextToken()
            args.append(self.Expr())
        return args

    def Value(self):
        next = self.PeekNextToken()
        if next == "(":
            next = self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
        elif next is not None and next[0].isalpha():
            next = self.PopNextToken()
            if self.PeekNextToken() == "(":
                if next in self.MULTI_FUNCTIONS:
                    handler = self.MULTI_FUNCTIONS[next]
                    self.PopNextToken()
                    args = self._parse_function_args()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = handler(*args)
                elif next in self.FUNCTIONS:
                    func = self.FUNCTIONS[next]
                    self.PopNextToken()
                    args = self._parse_function_args()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    result = func(args[0])
                else:
                    raise Exception(f"Unknown function '{next}'")
            else:
                if next not in self.CONSTANTS:
                    raise Exception(f"Unknown constant '{next}'")
                result = self.CONSTANTS[next]
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next or 'e' in next or 'E' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result


@register("calc7", description="Math functions (sin, cos, sqrt, ln, etc.)",
          short_desc="Math Functions", group="expression",
          examples=["sin(pi/2)", "log(100)", "sqrt(4)", "abs(3+4*i)"],
          i18n={"zh": "数学函数", "hi": "गणितीय फलन", "es": "Funciones Matemáticas", "fr": "Fonctions Mathématiques", "ar": "الدوال الرياضية", "pt": "Funções Matemáticas", "ru": "Математические функции", "ja": "数学関数", "de": "Mathematische Funktionen"})
def calc7(expression):
    '''
    Extends calc6 to support common math functions.
    Use the following grammar:
    Expr     <- Sum
    Sum      <- Product (('+' / '-') Product)*
    Product  <- Power (('*' / '/') Power)*
    Power    <- Value ('^' Power)?
    Value    <- Function '(' Expr ')' / Constant / Number / '(' Expr ')'
    Number   <- [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant <- 'pi' / 'e' / 'i'
    Function <- 'sin' / 'cos' / 'tan' / 'asin' / 'acos' / 'atan' /
               'sinh' / 'cosh' / 'tanh' / 'exp' / 'ln' / 'log' /
               'sqrt' / 'abs'
    '''
    calculator = Calculator7(expression)
    return format_complex(calculator.Parse())
