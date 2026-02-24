import math
import cmath
import fractions
from fractions import Fraction
from functools import reduce

from calc.registry import register

@register("calc1", description="Basic addition and subtraction",
          short_desc="Add & Subtract", group="expression",
          examples=["1+2+3", "123-456", "123+456-789"],
          i18n={"zh": "\u52a0\u51cf\u8fd0\u7b97", "hi": "\u091c\u094b\u0921\u093c \u0914\u0930 \u0918\u091f\u093e\u0935", "es": "Suma y Resta", "fr": "Addition et Soustraction", "ar": "\u0627\u0644\u062c\u0645\u0639 \u0648\u0627\u0644\u0637\u0631\u062d", "pt": "Adi\u00e7\u00e3o e Subtra\u00e7\u00e3o", "ru": "\u0421\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0438 \u0432\u044b\u0447\u0438\u0442\u0430\u043d\u0438\u0435", "ja": "\u52a0\u6e1b\u7b97", "de": "Addition und Subtraktion"})
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
          i18n={"zh": "\u56db\u5219\u8fd0\u7b97", "hi": "\u0917\u0941\u0923\u093e \u0914\u0930 \u092d\u093e\u0917", "es": "Multiplicar y Dividir", "fr": "Multiplication et Division", "ar": "\u0627\u0644\u0636\u0631\u0628 \u0648\u0627\u0644\u0642\u0633\u0645\u0629", "pt": "Multiplica\u00e7\u00e3o e Divis\u00e3o", "ru": "\u0423\u043c\u043d\u043e\u0436\u0435\u043d\u0438\u0435 \u0438 \u0434\u0435\u043b\u0435\u043d\u0438\u0435", "ja": "\u56db\u5247\u6f14\u7b97", "de": "Multiplikation und Division"})
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
    exp = None
    idx = 0

    def __init__(self, expression):
        self.exp = []
        current = ""
        for c in expression:
            if c >= '0' and c <= '9':
                current += c
            else:
                if current != "":
                    self.exp.append(current)
                    current = ""
            if c == '+' or c == '-' or c == '*' or c == '/' or c == '^' or c == '(' or c == ')':
                self.exp.append(str(c))
        if current != "":
            self.exp.append(current)
        self.idx = 0

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
            if not next.isdigit():
                raise Exception(f"Unexpected token {next}")
            result = int(next)
        return result

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


@register("calc3", description="Parentheses and exponents (recursive descent parser)",
          short_desc="Parentheses & Exponents", group="expression",
          examples=["2^10", "1+2*(3-4)", "(3^5+2)/(7*7)"],
          i18n={"zh": "\u62ec\u53f7\u4e0e\u5e42", "hi": "\u0915\u094b\u0937\u094d\u0920\u0915 \u0914\u0930 \u0918\u093e\u0924\u093e\u0902\u0915", "es": "Par\u00e9ntesis y Exponentes", "fr": "Parenth\u00e8ses et Exposants", "ar": "\u0627\u0644\u0623\u0642\u0648\u0627\u0633 \u0648\u0627\u0644\u0623\u064f\u0633\u064f\u0633", "pt": "Par\u00eanteses e Expoentes", "ru": "\u0421\u043a\u043e\u0431\u043a\u0438 \u0438 \u0441\u0442\u0435\u043f\u0435\u043d\u0438", "ja": "\u62ec\u5f27\u3068\u7d2f\u4e57", "de": "Klammern und Exponenten"})
def calc3(expression):
    '''
    Use the following grammar
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- [0-9]+ / '(' Expr ')'
    '''

    calculator = Calculator3(expression)
    return _format_result(calculator.Parse())

class Calculator4(Calculator3):
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
                if j < len(expression) and expression[j] in ('e', 'E'):
                    j += 1
                    if j < len(expression) and expression[j] in ('+', '-'):
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
                if '.' in next or 'e' in next or 'E' in next:
                    result = float(next)
                else:
                    result = int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")
        return result


@register("calc4", description="Scientific notation and decimals",
          short_desc="Scientific Notation", group="expression",
          examples=["1.5e3*2", "2.5e-3", "(1e2+1.5e2)*2e1"],
          i18n={"zh": "\u79d1\u5b66\u8ba1\u6570\u6cd5", "hi": "\u0935\u0948\u091c\u094d\u091e\u093e\u0928\u093f\u0915 \u0938\u0902\u0915\u0947\u0924\u0928", "es": "Notaci\u00f3n Cient\u00edfica", "fr": "Notation Scientifique", "ar": "\u0627\u0644\u062a\u0631\u0645\u064a\u0632 \u0627\u0644\u0639\u0644\u0645\u064a", "pt": "Nota\u00e7\u00e3o Cient\u00edfica", "ru": "\u041d\u0430\u0443\u0447\u043d\u0430\u044f \u0437\u0430\u043f\u0438\u0441\u044c", "ja": "\u79d1\u5b66\u8868\u8a18\u6cd5", "de": "Wissenschaftliche Notation"})
def calc4(expression):
    '''
    Extends calc3 to support scientific notation and decimals.
    Use the following grammar:
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- Number / '(' Expr ')'
    Number  <- [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    '''
    calculator = Calculator4(expression)
    return _format_result(calculator.Parse())


class Calculator5(Calculator4):
    CONSTANTS = {
        'pi': math.pi,
        'e':  math.e,
    }
    MULTI_FUNCTIONS = {}

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
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            if next[0].isalpha():
                if next not in self.CONSTANTS:
                    raise Exception(f"Unknown constant '{next}'")
                result = self.CONSTANTS[next]
            else:
                try:
                    if '.' in next or 'e' in next or 'E' in next:
                        result = float(next)
                    else:
                        result = int(next)
                except (ValueError, TypeError):
                    raise Exception(f"Unexpected token {next}")
        return result


@register("calc5", description="Named constants (pi, e)",
          short_desc="Constants (pi, e)", group="expression",
          examples=["2*pi", "e^2", "pi+e"],
          i18n={"zh": "\u5e38\u91cf (\u03c0, e)", "hi": "\u0938\u094d\u0925\u093f\u0930\u093e\u0902\u0915 (\u03c0, e)", "es": "Constantes (\u03c0, e)", "fr": "Constantes (\u03c0, e)", "ar": "\u0627\u0644\u062b\u0648\u0627\u0628\u062a (\u03c0, e)", "pt": "Constantes (\u03c0, e)", "ru": "\u041a\u043e\u043d\u0441\u0442\u0430\u043d\u0442\u044b (\u03c0, e)", "ja": "\u5b9a\u6570 (\u03c0, e)", "de": "Konstanten (\u03c0, e)"})
def calc5(expression):
    '''
    Extends calc4 to support named constants (pi, e).
    Use the following grammar:
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- Number / Constant / '(' Expr ')'
    Number  <- [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant <- 'pi' / 'e'
    '''
    calculator = Calculator5(expression)
    return _format_result(calculator.Parse())


class Calculator6(Calculator5):
    CONSTANTS = {**Calculator5.CONSTANTS, 'i': complex(0, 1)}


def format_complex(value):
    def fmt_num(x):
        if isinstance(x, fractions.Fraction):
            if x.denominator == 1:
                return str(x.numerator)
            return f"{x.numerator}/{x.denominator}"
        if isinstance(x, float) and math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)

    if isinstance(value, fractions.Fraction):
        return _format_result(value)
    if not isinstance(value, complex):
        return fmt_num(value)

    real = round(value.real, 15)
    imag = round(value.imag, 15)

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
          i18n={"zh": "\u590d\u6570\u8fd0\u7b97", "hi": "\u0938\u092e\u094d\u092e\u093f\u0936\u094d\u0930 \u0938\u0902\u0916\u094d\u092f\u093e\u090f\u0901", "es": "N\u00fameros Complejos", "fr": "Nombres Complexes", "ar": "\u0627\u0644\u0623\u0639\u062f\u0627\u062f \u0627\u0644\u0645\u0631\u0643\u0628\u0629", "pt": "N\u00fameros Complexos", "ru": "\u041a\u043e\u043c\u043f\u043b\u0435\u043a\u0441\u043d\u044b\u0435 \u0447\u0438\u0441\u043b\u0430", "ja": "\u8907\u7d20\u6570", "de": "Komplexe Zahlen"})
def calc6(expression):
    '''
    Extends calc5 to support imaginary unit i and complex numbers.
    Use the following grammar:
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- Number / Constant / '(' Expr ')'
    Number  <- [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant <- 'pi' / 'e' / 'i'
    '''
    calculator = Calculator6(expression)
    return format_complex(calculator.Parse())


class Calculator7(Calculator6):
    FUNCTIONS = {
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
        'sqrt':  cmath.sqrt,
        'abs':   abs,
    }

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
          i18n={"zh": "\u6570\u5b66\u51fd\u6570", "hi": "\u0917\u0923\u093f\u0924\u0940\u092f \u092b\u0932\u0928", "es": "Funciones Matem\u00e1ticas", "fr": "Fonctions Math\u00e9matiques", "ar": "\u0627\u0644\u062f\u0648\u0627\u0644 \u0627\u0644\u0631\u064a\u0627\u0636\u064a\u0629", "pt": "Fun\u00e7\u00f5es Matem\u00e1ticas", "ru": "\u041c\u0430\u0442\u0435\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u0444\u0443\u043d\u043a\u0446\u0438\u0438", "ja": "\u6570\u5b66\u95a2\u6570", "de": "Mathematische Funktionen"})
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
