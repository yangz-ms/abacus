import math
import cmath
import fractions
from fractions import Fraction
from functools import reduce

def calc1(expression, symbolic=False):
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

def calc2(expression, symbolic=False):
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
                result2 /= current
            current = 0
            op = c
        elif c != ' ':
            raise Exception(f"Invalid character '{c}'")

    return _format_result(result+result2)

class Calculator3:
    exp = None
    idx = 0

    def __init__(self, expression, symbolic=False):
        self.symbolic = symbolic
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
                if self.symbolic and isinstance(result, (int, fractions.Fraction)) and isinstance(nextResult, (int, fractions.Fraction)):
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


def calc3(expression, symbolic=False):
    '''
    Use the following grammar
    Expr    <- Sum
    Sum     <- Product (('+' / '-') Product)*
    Product <- Power (('*' / '/') Power)*
    Power   <- Value ('^' Power)?
    Value   <- [0-9]+ / '(' Expr ')'
    '''

    calculator = Calculator3(expression, symbolic=symbolic)
    return _format_result(calculator.Parse())

class Calculator4(Calculator3):
    def __init__(self, expression, symbolic=False):
        self.symbolic = symbolic
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


def calc4(expression, symbolic=False):
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
    calculator = Calculator4(expression, symbolic=symbolic)
    return _format_result(calculator.Parse())


class Calculator5(Calculator4):
    CONSTANTS = {
        'pi': math.pi,
        'e':  math.e,
    }
    MULTI_FUNCTIONS = {}

    def __init__(self, expression, symbolic=False):
        self.symbolic = symbolic
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


def calc5(expression, symbolic=False):
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
    calculator = Calculator5(expression, symbolic=symbolic)
    return _format_result(calculator.Parse())


class Calculator6(Calculator5):
    CONSTANTS = {**Calculator5.CONSTANTS, 'i': complex(0, 1)}


def format_complex(value):
    def fmt_num(x):
        if isinstance(x, float) and math.isfinite(x) and x == int(x):
            return str(int(x))
        return str(x)

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


def calc6(expression, symbolic=False):
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
    calculator = Calculator6(expression, symbolic=symbolic)
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


def calc7(expression, symbolic=False):
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
    calculator = Calculator7(expression, symbolic=symbolic)
    return format_complex(calculator.Parse())
