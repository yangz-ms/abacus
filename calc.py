import math

def calc(expression):
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

def calc2(expression):
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

    return str(result+result2)

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


def calc3(expression):
    '''
    Use the following grammar
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← [0-9]+ / '(' Expr ')'
    '''

    calculator = Calculator3(expression)
    return str(calculator.Expr())

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


def calc4(expression):
    '''
    Extends calc3 to support scientific notation and decimals.
    Use the following grammar:
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← Number / '(' Expr ')'
    Number  ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    '''
    calculator = Calculator4(expression)
    return str(calculator.Expr())


class Calculator5(Calculator4):
    CONSTANTS = {
        'pi': math.pi,
        'e':  math.e,
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


def calc5(expression):
    '''
    Extends calc4 to support named constants (pi, e).
    Use the following grammar:
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← Number / Constant / '(' Expr ')'
    Number  ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant ← 'pi' / 'e'
    '''
    calculator = Calculator5(expression)
    return str(calculator.Expr())


class Calculator6(Calculator5):
    CONSTANTS = {**Calculator5.CONSTANTS, 'i': complex(0, 1)}


def format_complex(value):
    def fmt_num(x):
        if isinstance(x, float) and x == int(x):
            return str(int(x))
        return str(x)

    if not isinstance(value, complex):
        return fmt_num(value)

    real, imag = value.real, value.imag

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


def calc6(expression):
    '''
    Extends calc5 to support imaginary unit i and complex numbers.
    Use the following grammar:
    Expr    ← Sum
    Sum     ← Product (('+' / '-') Product)*
    Product ← Power (('*' / '/') Power)*
    Power   ← Value ('^' Power)?
    Value   ← Number / Constant / '(' Expr ')'
    Number  ← [0-9]* ('.' [0-9]*)? (('e'/'E') ('+'/'-')? [0-9]+)?
    Constant ← 'pi' / 'e' / 'i'
    '''
    calculator = Calculator6(expression)
    return format_complex(calculator.Expr())


def test(expression, expected, op = calc, exception = None):
    caught = None
    try:
        result = op(expression)
    except Exception as e:
        caught = e
    
    if exception is not None:
        if type(caught) == type(exception):
            print(f"✅ Testing {expression}, got expected exception {caught}")
        else:
            print(f"❌ Testing {expression}, expected exception {exception}, got {caught}")
    else:
        if caught is not None:
            print(f"❌ Testing {expression}, got exception {caught}")
        else:
            if result == expected:
                print(f"✅ Testing {expression}, got expected {expected}")
            else:
                print(f"❌ Testing {expression}, expected {expected}, result {result}")

if __name__ == '__main__':
    test("1+2+3", "6", calc)
    test("123+456 - 789", "-210", calc)
    test("123-456", "-333", calc)

    test("1+2+3", "6", calc2)
    test("123+456 - 789", "-210", calc2)
    test("123-456", "-333", calc2)
    test("1*2*3", "6", calc2)
    test("123+456*789", "359907", calc2)
    test("1+2*3-4", "3", calc2)
    test("1+2*3-5/4", "5.75", calc2)
    test("1*2*3*4*5/6", "20.0", calc2)

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

    test("pi", "3.141592653589793", calc5)
    test("e", "2.718281828459045", calc5)
    test("2*pi", "6.283185307179586", calc5)
    test("pi+e", "5.859874482048838", calc5)
    test("e^2", "7.3890560989306495", calc5)
    test("1e2*pi", "314.1592653589793", calc5)
    test("1+2", "3", calc5)
    test("foo", "", calc5, Exception())

    test("i", "i", calc6)
    test("i*i", "-1", calc6)
    test("i^2", "-1", calc6)
    test("1+i", "1+i", calc6)
    test("1-i", "1-i", calc6)
    test("(1+i)*2", "2+2i", calc6)
    test("(1+i)*(1-i)", "2", calc6)
    test("(1+i)/(1-i)", "i", calc6)
    test("2+3*i", "2+3i", calc6)
    test("1+2", "3", calc6)
