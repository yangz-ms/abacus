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
