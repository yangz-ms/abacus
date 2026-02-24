import math
import fractions
from fractions import Fraction

from calc.registry import register
from calc.core import format_complex
from calc.polyineq import Calculator16


def _clean_float(x, decimals=10):
    """Round near-integers and clean floating point noise."""
    if isinstance(x, complex):
        r = _clean_float(x.real, decimals)
        i = _clean_float(x.imag, decimals)
        if i == 0:
            return r
        return complex(r, i)
    if isinstance(x, float):
        x = round(x, decimals)
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        return x
    return x


class Matrix:
    """A simple matrix class for arithmetic, determinant, inverse, etc."""

    def __init__(self, data):
        """data is a list of lists (rows)."""
        if not data or not data[0]:
            raise Exception("Matrix cannot be empty")
        self.data = [list(row) for row in data]
        self.rows = len(data)
        self.cols = len(data[0])
        for row in data:
            if len(row) != self.cols:
                raise Exception("All rows must have the same number of columns")

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            raise Exception("Matrix dimensions must match for addition")
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            raise Exception("Matrix dimensions must match for subtraction")
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Matrix([
                [self.data[i][j] * other for j in range(self.cols)]
                for i in range(self.rows)
            ])
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise Exception(f"Cannot multiply {self.rows}x{self.cols} by {other.rows}x{other.cols}")
            return Matrix([
                [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                 for j in range(other.cols)]
                for i in range(self.rows)
            ])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        return NotImplemented

    def __neg__(self):
        return Matrix([[-self.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __pow__(self, exp):
        if not isinstance(exp, int) or exp < 0:
            raise Exception("Matrix exponent must be a non-negative integer")
        if self.rows != self.cols:
            raise Exception("Matrix must be square for exponentiation")
        if exp == 0:
            return Matrix.identity(self.rows)
        result = Matrix.identity(self.rows)
        for _ in range(exp):
            result = result * self
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise Exception("Division by zero")
            return Matrix([
                [self.data[i][j] / other for j in range(self.cols)]
                for i in range(self.rows)
            ])
        return NotImplemented

    def det(self):
        """Determinant via cofactor expansion for small matrices, LU for larger."""
        if self.rows != self.cols:
            raise Exception("Determinant requires a square matrix")
        n = self.rows
        if n == 1:
            return self.data[0][0]
        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        if n == 3:
            a = self.data
            return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                    - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        if n <= 4:
            det_val = 0
            for j in range(n):
                minor = Matrix([
                    [self.data[r][c] for c in range(n) if c != j]
                    for r in range(1, n)
                ])
                sign = 1 if j % 2 == 0 else -1
                det_val += sign * self.data[0][j] * minor.det()
            return det_val
        # LU decomposition for larger matrices
        mat = [list(row) for row in self.data]
        det_val = 1
        for col in range(n):
            max_val = abs(mat[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(mat[row][col]) > max_val:
                    max_val = abs(mat[row][col])
                    max_row = row
            if max_val < 1e-12:
                return 0
            if max_row != col:
                mat[col], mat[max_row] = mat[max_row], mat[col]
                det_val *= -1
            det_val *= mat[col][col]
            for row in range(col + 1, n):
                factor = mat[row][col] / mat[col][col]
                for j in range(col + 1, n):
                    mat[row][j] -= factor * mat[col][j]
        return det_val

    def inv(self):
        """Inverse via Gauss-Jordan elimination."""
        if self.rows != self.cols:
            raise Exception("Inverse requires a square matrix")
        n = self.rows
        aug = [list(self.data[i]) + [1 if i == j else 0 for j in range(n)] for i in range(n)]
        for col in range(n):
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < 1e-12:
                raise Exception("Matrix is singular, cannot invert")
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
        return Matrix([[aug[i][j + n] for j in range(n)] for i in range(n)])

    def transpose(self):
        """Swap rows and columns."""
        return Matrix([[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)])

    def trace(self):
        """Sum of diagonal elements."""
        if self.rows != self.cols:
            raise Exception("Trace requires a square matrix")
        return sum(self.data[i][i] for i in range(self.rows))

    def rref(self):
        """Reduced row echelon form via Gaussian elimination."""
        mat = [list(row) for row in self.data]
        rows = self.rows
        cols = self.cols
        pivot_row = 0
        for col in range(cols):
            if pivot_row >= rows:
                break
            max_val = abs(mat[pivot_row][col])
            max_r = pivot_row
            for r in range(pivot_row + 1, rows):
                if abs(mat[r][col]) > max_val:
                    max_val = abs(mat[r][col])
                    max_r = r
            if max_val < 1e-12:
                continue
            mat[pivot_row], mat[max_r] = mat[max_r], mat[pivot_row]
            pivot = mat[pivot_row][col]
            for j in range(cols):
                mat[pivot_row][j] /= pivot
            for r in range(rows):
                if r == pivot_row:
                    continue
                factor = mat[r][col]
                for j in range(cols):
                    mat[r][j] -= factor * mat[pivot_row][j]
            pivot_row += 1
        return Matrix(mat)

    def _format_element(self, x):
        """Format a single matrix element for display."""
        x = _clean_float(x)
        if isinstance(x, float):
            if x == int(x):
                return str(int(x))
            return str(x)
        return str(x)

    def __str__(self):
        return '[' + ','.join(
            '[' + ','.join(self._format_element(x) for x in row) + ']'
            for row in self.data
        ) + ']'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def identity(n):
        """n x n identity matrix."""
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @staticmethod
    def dot(v1, v2):
        """Dot product of two vectors (1D lists)."""
        if len(v1) != len(v2):
            raise Exception("Vectors must have the same length for dot product")
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def cross(v1, v2):
        """Cross product of two 3D vectors."""
        if len(v1) != 3 or len(v2) != 3:
            raise Exception("Cross product requires 3D vectors")
        return [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]


class Calculator19(Calculator16):
    """Matrices and vectors: arithmetic, determinant, inverse, dot/cross product."""

    MATRIX_FUNCTIONS = {'det', 'inv', 'trans', 'trace', 'rref'}
    VECTOR_FUNCTIONS = {'dot', 'cross'}

    def __init__(self, expression):
        # Tokenize with support for '[' and ']' for matrix/vector literals
        self.exp = []
        self.idx = 0
        self._var_name = None
        self._var_names = set()
        self._multi_mode = False
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
                # Scientific notation
                if j < len(expression) and expression[j] in ('e', 'E'):
                    k = j + 1
                    if k < len(expression) and expression[k] in ('+', '-'):
                        k += 1
                    if k < len(expression) and expression[k] >= '0' and expression[k] <= '9':
                        j = k
                        while j < len(expression) and expression[j] >= '0' and expression[j] <= '9':
                            j += 1
                num_str = expression[i:j]
                # Degree suffix: number followed by 'd'
                if j < len(expression) and expression[j] == 'd' and (j + 1 >= len(expression) or not expression[j + 1].isalpha()):
                    import math as _math
                    val = float(num_str) * _math.pi / 180
                    self.exp.append(str(val))
                    j += 1
                else:
                    self.exp.append(num_str)
                i = j
            elif c.isalpha():
                j = i
                while j < len(expression) and expression[j].isalpha():
                    j += 1
                self.exp.append(expression[i:j])
                i = j
            elif c in ('<', '>'):
                if i + 1 < len(expression) and expression[i + 1] == '=':
                    self.exp.append(c + '=')
                    i += 2
                else:
                    self.exp.append(c)
                    i += 1
            elif c in ('+', '-', '*', '/', '^', '(', ')', ',', '[', ']', '!', '%'):
                self.exp.append(c)
                i += 1
            elif c == ' ':
                i += 1
            else:
                raise Exception(f"Invalid character '{c}'")

    def _parse_vector_list(self):
        """Parse a comma-separated list of expressions inside brackets."""
        items = []
        items.append(self.Expr())
        while self.PeekNextToken() == ',':
            self.PopNextToken()
            items.append(self.Expr())
        return items

    def Value(self):
        next = self.PeekNextToken()

        if next == '[':
            return self._parse_matrix_or_vector()

        if next == "(":
            self.PopNextToken()
            result = self.Expr()
            next = self.PopNextToken()
            if next != ")":
                raise Exception(f"Invalid token {next}")
            return result

        if next is not None and next[0].isalpha():
            name = self.PopNextToken()
            if self.PeekNextToken() == "(":
                if name in self.MATRIX_FUNCTIONS:
                    self.PopNextToken()
                    arg = self.Expr()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    if name == 'det':
                        if not isinstance(arg, Matrix):
                            raise Exception("det() requires a matrix argument")
                        return _clean_float(arg.det())
                    elif name == 'inv':
                        if not isinstance(arg, Matrix):
                            raise Exception("inv() requires a matrix argument")
                        return arg.inv()
                    elif name == 'trans':
                        if not isinstance(arg, Matrix):
                            raise Exception("trans() requires a matrix argument")
                        return arg.transpose()
                    elif name == 'trace':
                        if not isinstance(arg, Matrix):
                            raise Exception("trace() requires a matrix argument")
                        return _clean_float(arg.trace())
                    elif name == 'rref':
                        if not isinstance(arg, Matrix):
                            raise Exception("rref() requires a matrix argument")
                        return arg.rref()
                elif name in self.VECTOR_FUNCTIONS:
                    self.PopNextToken()
                    arg1 = self.Expr()
                    if self.PopNextToken() != ',':
                        raise Exception(f"{name}() requires two arguments")
                    arg2 = self.Expr()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    if name == 'dot':
                        v1 = self._to_vector(arg1)
                        v2 = self._to_vector(arg2)
                        return _clean_float(Matrix.dot(v1, v2))
                    elif name == 'cross':
                        v1 = self._to_vector(arg1)
                        v2 = self._to_vector(arg2)
                        result = Matrix.cross(v1, v2)
                        return [_clean_float(x) for x in result]
                elif name in self.MULTI_FUNCTIONS:
                    self.PopNextToken()  # consume '('
                    args = self._parse_function_args()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    return self.MULTI_FUNCTIONS[name](args)
                elif name in self.FUNCTIONS:
                    func = self.FUNCTIONS[name]
                    self.PopNextToken()
                    result = self.Expr()
                    closing = self.PopNextToken()
                    if closing != ")":
                        raise Exception(f"Invalid token {closing}")
                    return func(result)
                else:
                    raise Exception(f"Unknown function '{name}'")
            else:
                if name in self.CONSTANTS:
                    return self.CONSTANTS[name]
                else:
                    raise Exception(f"Unknown constant '{name}'")
        else:
            next = self.PopNextToken()
            if next is None:
                raise Exception("Unexpected end")
            try:
                if '.' in next or 'e' in next or 'E' in next:
                    return float(next)
                else:
                    return int(next)
            except (ValueError, TypeError):
                raise Exception(f"Unexpected token {next}")

    def _parse_matrix_or_vector(self):
        """Parse [[1,2],[3,4]] as Matrix or [1,2,3] as vector (list)."""
        self.PopNextToken()  # consume '['
        if self.PeekNextToken() == '[':
            rows = []
            rows.append(self._parse_row())
            while self.PeekNextToken() == ',':
                self.PopNextToken()
                rows.append(self._parse_row())
            closing = self.PopNextToken()
            if closing != ']':
                raise Exception(f"Expected ']', got {closing}")
            return Matrix(rows)
        else:
            items = self._parse_vector_list()
            closing = self.PopNextToken()
            if closing != ']':
                raise Exception(f"Expected ']', got {closing}")
            return items

    def _parse_row(self):
        """Parse a single row [a,b,c] inside a matrix."""
        opening = self.PopNextToken()
        if opening != '[':
            raise Exception(f"Expected '[', got {opening}")
        items = self._parse_vector_list()
        closing = self.PopNextToken()
        if closing != ']':
            raise Exception(f"Expected ']', got {closing}")
        return items

    def _to_vector(self, val):
        """Convert a value to a vector (list of numbers)."""
        if isinstance(val, list):
            return val
        if isinstance(val, Matrix):
            if val.cols == 1:
                return [val.data[i][0] for i in range(val.rows)]
            if val.rows == 1:
                return val.data[0]
            raise Exception("Cannot convert a multi-row/multi-col matrix to a vector")
        raise Exception("Expected a vector argument")

    def Product(self):
        result = self.Power()
        next = self.PeekNextToken()
        while next == "*" or next == "/" or next == "%":
            self.PopNextToken()
            right = self.Power()
            if next == "*":
                if isinstance(result, Matrix) and isinstance(right, Matrix):
                    result = result * right
                elif isinstance(result, Matrix) and isinstance(right, (int, float, complex)):
                    result = result * right
                elif isinstance(result, (int, float, complex)) and isinstance(right, Matrix):
                    result = right * result
                else:
                    result = result * right
            elif next == "/":
                if isinstance(result, Matrix) and isinstance(right, (int, float, complex)):
                    result = result / right
                else:
                    if isinstance(result, (int, fractions.Fraction)) and isinstance(right, (int, fractions.Fraction)):
                        result = fractions.Fraction(result) / fractions.Fraction(right)
                    else:
                        result = result / right
            elif next == "%":
                result = int(result) % int(right)
            next = self.PeekNextToken()
        return result

    def Sum(self):
        result = self.Product()
        next = self.PeekNextToken()
        while next == "+" or next == "-":
            self.PopNextToken()
            right = self.Product()
            if next == "+":
                result = result + right
            elif next == "-":
                result = result - right
            next = self.PeekNextToken()
        return result

    def Power(self):
        result = self.Postfix()
        next = self.PeekNextToken()
        if next == "^":
            self.PopNextToken()
            exponent = self.Power()
            if isinstance(result, Matrix):
                result = result ** int(exponent)
            else:
                result = pow(result, exponent)
        return result


def _format_matrix_result(value):
    """Format a calc19 result for output."""
    if isinstance(value, Matrix):
        cleaned = Matrix([
            [_clean_float(value.data[i][j]) for j in range(value.cols)]
            for i in range(value.rows)
        ])
        return str(cleaned)
    if isinstance(value, list):
        return '[' + ','.join(_format_scalar(x) for x in value) + ']'
    return _format_scalar(value)


def _format_scalar(x):
    """Format a scalar number for matrix output."""
    x = _clean_float(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        return str(x)
    if isinstance(x, complex):
        return format_complex(x)
    return str(x)


@register("calc19", description="Matrix arithmetic, determinant, inverse, transpose, dot and cross products",
          short_desc="Vectors & Matrices", group="expression",
          examples=["det([[1,2],[3,4]])", "inv([[2,1],[1,1]])", "dot([1,2,3],[4,5,6])", "[[1,2],[3,4]]*[[5,6],[7,8]]"],
          i18n={"zh": "\u77e9\u9635\u8fd0\u7b97", "hi": "\u0906\u0935\u094d\u092f\u0942\u0939", "es": "Matrices", "fr": "Matrices", "ar": "\u0627\u0644\u0645\u0635\u0641\u0648\u0641\u0627\u062a", "pt": "Matrizes", "ru": "\u041c\u0430\u0442\u0440\u0438\u0446\u044b", "ja": "\u884c\u5217", "de": "Matrizen"})
def calc19(expression):
    """Matrices and vectors."""
    calculator = Calculator19(expression)
    result = calculator.Parse()
    if isinstance(result, (int, float, complex)) and not isinstance(result, bool):
        return format_complex(result)
    return _format_matrix_result(result)
