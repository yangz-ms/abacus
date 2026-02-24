"""Calculator 18 -- Radical Expressions.

Provides symbolic radical simplification, rationalization, and arithmetic
using the Radical class from calc.polytools.
"""

from calc.registry import register
from calc.polytools import Radical, _split_args
from calc.rational import Calculator17, calc17


# ---------------------------------------------------------------------------
# Calculator18 class - Radical Expressions
# ---------------------------------------------------------------------------

class Calculator18(Calculator17):
    """Radical expressions: simplify radicals, rationalize denominators,
    radical arithmetic.

    Inherits all capabilities from Calculator17 (rational expressions)
    through the chain Calculator16 -> Calculator15 -> Calculator14 -> ...

    NOTE: This will later be changed to extend Calculator17 (rational
    expressions) once that module is ready.
    """


# ---------------------------------------------------------------------------
# Radical operation helpers
# ---------------------------------------------------------------------------

def _simplifyrad(n):
    """Simplify sqrt(n) by extracting perfect square factors.

    Creates Radical(0, 1, n) which auto-simplifies via _simplify().
    E.g., simplifyrad(50) -> 5*sqrt(2), simplifyrad(48) -> 4*sqrt(3).
    """
    n = int(n)
    if n < 0:
        raise Exception("Cannot simplify square root of a negative number")
    if n == 0:
        return str(Radical(0, 0, 0))
    result = Radical(0, 1, n)
    return str(result)


def _rationalize(a, b, n):
    """Rationalize a / (b * sqrt(n)).

    Creates numerator Radical(a, 0, 0) and denominator Radical(0, b, n),
    then divides. The Radical.__truediv__ method handles rationalization
    by multiplying by the conjugate.

    E.g., rationalize(1, 1, 2) -> (1/2)*sqrt(2)
    """
    a = int(a)
    b = int(b)
    n = int(n)
    if b == 0:
        raise Exception("Denominator coefficient b cannot be zero")
    if n <= 0:
        raise Exception("Radicand n must be a positive integer")
    numerator = Radical(a, 0, 0)
    denominator = Radical(0, b, n)
    result = numerator / denominator
    return str(result)


def _addrad(a1, b1, n1, a2, b2, n2):
    """Add two radical expressions: (a1 + b1*sqrt(n1)) + (a2 + b2*sqrt(n2)).

    The Radical class auto-simplifies radicands, so sqrt(8) becomes 2*sqrt(2).
    Addition only works when the simplified radicands match (or one side is rational).

    E.g., addrad(0, 1, 2, 0, 1, 8) -> 3*sqrt(2)
    """
    r1 = Radical(int(a1), int(b1), int(n1))
    r2 = Radical(int(a2), int(b2), int(n2))
    result = r1 + r2
    return str(result)


def _mulrad(a1, b1, n1, a2, b2, n2):
    """Multiply two radical expressions: (a1 + b1*sqrt(n1)) * (a2 + b2*sqrt(n2)).

    Multiplication requires matching radicands (after simplification) unless
    one side is purely rational.

    E.g., mulrad(1, 1, 2, 1, -1, 2) -> -1 (conjugate product: (1+sqrt(2))(1-sqrt(2)))
    """
    r1 = Radical(int(a1), int(b1), int(n1))
    r2 = Radical(int(a2), int(b2), int(n2))
    result = r1 * r2
    return str(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@register("calc18",
          description="Radical expressions: simplify radicals, rationalize denominators, radical arithmetic",
          short_desc="Radical Expressions",
          group="solver",
          examples=["simplifyrad(50)", "rationalize(1,1,2)", "addrad(0,1,2,0,1,8)"],
          i18n={
              "zh": "\u6839\u5f0f\u8868\u8fbe\u5f0f",
              "hi": "\u0915\u0930\u0923\u0940 \u0935\u094d\u092f\u0902\u091c\u0915",
              "es": "Expresiones Radicales",
              "fr": "Expressions Radicales",
              "ar": "\u0627\u0644\u062a\u0639\u0628\u064a\u0631\u0627\u062a \u0627\u0644\u062c\u0630\u0631\u064a\u0629",
              "pt": "Express\u00f5es Radicais",
              "ru": "\u0420\u0430\u0434\u0438\u043a\u0430\u043b\u044c\u043d\u044b\u0435 \u0432\u044b\u0440\u0430\u0436\u0435\u043d\u0438\u044f",
              "ja": "\u6839\u53f7\u5f0f",
              "de": "Wurzelausdr\u00fccke",
          })
def calc18(expression):
    """Radical expressions: simplify radicals, rationalize denominators,
    radical arithmetic.

    Supported functions:
      simplifyrad(n)              -- simplify sqrt(n)
      rationalize(a, b, n)        -- rationalize a / (b*sqrt(n))
      addrad(a1, b1, n1, a2, b2, n2) -- add two radical expressions
      mulrad(a1, b1, n1, a2, b2, n2) -- multiply two radical expressions

    Everything else falls through to calc17.
    """
    expr = expression.strip()

    # simplifyrad(n)
    if expr.startswith("simplifyrad(") and expr.endswith(")"):
        inner = expr[len("simplifyrad("):-1].strip()
        return _simplifyrad(inner)

    # rationalize(a, b, n)
    if expr.startswith("rationalize(") and expr.endswith(")"):
        inner = expr[len("rationalize("):-1]
        parts = _split_args(inner)
        if len(parts) != 3:
            raise Exception("rationalize requires exactly 3 arguments: a, b, n")
        return _rationalize(parts[0].strip(), parts[1].strip(), parts[2].strip())

    # addrad(a1, b1, n1, a2, b2, n2)
    if expr.startswith("addrad(") and expr.endswith(")"):
        inner = expr[len("addrad("):-1]
        parts = _split_args(inner)
        if len(parts) != 6:
            raise Exception("addrad requires exactly 6 arguments: a1, b1, n1, a2, b2, n2")
        return _addrad(parts[0].strip(), parts[1].strip(), parts[2].strip(),
                       parts[3].strip(), parts[4].strip(), parts[5].strip())

    # mulrad(a1, b1, n1, a2, b2, n2)
    if expr.startswith("mulrad(") and expr.endswith(")"):
        inner = expr[len("mulrad("):-1]
        parts = _split_args(inner)
        if len(parts) != 6:
            raise Exception("mulrad requires exactly 6 arguments: a1, b1, n1, a2, b2, n2")
        return _mulrad(parts[0].strip(), parts[1].strip(), parts[2].strip(),
                       parts[3].strip(), parts[4].strip(), parts[5].strip())

    # Fall through to calc17 for everything else
    return calc17(expression)
