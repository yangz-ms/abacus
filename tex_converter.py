import re


def input_to_tex(expression, calculator='calc1'):
    """Convert a calculator input expression string to LaTeX."""
    # Handle systems of equations (semicolon-separated)
    if ';' in expression:
        parts = [s.strip() for s in expression.split(';')]
        tex_parts = [input_to_tex(p, calculator) for p in parts]
        return r'\begin{cases} ' + r' \\ '.join(tex_parts) + r' \end{cases}'

    # Handle equation with '='
    if '=' in expression:
        sides = expression.split('=', 1)
        left_tex = _input_expr_to_tex(sides[0].strip(), calculator)
        right_tex = _input_expr_to_tex(sides[1].strip(), calculator)
        return left_tex + ' = ' + right_tex

    return _input_expr_to_tex(expression, calculator)


def _input_expr_to_tex(expr, calculator):
    """Convert a single expression (no = or ;) to LaTeX."""
    expr = expr.strip()
    if not expr:
        return ''

    # Tokenize
    tokens = _tokenize_input(expr)

    # Convert tokens to TeX
    return _tokens_to_tex(tokens, calculator)


def _tokenize_input(expr):
    """Tokenize an input expression into a list of token dicts."""
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]

        # Whitespace
        if c == ' ':
            i += 1
            continue

        # Number (possibly with decimal and scientific notation)
        if c.isdigit() or (c == '.' and i + 1 < len(expr) and expr[i + 1].isdigit()):
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            if j < len(expr) and expr[j] == '.':
                j += 1
                while j < len(expr) and expr[j].isdigit():
                    j += 1
            # Scientific notation: e/E followed by optional +/- and digits
            if j < len(expr) and expr[j] in ('e', 'E'):
                k = j + 1
                if k < len(expr) and expr[k] in ('+', '-'):
                    k += 1
                if k < len(expr) and expr[k].isdigit():
                    j = k
                    while j < len(expr) and expr[j].isdigit():
                        j += 1
                # else: not scientific notation, 'e' is probably Euler's number
            tokens.append({'type': 'number', 'value': expr[i:j]})
            i = j
            continue

        # Identifier (function name, constant, or variable)
        if c.isalpha():
            j = i
            while j < len(expr) and expr[j].isalpha():
                j += 1
            tokens.append({'type': 'ident', 'value': expr[i:j]})
            i = j
            continue

        # Operators and parens
        if c in ('+', '-', '*', '/', '^', '(', ')'):
            tokens.append({'type': 'op', 'value': c})
            i += 1
            continue

        # Unknown character - just pass through
        tokens.append({'type': 'other', 'value': c})
        i += 1

    return tokens


_FUNCTIONS = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
              'sinh', 'cosh', 'tanh', 'exp', 'ln', 'log', 'sqrt', 'abs'}

_TEX_FUNCTIONS = {'sin': r'\sin', 'cos': r'\cos', 'tan': r'\tan',
                  'asin': r'\arcsin', 'acos': r'\arccos', 'atan': r'\arctan',
                  'sinh': r'\sinh', 'cosh': r'\cosh', 'tanh': r'\tanh',
                  'exp': r'\exp', 'ln': r'\ln', 'log': r'\log'}

_CONSTANTS = {'pi': r'\pi', 'e': 'e', 'i': 'i'}

_VARIABLES = set('abcdefghjklmnopqrstuvwxyz')  # single-letter variables (excluding 'i' and 'e')


def _is_number_token(tok):
    return tok['type'] == 'number'


def _is_variable(name):
    return len(name) == 1 and name.isalpha() and name not in ('e', 'i') and name not in _FUNCTIONS


def _format_sci_notation(value):
    """Convert scientific notation like 1.5e3 to 1.5 \\times 10^{3}."""
    for sep in ('e', 'E'):
        if sep in value:
            mantissa, exponent = value.split(sep, 1)
            # Clean up exponent: remove leading +
            if exponent.startswith('+'):
                exponent = exponent[1:]
            return mantissa + r' \times 10^{' + exponent + '}'
    return value


def _tokens_to_tex(tokens, calculator):
    """Convert a list of tokens to a TeX string."""
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok['type'] == 'number':
            val = tok['value']
            if 'e' in val or 'E' in val:
                result.append(_format_sci_notation(val))
            else:
                result.append(val)
            i += 1

        elif tok['type'] == 'ident':
            name = tok['value']

            # Function call: name followed by '('
            if name in _FUNCTIONS and i + 1 < len(tokens) and tokens[i + 1]['value'] == '(':
                # Find matching closing paren
                paren_start = i + 1
                depth = 0
                j = paren_start
                while j < len(tokens):
                    if tokens[j]['value'] == '(':
                        depth += 1
                    elif tokens[j]['value'] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                paren_end = j

                # Get inner tokens
                inner_tokens = tokens[paren_start + 1:paren_end]
                inner_tex = _tokens_to_tex(inner_tokens, calculator)

                if name == 'sqrt':
                    result.append(r'\sqrt{' + inner_tex + '}')
                elif name == 'abs':
                    result.append('|' + inner_tex + '|')
                elif name in _TEX_FUNCTIONS:
                    result.append(_TEX_FUNCTIONS[name] + '(' + inner_tex + ')')
                else:
                    result.append(r'\mathrm{' + name + '}(' + inner_tex + ')')

                i = paren_end + 1

            # Known constant
            elif name in _CONSTANTS:
                result.append(_CONSTANTS[name])
                i += 1

            # Variable (single letter)
            elif _is_variable(name):
                result.append(name)
                i += 1

            # Unknown identifier - pass through
            else:
                result.append(name)
                i += 1

        elif tok['type'] == 'op':
            op = tok['value']

            if op == '*':
                # Determine context: number*number -> \times, otherwise \cdot or implicit
                prev_tok = _find_prev_meaningful(tokens, i)
                next_tok = _find_next_meaningful(tokens, i)

                if prev_tok and next_tok:
                    prev_is_num = _is_pure_number(prev_tok)
                    next_is_num = _is_pure_number(next_tok)
                    prev_is_var_or_const = (prev_tok['type'] == 'ident' and
                                            (prev_tok['value'] in _CONSTANTS or _is_variable(prev_tok['value'])))
                    next_is_var_or_const = (next_tok['type'] == 'ident' and
                                            (next_tok['value'] in _CONSTANTS or _is_variable(next_tok['value'])))
                    next_is_func = (next_tok['type'] == 'ident' and next_tok['value'] in _FUNCTIONS)
                    next_is_open_paren = (next_tok['type'] == 'op' and next_tok['value'] == '(')

                    if prev_is_num and next_is_num:
                        result.append(r' \times ')
                    elif prev_is_num and next_is_open_paren:
                        result.append(r' \times ')
                    elif prev_is_num and (next_is_var_or_const or next_is_func):
                        # Implicit multiplication: 2*x -> 2x, 2*pi -> 2\pi, 4*i -> 4i
                        next_val = next_tok['value']
                        # Need a space before multi-char constants that become TeX commands
                        if next_val in _CONSTANTS and len(_CONSTANTS[next_val]) > 1:
                            result.append(' ')
                        else:
                            result.append('')
                    elif prev_is_var_or_const and (next_is_var_or_const or next_is_func):
                        # Implicit: i*pi -> i\pi, x*y -> xy
                        result.append(' ')
                    else:
                        result.append(r' \cdot ')
                else:
                    result.append(r' \cdot ')
                i += 1

            elif op == '/':
                result.append(' / ')
                i += 1

            elif op == '^':
                # Collect the exponent
                i += 1
                if i < len(tokens) and tokens[i]['value'] == '(':
                    # Exponent is a parenthesized expression
                    depth = 0
                    j = i
                    while j < len(tokens):
                        if tokens[j]['value'] == '(':
                            depth += 1
                        elif tokens[j]['value'] == ')':
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    inner_tokens = tokens[i + 1:j]
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append('^{' + inner_tex + '}')
                    i = j + 1
                elif i < len(tokens):
                    # Single token exponent
                    exp_tok = tokens[i]
                    if exp_tok['type'] == 'number':
                        result.append('^{' + exp_tok['value'] + '}')
                    elif exp_tok['type'] == 'ident':
                        if exp_tok['value'] in _CONSTANTS:
                            result.append('^{' + _CONSTANTS[exp_tok['value']] + '}')
                        else:
                            result.append('^{' + exp_tok['value'] + '}')
                    else:
                        result.append('^{' + exp_tok['value'] + '}')
                    i += 1

            elif op == '+':
                result.append(' + ')
                i += 1

            elif op == '-':
                # Check if this is unary minus (at start or after open paren/operator)
                prev_tok = _find_prev_meaningful(tokens, i)
                if prev_tok is None or (prev_tok['type'] == 'op' and prev_tok['value'] in ('(', '+', '-', '*', '/', '^')):
                    result.append('-')
                else:
                    result.append(' - ')
                i += 1

            elif op == '(':
                result.append('(')
                i += 1

            elif op == ')':
                result.append(')')
                i += 1

            else:
                result.append(op)
                i += 1

        else:
            result.append(tok['value'])
            i += 1

    return ''.join(result)


def _find_prev_meaningful(tokens, idx):
    """Find the previous non-whitespace token before idx."""
    if idx <= 0:
        return None
    return tokens[idx - 1]


def _find_next_meaningful(tokens, idx):
    """Find the next non-whitespace token after idx."""
    if idx + 1 >= len(tokens):
        return None
    return tokens[idx + 1]


def _is_pure_number(tok):
    """Check if a token represents a pure number."""
    return tok['type'] == 'number'


def output_to_tex(result, calculator='calc1'):
    """Convert calculator output to LaTeX."""
    result = result.strip()
    if not result:
        return ''

    # Handle semicolon-separated results (solutions or multi-var)
    if '; ' in result:
        parts = [s.strip() for s in result.split('; ')]
        tex_parts = [_output_part_to_tex(p, calculator) for p in parts]
        return r', \quad '.join(tex_parts)

    return _output_part_to_tex(result, calculator)


def _output_part_to_tex(part, calculator):
    """Convert a single output part to LaTeX."""
    part = part.strip()

    # Solution format: var=value
    if '=' in part:
        var, val = part.split('=', 1)
        return var.strip() + ' = ' + _output_value_to_tex(val.strip())

    return _output_value_to_tex(part)


def _output_value_to_tex(val):
    """Convert a single output value (number, polynomial, complex) to LaTeX."""
    # Plain integer or float
    if re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', val):
        if 'e' in val or 'E' in val:
            return _format_sci_notation(val)
        return val

    # Complex number patterns
    # Pure imaginary: "i", "-i", "3i", "-3i", "0.5i"
    m = re.fullmatch(r'(-?)(\d+\.?\d*)?i', val)
    if m:
        sign = m.group(1)
        coeff = m.group(2)
        if coeff is None or coeff == '':
            return sign + 'i'
        return sign + coeff + 'i'

    # Complex: "a+bi", "a-bi", "a+i", "a-i"
    m = re.fullmatch(r'(-?\d+\.?\d*)([+-])(\d+\.?\d*)?i', val)
    if m:
        real = m.group(1)
        sign = m.group(2)
        imag_coeff = m.group(3)
        if imag_coeff is None or imag_coeff == '':
            return real + sign + 'i'
        return real + sign + imag_coeff + 'i'

    # Polynomial: contains variable terms with ^ and *
    # Handle patterns like "5*x", "x^2", "3*x^2+2*x+1", "x^2-1"
    return _polynomial_to_tex(val)


def _polynomial_to_tex(val):
    """Convert polynomial output string to LaTeX."""
    result = val

    # Convert x^N to x^{N}
    result = re.sub(r'([a-z])\^(\d+)', r'\1^{\2}', result)

    # Convert coefficient*variable to coefficient variable (implicit multiplication)
    # e.g., "5*x" -> "5x", "3*x^{2}" -> "3x^{2}"
    result = re.sub(r'(\d+\.?\d*)\*([a-z])', r'\1\2', result)

    # Convert -1* patterns: already handled by calc output as just "-"

    return result


if __name__ == '__main__':
    # -- input_to_tex tests --

    # calc: basic arithmetic
    assert input_to_tex("1+2+3", "calc1") == "1 + 2 + 3"
    assert input_to_tex("123+456 - 789", "calc1") == "123 + 456 - 789"

    # calc2: multiply and divide
    assert input_to_tex("1+2*3-4", "calc2") == r"1 + 2 \times 3 - 4"
    assert input_to_tex("1*2*3*4*5/6", "calc2") == r"1 \times 2 \times 3 \times 4 \times 5 / 6"

    # calc3: parentheses and powers
    assert input_to_tex("1+2*(3-4)", "calc3") == r"1 + 2 \times (3 - 4)"
    assert input_to_tex("(3^5+2)/(7*7)", "calc3") == r"(3^{5} + 2) / (7 \times 7)"

    # calc4: scientific notation
    assert input_to_tex("1.5e3*2", "calc4") == r"1.5 \times 10^{3} \times 2"
    assert input_to_tex("2.5e-3", "calc4") == r"2.5 \times 10^{-3}"

    # calc5: constants
    assert input_to_tex("2*pi", "calc5") == r"2 \pi"
    assert input_to_tex("e^2", "calc5") == "e^{2}"

    # calc6: complex
    assert input_to_tex("(1+i)*(1-i)", "calc6") == r"(1 + i) \cdot (1 - i)"
    assert input_to_tex("e^(i*pi)", "calc6") == r"e^{i \pi}"

    # calc7: functions
    assert input_to_tex("sin(pi/2)", "calc7") == r"\sin(\pi / 2)"
    assert input_to_tex("sqrt(4)", "calc7") == r"\sqrt{4}"
    assert input_to_tex("ln(1)", "calc7") == r"\ln(1)"
    assert input_to_tex("abs(3+4*i)", "calc7") == r"|3 + 4i|"

    # calc8: algebra
    assert input_to_tex("2*x+3*x", "calc8") == "2x + 3x"
    assert input_to_tex("(x+1)*(x-1)", "calc8") == r"(x + 1) \cdot (x - 1)"
    assert input_to_tex("x^2-5*x+6=0", "calc8") == "x^{2} - 5x + 6 = 0"
    assert input_to_tex("3*x^2+2*x+1", "calc8") == "3x^{2} + 2x + 1"

    # calc9: multi-variable
    assert input_to_tex("x+y+x", "calc9") == "x + y + x"
    assert input_to_tex("x+y=2; x-y=0", "calc9") == r"\begin{cases} x + y = 2 \\ x - y = 0 \end{cases}"
    assert input_to_tex("x+y+z=6; x-y=0; x+z=4", "calc9") == r"\begin{cases} x + y + z = 6 \\ x - y = 0 \\ x + z = 4 \end{cases}"

    # -- output_to_tex tests --

    # Plain numbers
    assert output_to_tex("6", "calc1") == "6"
    assert output_to_tex("-210", "calc1") == "-210"
    assert output_to_tex("3000.0", "calc4") == "3000.0"
    assert output_to_tex("0.0025", "calc4") == "0.0025"

    # Constants
    assert output_to_tex("3.141592653589793", "calc5") == "3.141592653589793"

    # Complex
    assert output_to_tex("i", "calc6") == "i"
    assert output_to_tex("2+3i", "calc6") == "2+3i"
    assert output_to_tex("-1", "calc6") == "-1"
    assert output_to_tex("1+i", "calc6") == "1+i"

    # Polynomials
    assert output_to_tex("5*x", "calc8") == "5x"
    assert output_to_tex("x^2-1", "calc8") == "x^{2}-1"
    assert output_to_tex("3*x^2+2*x+1", "calc8") == "3x^{2}+2x+1"

    # Solutions
    assert output_to_tex("x=2", "calc8") == "x = 2"
    assert output_to_tex("x=-1; x=1", "calc8") == r"x = -1, \quad x = 1"
    assert output_to_tex("x=2; x=3", "calc8") == r"x = 2, \quad x = 3"

    # Multi-var solutions
    assert output_to_tex("2*x+y", "calc9") == "2x+y"
    assert output_to_tex("x=1; y=1", "calc9") == r"x = 1, \quad y = 1"
    assert output_to_tex("x=2; y=2; z=2", "calc9") == r"x = 2, \quad y = 2, \quad z = 2"

    print("All tex_converter tests passed.")
