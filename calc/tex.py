import re


def input_to_tex(expression, calculator='calc1', symbolic=False):
    """Convert a calculator input expression string to LaTeX."""
    # Handle systems of equations (semicolon-separated)
    if ';' in expression:
        parts = [s.strip() for s in expression.split(';')]
        tex_parts = [input_to_tex(p, calculator) for p in parts]
        return r'\begin{cases} ' + r' \\ '.join(tex_parts) + r' \end{cases}'

    # Handle equation with '='
    if '=' in expression and '<=' not in expression and '>=' not in expression:
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

        # Two-char comparison operators (<= >=)
        if c in ('<', '>') and i + 1 < len(expr) and expr[i + 1] == '=':
            tokens.append({'type': 'op', 'value': c + '='})
            i += 2
            continue

        # Operators and parens
        if c in ('+', '-', '*', '/', '^', '(', ')', '!', '%', ',', '[', ']', '<', '>'):
            tokens.append({'type': 'op', 'value': c})
            i += 1
            continue

        # Unknown character - just pass through
        tokens.append({'type': 'other', 'value': c})
        i += 1

    return tokens


_FUNCTIONS = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
              'sinh', 'cosh', 'tanh', 'exp', 'ln', 'log', 'sqrt', 'abs',
              'gcd', 'lcm', 'factor', 'floor', 'ceil', 'round', 'isprime',
              'C', 'P', 'sec', 'csc', 'cot', 'logb', 'polar', 'rect',
              'divpoly', 'complsq', 'binom',
              'det', 'inv', 'trans', 'trace', 'dot', 'cross', 'rref'}

_TEX_FUNCTIONS = {'sin': r'\sin', 'cos': r'\cos', 'tan': r'\tan',
                  'asin': r'\arcsin', 'acos': r'\arccos', 'atan': r'\arctan',
                  'sinh': r'\sinh', 'cosh': r'\cosh', 'tanh': r'\tanh',
                  'exp': r'\exp', 'ln': r'\ln', 'log': r'\log',
                  'gcd': r'\gcd',
                  'sec': r'\sec', 'csc': r'\csc', 'cot': r'\cot',
                  'det': r'\det', 'trace': r'\mathrm{tr}'}

_CONSTANTS = {'pi': r'\pi', 'e': 'e', 'i': 'i'}

_VARIABLES = set('abcdefghjklmnopqrstuvwxyz')  # single-letter variables (excluding 'i' and 'e')


def _split_tokens_by_comma(tokens):
    """Split a list of tokens by comma tokens at depth 0."""
    groups = []
    current = []
    depth = 0
    for tok in tokens:
        if tok['value'] == '(' and tok['type'] == 'op':
            depth += 1
        elif tok['value'] == ')' and tok['type'] == 'op':
            depth -= 1
        if tok['value'] == ',' and tok['type'] == 'op' and depth == 0:
            groups.append(current)
            current = []
        else:
            current.append(tok)
    groups.append(current)
    return groups


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


def _parse_bracket_to_tex(tokens, start, calculator):
    """Parse a [...] or [[...],[...]] at position start and return (tex_string, new_index)."""
    i = start + 1  # skip opening '['
    if i < len(tokens) and tokens[i]['value'] == '[':
        # Matrix: [[row1],[row2],...]
        rows_tex = []
        while i < len(tokens) and tokens[i]['value'] == '[':
            i += 1  # skip inner '['
            row_tokens = []
            depth = 0
            while i < len(tokens):
                if tokens[i]['value'] == '[':
                    depth += 1
                elif tokens[i]['value'] == ']':
                    if depth == 0:
                        break
                    depth -= 1
                row_tokens.append(tokens[i])
                i += 1
            i += 1  # skip inner ']'
            # Split row_tokens by comma
            groups = _split_tokens_by_comma(row_tokens)
            row_tex = ' & '.join(_tokens_to_tex(g, calculator) for g in groups)
            rows_tex.append(row_tex)
            if i < len(tokens) and tokens[i]['value'] == ',':
                i += 1  # skip comma between rows
        if i < len(tokens) and tokens[i]['value'] == ']':
            i += 1  # skip outer ']'
        return r'\begin{bmatrix} ' + r' \\ '.join(rows_tex) + r' \end{bmatrix}', i
    else:
        # Vector: [a,b,c]
        vec_tokens = []
        depth = 0
        while i < len(tokens):
            if tokens[i]['value'] == '[':
                depth += 1
            elif tokens[i]['value'] == ']':
                if depth == 0:
                    break
                depth -= 1
            vec_tokens.append(tokens[i])
            i += 1
        if i < len(tokens) and tokens[i]['value'] == ']':
            i += 1  # skip ']'
        groups = _split_tokens_by_comma(vec_tokens)
        entries_tex = [_tokens_to_tex(g, calculator) for g in groups]
        return r'\begin{bmatrix} ' + r' \\ '.join(entries_tex) + r' \end{bmatrix}', i


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
            # Degree notation: number followed by 'd'
            if i + 1 < len(tokens) and tokens[i + 1]['type'] == 'ident' and tokens[i + 1]['value'] == 'd':
                result.append(r'^{\circ}')
                i += 2
            else:
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

                if name == 'C':
                    # C(n,r) -> \binom{n}{r}
                    arg_groups = _split_tokens_by_comma(inner_tokens)
                    arg1_tex = _tokens_to_tex(arg_groups[0], calculator) if len(arg_groups) > 0 else ''
                    arg2_tex = _tokens_to_tex(arg_groups[1], calculator) if len(arg_groups) > 1 else ''
                    result.append(r'\binom{' + arg1_tex + '}{' + arg2_tex + '}')
                elif name == 'logb':
                    # logb(b,x) -> \log_{b}(x)
                    arg_groups = _split_tokens_by_comma(inner_tokens)
                    base_tex = _tokens_to_tex(arg_groups[0], calculator) if len(arg_groups) > 0 else ''
                    val_tex = _tokens_to_tex(arg_groups[1], calculator) if len(arg_groups) > 1 else ''
                    result.append(r'\log_{' + base_tex + '}(' + val_tex + ')')
                elif name == 'inv':
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append(inner_tex + '^{-1}')
                elif name == 'trans':
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append(inner_tex + '^{T}')
                elif name == 'sqrt':
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append(r'\sqrt{' + inner_tex + '}')
                elif name == 'abs':
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append('|' + inner_tex + '|')
                elif name == 'floor':
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append(r'\lfloor ' + inner_tex + r' \rfloor')
                elif name == 'ceil':
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append(r'\lceil ' + inner_tex + r' \rceil')
                elif name in _TEX_FUNCTIONS:
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
                    result.append(_TEX_FUNCTIONS[name] + '(' + inner_tex + ')')
                else:
                    inner_tex = _tokens_to_tex(inner_tokens, calculator)
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
                    # Single token exponent — check for chained ^
                    exp_tok = tokens[i]
                    if exp_tok['type'] == 'number':
                        exp_tex = exp_tok['value']
                    elif exp_tok['type'] == 'ident':
                        exp_tex = _CONSTANTS.get(exp_tok['value'], exp_tok['value'])
                    else:
                        exp_tex = exp_tok['value']
                    i += 1
                    # Handle chained exponents: 2^2^2 -> 2^{2^{2}}
                    if i < len(tokens) and tokens[i]['value'] == '^':
                        rest_tex = _tokens_to_tex(tokens[i:], calculator)
                        exp_tex = exp_tex + rest_tex
                        i = len(tokens)
                    result.append('^{' + exp_tex + '}')

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

            elif op == '!':
                result.append('!')
                i += 1

            elif op == '%':
                result.append(r' \bmod ')
                i += 1

            elif op == ',':
                result.append(', ')
                i += 1

            elif op == '<=':
                result.append(r' \leq ')
                i += 1

            elif op == '>=':
                result.append(r' \geq ')
                i += 1

            elif op == '<':
                result.append(' < ')
                i += 1

            elif op == '>':
                result.append(' > ')
                i += 1

            elif op == '[':
                # Matrix or vector literal
                bracket_tex, new_i = _parse_bracket_to_tex(tokens, i, calculator)
                result.append(bracket_tex)
                i = new_i

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


def output_to_tex(result, calculator='calc1', symbolic=False):
    """Convert calculator output to LaTeX."""
    result = result.strip()
    if not result:
        return ''

    # Handle calc15 interval output
    if calculator == 'calc15':
        return _calc15_output_to_tex(result)

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

    # Matrix output: [[a,b],[c,d]]
    if val.startswith('[[') and val.endswith(']]'):
        return _matrix_output_to_tex(val)

    # Vector output: [a,b,c]
    if val.startswith('[') and val.endswith(']') and not val.startswith('[['):
        return _vector_output_to_tex(val)

    # Polynomial: contains variable terms with ^ and *
    # Handle patterns like "5*x", "x^2", "3*x^2+2*x+1", "x^2-1"
    return _polynomial_to_tex(val)


def _polynomial_to_tex(val):
    """Convert polynomial output string to LaTeX."""
    result = val

    # Check if this looks like a prime factorization (e.g. "2^2*3*5")
    if re.fullmatch(r'\d+(\^\d+)?(\*\d+(\^\d+)?)*', val):
        # Factor output: convert N^M to N^{M} and * to \cdot
        result = re.sub(r'(\d+)\^(\d+)', r'\1^{\2}', result)
        result = result.replace('*', r' \cdot ')
        return result

    # Convert sqrt(N) to \sqrt{N}
    result = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', result)

    # Convert x^N to x^{N}
    result = re.sub(r'([a-z])\^(\d+)', r'\1^{\2}', result)

    # Convert (...)^N to (...)^{N} for completing-the-square output
    result = re.sub(r'\)\^(\d+)', r')^{\1}', result)

    # Convert coefficient*variable to coefficient variable (implicit multiplication)
    # e.g., "5*x" -> "5x", "3*x^{2}" -> "3x^{2}"
    result = re.sub(r'(\d+\.?\d*)\*([a-z])', r'\1\2', result)

    # Convert factored form: )*(  -> )(  for implicit multiplication between factors
    result = result.replace(')*(', ')(')
    # Handle scalar*(  like 2*(x-1) -> 2(x-1)
    result = re.sub(r'(\d+)\*\(', r'\1(', result)

    return result


def _matrix_output_to_tex(val):
    """Convert matrix output like [[1,2],[3,4]] to bmatrix TeX."""
    inner = val[1:-1]
    rows = []
    i = 0
    while i < len(inner):
        if inner[i] == '[':
            j = inner.index(']', i)
            row_str = inner[i+1:j]
            rows.append(row_str.split(','))
            i = j + 1
        elif inner[i] == ',':
            i += 1
        else:
            i += 1
    rows_tex = [' & '.join(entries) for entries in rows]
    return r'\begin{bmatrix} ' + r' \\ '.join(rows_tex) + r' \end{bmatrix}'


def _vector_output_to_tex(val):
    """Convert vector output like [1,2,3] to column vector bmatrix TeX."""
    inner = val[1:-1]
    entries = inner.split(',')
    return r'\begin{bmatrix} ' + r' \\ '.join(entries) + r' \end{bmatrix}'


def _calc15_output_to_tex(result):
    """Convert calc15 output (intervals) to LaTeX."""
    if result == 'no solution':
        return r'\emptyset'
    if result == '(-inf,inf)':
        return r'(-\infty, \infty)'

    if any(c in result for c in ('(', '[')) and any(c in result for c in (')', ']')):
        return _interval_to_tex(result)

    return _output_value_to_tex(result)


def _interval_to_tex(result):
    """Convert interval notation to LaTeX."""
    parts = result.split(' U ')
    tex_parts = []
    for part in parts:
        part = part.strip()
        part = re.sub(r'-inf\b', r'-\\infty', part)
        part = re.sub(r'(?<!-)inf\b', r'\\infty', part)
        part = part.replace(',', ', ')
        tex_parts.append(part)
    return r' \cup '.join(tex_parts)
