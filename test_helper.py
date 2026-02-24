def test(expression, expected, op, exception=None):
    caught = None
    try:
        result = op(expression)
    except Exception as e:
        caught = e

    if exception is not None:
        if type(caught) == type(exception):
            print(f"  PASS Testing {expression}, got expected exception {caught}")
        else:
            print(f"  FAIL Testing {expression}, expected exception {exception}, got {caught}")
    else:
        if caught is not None:
            print(f"  FAIL Testing {expression}, got exception {caught}")
        else:
            if result == expected:
                print(f"  PASS Testing {expression}, got expected {expected}")
            else:
                print(f"  FAIL Testing {expression}, expected {expected}, result {result}")
