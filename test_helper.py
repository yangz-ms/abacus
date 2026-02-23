def test(expression, expected, op, exception=None):
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
