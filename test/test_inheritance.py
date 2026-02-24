"""
Comprehensive tests for cross-calculator inheritance, edge cases,
fraction mode, and performance.
"""

import time
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import (calc3, calc5, calc6, calc7, calc8, calc9, calc10, calc11,
                  calc12, calc13, calc14, calc15, calc16, calc19)


passed = 0
failed = 0
errors = []


def test(description, fn, expected):
    global passed, failed
    try:
        result = fn()
        if result == expected:
            passed += 1
            print(f"  PASS {description}")
        else:
            failed += 1
            msg = f"  FAIL {description}: got {result!r}, expected {expected!r}"
            print(msg)
            errors.append(msg)
    except Exception as e:
        failed += 1
        msg = f"  FAIL {description}: raised {type(e).__name__}: {e}"
        print(msg)
        errors.append(msg)


def test_raises(description, fn):
    """Test that function raises an exception (graceful error handling)."""
    global passed, failed
    try:
        result = fn()
        failed += 1
        msg = f"  FAIL {description}: expected exception, got {result!r}"
        print(msg)
        errors.append(msg)
    except Exception:
        passed += 1
        print(f"  PASS {description} (raised exception as expected)")


# =========================================================================
# 1. Cross-Calculator Inheritance
#    Each higher calculator should be able to perform operations from
#    all lower calculators in the chain.
# =========================================================================
print("=" * 60)
print("CROSS-CALCULATOR INHERITANCE TESTS")
print("=" * 60)

# calc19 should inherit all lower-level numeric features
test("calc19 -> calc6: gcd(12,8)=4",
     lambda: calc19("gcd(12,8)"), "4")
test("calc19 -> calc10: C(10,3)=120",
     lambda: calc19("C(10,3)"), "120")
test("calc19 -> calc8: sin(90d)=1",
     lambda: calc19("sin(90d)"), "1")
test("calc19 -> calc8: cos(180d)=-1",
     lambda: calc19("cos(180d)"), "-1")
test("calc19 -> matrix: det([[1,2],[3,4]])=-2",
     lambda: calc19("det([[1,2],[3,4]])"), "-2")
test("calc19 -> matrix: trace([[1,2],[3,4]])=5",
     lambda: calc19("trace([[1,2],[3,4]])"), "5")
test("calc19 -> basic: 2+3=5",
     lambda: calc19("2+3"), "5")

# calc16 should inherit calc15 features
test("calc16 -> calc15: factor(x^2-5*x+6)",
     lambda: calc16("factor(x^2-5*x+6)"), "(x-2)*(x-3)")
test("calc16 -> calc15: binom(x+1,3)",
     lambda: calc16("binom(x+1,3)"), "x^3+3*x^2+3*x+1")
test("calc16 -> calc12: 2*x+3*x=5*x",
     lambda: calc16("2*x+3*x"), "5*x")
test("calc16 -> basic: 2+3=5",
     lambda: calc16("2+3"), "5")

# calc15 should inherit calc14 features
test("calc15 -> calc14: x+y=2; x-y=0",
     lambda: calc15("x+y=2; x-y=0"), "x=1; y=1")
test("calc15 -> calc12: x^2-5*x+6=0",
     lambda: calc15("x^2-5*x+6=0"), "x=2; x=3")
test("calc15 -> basic: 2+3=5",
     lambda: calc15("2+3"), "5")

# calc14 should inherit calc13 and calc12 features
test("calc14 -> calc12: 2*x=4 -> x=2",
     lambda: calc14("2*x=4"), "x=2")
test("calc14 -> calc12: x^2=1",
     lambda: calc14("x^2=1"), "x=-1; x=1")
test("calc14 -> basic: 2+3=5",
     lambda: calc14("2+3"), "5")

# calc13 should inherit calc12 features
test("calc13 -> calc12: 2*x+3*x=5*x",
     lambda: calc13("2*x+3*x"), "5*x")
test("calc13 -> calc12: 2*x=4 -> x=2",
     lambda: calc13("2*x=4"), "x=2")

# calc12 should handle all numeric features via Calculator11
test("calc12 -> calc6: gcd(12,8)=4",
     lambda: calc12("gcd(12,8)"), "4")
test("calc12 -> calc8: sin(90d)=1",
     lambda: calc12("sin(90d)"), "1")
test("calc12 -> calc10: C(10,3)=120",
     lambda: calc12("C(10,3)"), "120")

# calc11 should inherit calc10, calc9, calc8, calc7, calc6
test("calc11 -> calc10: C(10,3)=120",
     lambda: calc11("C(10,3)"), "120")
test("calc11 -> calc9: exp(1)",
     lambda: calc11("exp(1)"), "2.718281828459045")
test("calc11 -> calc8: sin(90d)=1",
     lambda: calc11("sin(90d)"), "1")
test("calc11 -> calc7: pi",
     lambda: calc11("pi"), "3.141592653589793")
test("calc11 -> calc6: gcd(12,8)=4",
     lambda: calc11("gcd(12,8)"), "4")

# calc10 should inherit calc9, calc8, calc7, calc6
test("calc10 -> calc9: logb(2,8)=3",
     lambda: calc10("logb(2,8)"), "3")
test("calc10 -> calc8: sin(90d)=1",
     lambda: calc10("sin(90d)"), "1")
test("calc10 -> calc7: pi",
     lambda: calc10("pi"), "3.141592653589793")
test("calc10 -> calc6: gcd(12,8)=4",
     lambda: calc10("gcd(12,8)"), "4")

# calc9 should inherit calc8, calc7, calc6
test("calc9 -> calc8: cos(180d)=-1",
     lambda: calc9("cos(180d)"), "-1")
test("calc9 -> calc7: e^2",
     lambda: calc9("e^2"), "7.3890560989306495")
test("calc9 -> calc6: lcm(4,6)=12",
     lambda: calc9("lcm(4,6)"), "12")

# calc8 should inherit calc7, calc6
test("calc8 -> calc7: 1e2=100",
     lambda: calc8("1e2"), "100")
test("calc8 -> calc6: gcd(12,8)=4",
     lambda: calc8("gcd(12,8)"), "4")

# calc7 should inherit calc6
test("calc7 -> calc6: factor(60)=2^2*3*5",
     lambda: calc7("factor(60)"), "2^2*3*5")
test("calc7 -> calc6: 17%3=2",
     lambda: calc7("17%3"), "2")

# calc6 should inherit calc5
test("calc6 -> calc5: sqrt(4)=2",
     lambda: calc6("sqrt(4)"), "2")
test("calc6 -> basic: 1+2=3",
     lambda: calc6("1+2"), "3")

# =========================================================================
# 2. Edge Cases
# =========================================================================
print()
print("=" * 60)
print("EDGE CASE TESTS")
print("=" * 60)

# Empty input should raise an exception
for name, fn in [("calc3", calc3), ("calc5", calc5), ("calc6", calc6),
                 ("calc7", calc7), ("calc8", calc8), ("calc9", calc9),
                 ("calc10", calc10), ("calc11", calc11), ("calc12", calc12),
                 ("calc13", calc13), ("calc14", calc14), ("calc15", calc15),
                 ("calc16", calc16), ("calc19", calc19)]:
    test_raises(f'{name}("") raises exception', lambda f=fn: f(""))

# Division by zero
test_raises('calc5("1/0") raises exception', lambda: calc5("1/0"))

# Very large numbers
test("calc10: 20! = 2432902008176640000",
     lambda: calc10("20!"), "2432902008176640000")
test("calc6: factor(1000000007) is prime",
     lambda: calc6("factor(1000000007)"), "1000000007")

# Complex + trig
test('calc11: sin(90d)+i = 1+i',
     lambda: calc11("sin(90d)+i"), "1+i")

# Matrix edge cases
test('calc19: det([[1]]) = 1',
     lambda: calc19("det([[1]])"), "1")
test_raises('calc19: inv([[1,0],[0,0]]) raises (singular)',
            lambda: calc19("inv([[1,0],[0,0]])"))

# Inequality edge cases
test('calc13: 3>2 = (-inf,inf)',
     lambda: calc13("3>2"), "(-inf,inf)")
test('calc16: x^2+1<0 = no solution',
     lambda: calc16("x^2+1<0"), "no solution")

# Nested operations
test('calc6: gcd(120,80) = 40',
     lambda: calc6("gcd(120,80)"), "40")

# =========================================================================
# 3. Fraction Default Mode
# =========================================================================
print()
print("=" * 60)
print("FRACTION DEFAULT MODE TESTS")
print("=" * 60)

test('calc3("1/3") = "1/3"',
     lambda: calc3("1/3"), "1/3")
test('calc3("10/3") = "10/3"',
     lambda: calc3("10/3"), "10/3")
test('calc8("sin(pi/6)") ~= 0.5',
     lambda: calc8("sin(pi/6)"), "0.49999999999999994")
test('calc12("x+1=0") = "x=-1"',
     lambda: calc12("x+1=0"), "x=-1")

# =========================================================================
# 4. Performance
# =========================================================================
print()
print("=" * 60)
print("PERFORMANCE TESTS")
print("=" * 60)

benchmarks = [
    ("calc12 solve (x^2-5x+6=0)", lambda: calc12("x^2-5*x+6=0")),
    ("calc6 gcd(12,8)", lambda: calc6("gcd(12,8)")),
    ("calc16 inequality (x^2-4>0)", lambda: calc16("x^2-4>0")),
    ("calc19 det 2x2", lambda: calc19("det([[1,2],[3,4]])")),
    ("calc10 20!", lambda: calc10("20!")),
]

perf_ok = True
for label, fn in benchmarks:
    start = time.time()
    for _ in range(100):
        fn()
    elapsed = time.time() - start
    avg_ms = elapsed * 10  # elapsed/100 * 1000
    status = "OK" if avg_ms < 10 else "SLOW"
    if status == "SLOW":
        perf_ok = False
    print(f"  {status} 100x {label}: {elapsed:.3f}s ({avg_ms:.1f}ms avg)")

if perf_ok:
    print("  All performance benchmarks within 10ms per call.")
else:
    print("  WARNING: Some benchmarks exceeded 10ms per call.")

# =========================================================================
# Summary
# =========================================================================
print()
print("=" * 60)
print(f"SUMMARY: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(e)

sys.exit(0 if failed == 0 else 1)
