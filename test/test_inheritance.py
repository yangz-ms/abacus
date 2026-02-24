"""
Comprehensive tests for cross-calculator inheritance, edge cases,
symbolic mode, and performance.
"""

import time
import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import (calc3, calc7, calc8, calc9, calc10, calc11,
                  calc12, calc13, calc14, calc15)


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

# calc15 should inherit all lower calculator features
test("calc15 -> calc8: gcd(12,8)=4",
     lambda: calc15("gcd(12,8)"), "4")
test("calc15 -> calc8: 5!=120",
     lambda: calc15("5!"), "120")
test("calc15 -> calc8: factor(60)",
     lambda: calc15("factor(60)"), "2^2*3*5")
test("calc15 -> calc9: C(10,3)=120",
     lambda: calc15("C(10,3)"), "120")
test("calc15 -> calc9: P(5,2)=20",
     lambda: calc15("P(5,2)"), "20")
test("calc15 -> calc10: sin(90d)=1",
     lambda: calc15("sin(90d)"), "1")
test("calc15 -> calc10: cos(180d)=-1",
     lambda: calc15("cos(180d)"), "-1")
test("calc15 -> calc11: det([[1,2],[3,4]])=-2",
     lambda: calc15("det([[1,2],[3,4]])"), "-2")
test("calc15 -> calc11: trace([[1,2],[3,4]])=5",
     lambda: calc15("trace([[1,2],[3,4]])"), "5")
test("calc15 -> calc12: x^2-5*x+6=0",
     lambda: calc15("x^2-5*x+6=0"), "x=2; x=3")
test("calc15 -> basic: 2+3=5",
     lambda: calc15("2+3"), "5")

# calc14 should inherit calc8-calc13
test("calc14 -> calc8: gcd(12,8)=4",
     lambda: calc14("gcd(12,8)"), "4")
test("calc14 -> calc8: lcm(4,6)=12",
     lambda: calc14("lcm(4,6)"), "12")
test("calc14 -> calc9: C(10,3)=120",
     lambda: calc14("C(10,3)"), "120")
test("calc14 -> calc10: sin(90d)=1",
     lambda: calc14("sin(90d)"), "1")
test("calc14 -> calc10: cos(180d)=-1",
     lambda: calc14("cos(180d)"), "-1")
test("calc14 -> calc11: det([[1,2],[3,4]])=-2",
     lambda: calc14("det([[1,2],[3,4]])"), "-2")

# calc13 should inherit calc8-calc12
test("calc13 -> calc8: 5!=120",
     lambda: calc13("5!"), "120")
test("calc13 -> calc8: gcd(12,8)=4",
     lambda: calc13("gcd(12,8)"), "4")
test("calc13 -> calc10: sin(90d)=1",
     lambda: calc13("sin(90d)"), "1")
test("calc13 -> calc10: tan(45d)=1",
     lambda: calc13("tan(45d)"), "1")

# calc12 should inherit calc8-calc11
test("calc12 -> calc8: gcd(12,8)=4",
     lambda: calc12("gcd(12,8)"), "4")
test("calc12 -> calc10: sin(90d)=1",
     lambda: calc12("sin(90d)"), "1")

# =========================================================================
# 2. Edge Cases
# =========================================================================
print()
print("=" * 60)
print("EDGE CASE TESTS")
print("=" * 60)

# Empty input should raise an exception
for name, fn in [("calc3", calc3), ("calc7", calc7), ("calc8", calc8),
                 ("calc10", calc10), ("calc11", calc11), ("calc12", calc12),
                 ("calc13", calc13), ("calc14", calc14), ("calc15", calc15)]:
    test_raises(f'{name}("") raises exception', lambda f=fn: f(""))

# Division by zero
test_raises('calc7("1/0") raises exception', lambda: calc7("1/0"))

# Very large numbers
test("calc8: 20! = 2432902008176640000",
     lambda: calc8("20!"), "2432902008176640000")
test("calc8: factor(1000000007) is prime",
     lambda: calc8("factor(1000000007)"), "1000000007")

# Complex + trig
test('calc10: sin(90d)+i = 1+i',
     lambda: calc10("sin(90d)+i"), "1+i")

# Matrix edge cases
test('calc11: det([[1]]) = 1',
     lambda: calc11("det([[1]])"), "1")
test_raises('calc11: inv([[1,0],[0,0]]) raises (singular)',
            lambda: calc11("inv([[1,0],[0,0]])"))

# Inequality edge cases
test('calc15: x^2+1<0 = no solution',
     lambda: calc15("x^2+1<0"), "no solution")

# Nested operations
test('calc8: gcd(5!,4!) = 24',
     lambda: calc8("gcd(5!,4!)"), "24")

# =========================================================================
# 3. Symbolic Mode
# =========================================================================
print()
print("=" * 60)
print("SYMBOLIC MODE TESTS")
print("=" * 60)

test('calc3("1/3", symbolic=True) = "1/3"',
     lambda: calc3("1/3", symbolic=True), "1/3")
test('calc3("1/3", symbolic=False) = "0.3333..."',
     lambda: calc3("1/3", symbolic=False), "0.3333333333333333")
test('calc7("sin(pi/6)", symbolic=True) = "0.5"',
     lambda: calc7("sin(pi/6)", symbolic=True), "0.5")
test('calc12("x+1=0", symbolic=True) = "x=-1"',
     lambda: calc12("x+1=0", symbolic=True), "x=-1")
test('calc12("x+1=0", symbolic=False) = "x=-1"',
     lambda: calc12("x+1=0", symbolic=False), "x=-1")

# =========================================================================
# 4. Performance
# =========================================================================
print()
print("=" * 60)
print("PERFORMANCE TESTS")
print("=" * 60)

benchmarks = [
    ("calc12 solve (x^2-5x+6=0)", lambda: calc12("x^2-5*x+6=0")),
    ("calc8 gcd(12,8)", lambda: calc8("gcd(12,8)")),
    ("calc15 inequality (x^2-4>0)", lambda: calc15("x^2-4>0")),
    ("calc11 det 2x2", lambda: calc11("det([[1,2],[3,4]])")),
    ("calc8 20!", lambda: calc8("20!")),
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
