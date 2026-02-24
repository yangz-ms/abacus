import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import time
import subprocess

from calc.tex import input_to_tex, output_to_tex

def test_tex_converter():
    """Test TeX conversion for all calculator versions."""
    tests = [
        # (input_expr, calculator, expected_contains)
        ("1+2+3", "calc", "+"),
        ("1+2*3-4", "calc2", "\\times"),
        ("1+2*(3-4)", "calc3", "("),
        ("1.5e3*2", "calc4", "10^{"),
        ("2*pi", "calc5", "\\pi"),
        ("e^(i*pi)", "calc6", "\\pi"),
        ("sin(pi/2)", "calc7", "\\sin"),
        ("sqrt(4)", "calc7", "\\sqrt"),
        ("2*x+3*x", "calc8", "x"),
        ("x^2-5*x+6=0", "calc8", "x^{2}"),
        ("x+y=2; x-y=0", "calc9", "\\begin{cases}"),
    ]

    passed = 0
    failed = 0
    for expr, calc, expected in tests:
        tex = input_to_tex(expr, calc)
        if expected in tex:
            print(f"  PASS input_to_tex({expr!r}, {calc!r}) = {tex!r}")
            passed += 1
        else:
            print(f"  FAIL input_to_tex({expr!r}, {calc!r}) = {tex!r}, expected to contain {expected!r}")
            failed += 1

    output_tests = [
        ("6", "calc", "6"),
        ("5*x", "calc8", "5x"),
        ("x^2-1", "calc8", "x^{2}"),
        ("x=2; x=3", "calc8", "\\quad"),
        ("x=1; y=1", "calc9", "\\quad"),
    ]

    for result, calc, expected in output_tests:
        tex = output_to_tex(result, calc)
        if expected in tex:
            print(f"  PASS output_to_tex({result!r}, {calc!r}) = {tex!r}")
            passed += 1
        else:
            print(f"  FAIL output_to_tex({result!r}, {calc!r}) = {tex!r}, expected to contain {expected!r}")
            failed += 1

    return passed, failed


def test_api():
    """Test the API endpoints by starting the server."""
    import urllib.request
    import urllib.error

    # Start server
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for startup
    time.sleep(3)

    passed = 0
    failed = 0

    try:
        # Test GET /api/calculators
        req = urllib.request.Request("http://127.0.0.1:8765/api/calculators")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            if len(data) == 15:
                print(f"  PASS GET /api/calculators returned {len(data)} calculators")
                passed += 1
            else:
                print(f"  FAIL GET /api/calculators returned {len(data)} calculators, expected 15")
                failed += 1

        # Test POST /api/calculate for each calculator
        test_cases = [
            ("calc1", "1+2+3", "6"),
            ("calc2", "1+2*3-4", "3"),
            ("calc3", "1+2*(3-4)", "-1"),
            ("calc4", "1.5e3*2", "3000.0"),
            ("calc5", "2*pi", "6.283185307179586"),
            ("calc6", "(1+i)*(1-i)", "2"),
            ("calc7", "sin(pi/2)", "1"),
            ("calc12", "x^2-5*x+6=0", "x=2; x=3"),
            ("calc13", "x+y=2; x-y=0", "x=1; y=1"),
        ]

        for calc, expr, expected in test_cases:
            body = json.dumps({"calculator": calc, "expression": expr}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8765/api/calculate",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                if data["result"] == expected and data["error"] is None:
                    has_tex = bool(data.get("input_tex")) and bool(data.get("output_tex"))
                    tex_status = "with TeX" if has_tex else "NO TEX"
                    print(f"  PASS {calc}({expr!r}) = {data['result']!r} [{tex_status}]")
                    passed += 1
                    if not has_tex:
                        print(f"    WARNING: input_tex={data.get('input_tex')!r}, output_tex={data.get('output_tex')!r}")
                else:
                    print(f"  FAIL {calc}({expr!r}) expected {expected!r}, got result={data.get('result')!r} error={data.get('error')!r}")
                    failed += 1

        # Test error handling
        body = json.dumps({"calculator": "invalid", "expression": "1+2"}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8765/api/calculate",
            data=body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            if data["error"] is not None:
                print(f"  PASS error handling for invalid calculator: {data['error']!r}")
                passed += 1
            else:
                print(f"  FAIL no error for invalid calculator")
                failed += 1

        # Test concurrency: send 50 requests simultaneously
        import concurrent.futures

        def send_request(i):
            body = json.dumps({"calculator": "calc1", "expression": f"{i}+{i}"}).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8765/api/calculate",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data["result"] == str(i + i) and data["error"] is None

        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(send_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.time() - start

        all_ok = all(results)
        if all_ok:
            print(f"  PASS concurrency test: 50 requests in {elapsed:.2f}s, all correct")
            passed += 1
        else:
            fails = sum(1 for r in results if not r)
            print(f"  FAIL concurrency test: {fails}/50 failed")
            failed += 1

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    return passed, failed


if __name__ == "__main__":
    print("=== TeX Converter Tests ===")
    tp, tf = test_tex_converter()

    print("\n=== API Tests ===")
    ap, af = test_api()

    total_pass = tp + ap
    total_fail = tf + af
    print(f"\n=== TOTAL: {total_pass} passed, {total_fail} failed ===")
    sys.exit(1 if total_fail > 0 else 0)
