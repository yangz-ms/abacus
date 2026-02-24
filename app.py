import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from calc import calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9, calc10, calc11, calc12, calc13
from calc import calc14, calc15
from tex_converter import input_to_tex, output_to_tex

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CALCULATORS = {
    "calc1": {
        "function": calc1,
        "description": "Basic addition and subtraction",
        "short_desc": "Add & Subtract",
        "examples": ["1+2+3", "123-456", "123+456-789"],
        "group": "expression",
    },
    "calc2": {
        "function": calc2,
        "description": "Add, subtract, multiply, and divide",
        "short_desc": "Multiply & Divide",
        "examples": ["1+2*3-4", "123+456*789", "1*2*3*4*5/6"],
        "group": "expression",
    },
    "calc3": {
        "function": calc3,
        "description": "Parentheses and exponents (recursive descent parser)",
        "short_desc": "Parentheses & Exponents",
        "examples": ["2^10", "1+2*(3-4)", "(3^5+2)/(7*7)"],
        "group": "expression",
    },
    "calc4": {
        "function": calc4,
        "description": "Scientific notation and decimals",
        "short_desc": "Scientific Notation",
        "examples": ["1.5e3*2", "2.5e-3", "(1e2+1.5e2)*2e1"],
        "group": "expression",
    },
    "calc5": {
        "function": calc5,
        "description": "Named constants (pi, e)",
        "short_desc": "Constants (pi, e)",
        "examples": ["2*pi", "e^2", "pi+e"],
        "group": "expression",
    },
    "calc6": {
        "function": calc6,
        "description": "Complex numbers (imaginary unit i)",
        "short_desc": "Complex Numbers",
        "examples": ["i^2", "(1+i)*(1-i)", "(1+i)/(1-i)", "e^(i*pi)"],
        "group": "expression",
    },
    "calc7": {
        "function": calc7,
        "description": "Math functions (sin, cos, sqrt, ln, etc.)",
        "short_desc": "Math Functions",
        "examples": ["sin(pi/2)", "log(100)", "sqrt(4)", "abs(3+4*i)"],
        "group": "expression",
    },
    "calc8": {
        "function": calc8,
        "description": "Number theory: GCD, LCM, factorial, prime factorization, modulo, rounding",
        "short_desc": "Number Theory",
        "group": "expression",
        "examples": ["gcd(12,8)", "5!", "factor(60)", "17%3", "floor(3.7)"],
    },
    "calc9": {
        "function": calc9,
        "description": "Combinatorics: permutations and combinations",
        "short_desc": "Combinatorics",
        "group": "expression",
        "examples": ["C(10,3)", "P(5,2)", "C(52,5)"],
    },
    "calc10": {
        "function": calc10,
        "description": "Extended trig (degree mode), reciprocal trig, arbitrary-base logarithm, polar/rectangular conversion",
        "short_desc": "Trig & Logs",
        "group": "expression",
        "examples": ["sin(90d)", "sec(pi/4)", "logb(2,8)", "polar(3,4)"],
    },
    "calc11": {
        "function": calc11,
        "description": "Matrix arithmetic, determinant, inverse, transpose, dot and cross products",
        "short_desc": "Matrices",
        "group": "expression",
        "examples": ["det([[1,2],[3,4]])", "inv([[2,1],[1,1]])", "dot([1,2,3],[4,5,6])", "[[1,2],[3,4]]*[[5,6],[7,8]]"],
    },
    "calc12": {
        "function": calc12,
        "description": "Algebra: simplify expressions and solve equations",
        "short_desc": "Algebra & Equations",
        "examples": ["(x+1)*(x-1)", "(x+1)^2", "x^2-5*x+6=0", "x^3-6*x^2+11*x-6=0"],
        "group": "solver",
    },
    "calc13": {
        "function": calc13,
        "description": "Multi-variable linear equation systems",
        "short_desc": "Linear Systems",
        "examples": ["3*x+2*y-x", "x+y=2; x-y=0", "x+y+z=6; x-y=0; x+z=4"],
        "group": "solver",
    },
    "calc14": {
        "function": calc14,
        "description": "Polynomial factoring, long division, completing the square, binomial expansion, higher-degree equation solving",
        "short_desc": "Poly Tools",
        "group": "solver",
        "examples": ["factor(x^2-5*x+6)", "divpoly(x^3-1,x-1)", "complsq(x^2+6*x+5)", "binom(x+2,5)", "x^4-1=0"],
    },
    "calc15": {
        "function": calc15,
        "description": "Solve inequalities (linear, quadratic, absolute value) and classify conic sections",
        "short_desc": "Ineq & Conics",
        "group": "solver",
        "examples": ["2*x+3>7", "x^2-4<0", "abs(x-2)<=5", "conic(x^2+y^2-25)", "conic(x^2/9+y^2/4-1)"],
    },
}


class CalculateRequest(BaseModel):
    calculator: str
    expression: str
    symbolic: bool = False


class CalculateResponse(BaseModel):
    calculator: str
    expression: str
    result: str
    input_tex: str
    output_tex: str
    error: Optional[str]


@app.post("/api/calculate", response_model=CalculateResponse)
async def calculate(request: CalculateRequest):
    if request.calculator not in CALCULATORS:
        return CalculateResponse(
            calculator=request.calculator,
            expression=request.expression,
            result="",
            input_tex="",
            output_tex="",
            error=f"Unknown calculator: {request.calculator}",
        )

    try:
        func = CALCULATORS[request.calculator]["function"]
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: func(request.expression, symbolic=request.symbolic))
        tex_input = input_to_tex(request.expression, request.calculator, symbolic=request.symbolic)
        tex_output = output_to_tex(result, request.calculator, symbolic=request.symbolic)
        return CalculateResponse(
            calculator=request.calculator,
            expression=request.expression,
            result=result,
            input_tex=tex_input,
            output_tex=tex_output,
            error=None,
        )
    except Exception as e:
        return CalculateResponse(
            calculator=request.calculator,
            expression=request.expression,
            result="",
            input_tex="",
            output_tex="",
            error=str(e),
        )


@app.get("/api/calculators")
async def list_calculators():
    return [
        {
            "id": name,
            "name": name,
            "description": info["description"],
            "short_desc": info["short_desc"],
            "examples": info["examples"],
            "group": info.get("group", "expression"),
        }
        for name, info in CALCULATORS.items()
    ]


static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Calculator API is running. POST to /api/calculate to use it."}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4)
