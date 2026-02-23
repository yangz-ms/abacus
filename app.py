import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from calc import calc1, calc2, calc3, calc4, calc5, calc6, calc7, calc8, calc9
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
    },
    "calc2": {
        "function": calc2,
        "description": "Add, subtract, multiply, and divide",
        "short_desc": "Multiply & Divide",
        "examples": ["1+2*3-4", "123+456*789", "1*2*3*4*5/6"],
    },
    "calc3": {
        "function": calc3,
        "description": "Parentheses and exponents (recursive descent parser)",
        "short_desc": "Parentheses & Exponents",
        "examples": ["2^10", "1+2*(3-4)", "(3^5+2)/(7*7)"],
    },
    "calc4": {
        "function": calc4,
        "description": "Scientific notation and decimals",
        "short_desc": "Scientific Notation",
        "examples": ["1.5e3*2", "2.5e-3", "(1e2+1.5e2)*2e1"],
    },
    "calc5": {
        "function": calc5,
        "description": "Named constants (pi, e)",
        "short_desc": "Constants (pi, e)",
        "examples": ["2*pi", "e^2", "pi+e"],
    },
    "calc6": {
        "function": calc6,
        "description": "Complex numbers (imaginary unit i)",
        "short_desc": "Complex Numbers",
        "examples": ["i^2", "(1+i)*(1-i)", "(1+i)/(1-i)", "e^(i*pi)"],
    },
    "calc7": {
        "function": calc7,
        "description": "Math functions (sin, cos, sqrt, ln, etc.)",
        "short_desc": "Math Functions",
        "examples": ["sin(pi/2)", "log(100)", "sqrt(4)", "abs(3+4*i)"],
    },
    "calc8": {
        "function": calc8,
        "description": "Algebra: simplify expressions and solve equations",
        "short_desc": "Algebra & Equations",
        "examples": ["(x+1)*(x-1)", "(x+1)^2", "x^2-5*x+6=0", "x^3-6*x^2+11*x-6=0"],
    },
    "calc9": {
        "function": calc9,
        "description": "Multi-variable linear equation systems",
        "short_desc": "Linear Systems",
        "examples": ["3*x+2*y-x", "x+y=2; x-y=0", "x+y+z=6; x-y=0; x+z=4"],
    },
}


class CalculateRequest(BaseModel):
    calculator: str
    expression: str


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
        result = await loop.run_in_executor(None, func, request.expression)
        tex_input = input_to_tex(request.expression, request.calculator)
        tex_output = output_to_tex(result, request.calculator)
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
