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
    "calc1": {"function": calc1, "description": "Basic addition and subtraction"},
    "calc2": {"function": calc2, "description": "Add, subtract, multiply, and divide"},
    "calc3": {"function": calc3, "description": "Parentheses and exponents (recursive descent parser)"},
    "calc4": {"function": calc4, "description": "Scientific notation and decimals"},
    "calc5": {"function": calc5, "description": "Named constants (pi, e)"},
    "calc6": {"function": calc6, "description": "Complex numbers (imaginary unit i)"},
    "calc7": {"function": calc7, "description": "Math functions (sin, cos, sqrt, ln, etc.)"},
    "calc8": {"function": calc8, "description": "Algebra: simplify expressions and solve equations"},
    "calc9": {"function": calc9, "description": "Multi-variable linear equation systems"},
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
        {"name": name, "description": info["description"]}
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
