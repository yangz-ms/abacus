import asyncio
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from calc import get_registry
from tex_converter import input_to_tex, output_to_tex

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CALCULATORS = get_registry()


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
            "i18n": info.get("i18n", {}),
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
