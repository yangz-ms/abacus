import asyncio
import multiprocessing
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from calc import get_registry
from calc.tex import input_to_tex, output_to_tex

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CALCULATORS = get_registry()

_CALC_TIMEOUT = 5.0
_MAX_CONCURRENT = 4
_calc_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


def _worker(calculator_id, expression, result_queue):
    """Target function for the worker process."""
    try:
        from calc import get_registry
        registry = get_registry()
        func = registry[calculator_id]["function"]
        result_queue.put(("ok", func(expression)))
    except Exception as e:
        result_queue.put(("error", str(e)))


async def _run_calc_with_timeout(calculator_id, expression):
    """Run calculation in a subprocess that gets killed on timeout."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_worker,
        args=(calculator_id, expression, queue),
        daemon=True,
    )
    proc.start()

    loop = asyncio.get_event_loop()
    try:
        # Poll the queue in a thread so we don't block the event loop
        status, value = await asyncio.wait_for(
            loop.run_in_executor(None, queue.get, True, _CALC_TIMEOUT + 1),
            timeout=_CALC_TIMEOUT,
        )
    except (asyncio.TimeoutError, Exception):
        # Kill the process immediately
        proc.kill()
        proc.join(timeout=1)
        raise asyncio.TimeoutError()

    proc.join(timeout=1)
    if status == "error":
        raise Exception(value)
    return value


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

    if _calc_semaphore._value <= 0:
        return CalculateResponse(
            calculator=request.calculator,
            expression=request.expression,
            result="",
            input_tex="",
            output_tex="",
            error="Server busy, please try again shortly",
        )

    async with _calc_semaphore:
        try:
            result = await _run_calc_with_timeout(
                request.calculator, request.expression
            )
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
        except asyncio.TimeoutError:
            return CalculateResponse(
                calculator=request.calculator,
                expression=request.expression,
                result="",
                input_tex="",
                output_tex="",
                error="Calculation timed out (limit: 5 seconds)",
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
