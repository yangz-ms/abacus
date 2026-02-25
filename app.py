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
_POOL_SIZE = 4


# ---------------------------------------------------------------------------
# Persistent worker pool: pre-spawned processes that loop handling requests.
# On timeout the stuck worker is killed and a fresh one takes its place.
# ---------------------------------------------------------------------------

def _worker_loop(work_q, result_q):
    """Long-lived worker: import once, handle many requests."""
    from calc import get_registry
    registry = get_registry()
    while True:
        calc_id, expr = work_q.get()
        try:
            result = registry[calc_id]["function"](expr)
            result_q.put(("ok", result))
        except Exception as e:
            result_q.put(("error", str(e)))


class _WorkerPool:
    def __init__(self, size):
        self._size = size
        self._available = None  # asyncio.Queue, created lazily
        self._workers = []

    def _ensure_started(self):
        if self._available is not None:
            return
        self._available = asyncio.Queue()
        for _ in range(self._size):
            self._spawn()

    def _spawn(self):
        wq = multiprocessing.Queue()
        rq = multiprocessing.Queue()
        p = multiprocessing.Process(target=_worker_loop, args=(wq, rq), daemon=True)
        p.start()
        worker = (p, wq, rq)
        self._workers.append(worker)
        self._available.put_nowait(worker)

    async def run(self, calc_id, expression):
        self._ensure_started()

        # Get an available worker (wait up to 1s, else "server busy")
        try:
            worker = await asyncio.wait_for(self._available.get(), timeout=1.0)
        except asyncio.TimeoutError:
            raise Exception("Server busy, please try again shortly")

        proc, work_q, result_q = worker

        # Replace dead workers
        if not proc.is_alive():
            self._workers.remove(worker)
            self._spawn()
            worker = await self._available.get()
            proc, work_q, result_q = worker

        work_q.put((calc_id, expression))

        loop = asyncio.get_event_loop()
        try:
            status, value = await asyncio.wait_for(
                loop.run_in_executor(None, result_q.get, True, _CALC_TIMEOUT + 1),
                timeout=_CALC_TIMEOUT,
            )
        except asyncio.TimeoutError:
            # Kill the stuck worker and spawn a replacement
            proc.kill()
            proc.join(timeout=1)
            if worker in self._workers:
                self._workers.remove(worker)
            self._spawn()
            raise

        # Return worker to pool
        self._available.put_nowait(worker)

        if status == "error":
            raise Exception(value)
        return value


_pool = _WorkerPool(_POOL_SIZE)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

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
        result = await _pool.run(request.calculator, request.expression)
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
