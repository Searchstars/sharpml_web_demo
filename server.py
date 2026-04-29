import asyncio
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
CACHES_DIR = ROOT / "caches"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
CACHES_DIR.mkdir(exist_ok=True)

# Pin every cache the toolchain might write into the project workspace.
os.environ.setdefault("TORCH_HOME", str(CACHES_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(CACHES_DIR / "huggingface"))
os.environ.setdefault("UV_CACHE_DIR", str(CACHES_DIR / "uv"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHES_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHES_DIR / "matplotlib"))

SHARP_CMD = [
    "uvx",
    "--from=git+https://github.com/apple/ml-sharp",
    "sharp",
    "predict",
    "--device",
    os.environ.get("SHARP_DEVICE", "mps"),
]

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="ml-sharp demo")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _resolve_uvx() -> str:
    candidate = shutil.which("uvx")
    if candidate:
        return candidate
    home_local = Path.home() / ".local" / "bin" / "uvx"
    if home_local.exists():
        return str(home_local)
    raise RuntimeError("uvx not found on PATH; install uv first")


async def _run_sharp(input_path: Path, output_dir: Path) -> tuple[int, str, str]:
    uvx = _resolve_uvx()
    cmd = [uvx, *SHARP_CMD[1:], "-i", str(input_path), "-o", str(output_dir)]
    env = os.environ.copy()
    env.setdefault("PATH", "")
    env["PATH"] = f"{Path(uvx).parent}:{env['PATH']}"
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


@app.post("/api/process")
async def process(file: UploadFile) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "no filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"unsupported extension {ext}")

    job_id = uuid.uuid4().hex[:12]
    in_dir = UPLOADS_DIR / job_id
    out_dir = OUTPUTS_DIR / job_id
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = in_dir / f"input{ext}"
    with image_path.open("wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)

    started = time.monotonic()
    rc, stdout, stderr = await _run_sharp(image_path, out_dir)
    elapsed = time.monotonic() - started

    if rc != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": "sharp predict failed",
                "returncode": rc,
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
            },
        )

    plys = sorted(out_dir.rglob("*.ply"))
    if not plys:
        return JSONResponse(
            status_code=500,
            content={
                "error": "no .ply produced",
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
            },
        )

    ply = plys[0]
    return JSONResponse(
        {
            "job_id": job_id,
            "image_url": f"/uploads/{job_id}/{image_path.name}",
            "ply_url": f"/outputs/{job_id}/{ply.relative_to(out_dir).as_posix()}",
            "ply_size": ply.stat().st_size,
            "elapsed_sec": round(elapsed, 2),
        }
    )


app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
