"""
MarketMinds — PyInstaller Build Script

Creates a portable Windows application that:
  - Bundles frontend, backend, and data
  - Stores user data (API keys, vectors) in %LOCALAPPDATA%\MarketMinds
  - Requires NO Python, pip, or any developer tools on the target machine
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def build():
    print("=" * 60)
    print("  MarketMinds — Build Script")
    print("=" * 60)

    # Step 1: Build the PyInstaller command
    print("\n[1/2] Assembling PyInstaller command...")

    sep = os.pathsep  # ';' on Windows

    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",   # Keep console visible so user can see startup logs

        # Bundle application code and assets
        "--add-data", f"frontend{sep}frontend",
        "--add-data", f"backend{sep}backend",
        "--add-data", f"data{sep}data",

        # ── Hidden imports: uvicorn internals ──
        "--hidden-import", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.http.h11_impl",
        "--hidden-import", "uvicorn.protocols.http.httptools_impl",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "uvicorn.lifespan.off",

        # ── Hidden imports: FastAPI / Starlette ──
        "--hidden-import", "fastapi.middleware",
        "--hidden-import", "fastapi.middleware.cors",
        "--hidden-import", "fastapi.staticfiles",
        "--hidden-import", "starlette.responses",
        "--hidden-import", "starlette.routing",
        "--hidden-import", "starlette.middleware",
        "--hidden-import", "starlette.middleware.cors",
        "--hidden-import", "starlette.staticfiles",
        "--hidden-import", "multipart",
        "--hidden-import", "python-multipart",

        # ── Hidden imports: AI / ML ──
        "--hidden-import", "pypdf",
        "--hidden-import", "pypdf._readers",

        # ── Hidden imports: Google Gemini ──
        "--hidden-import", "google.generativeai",
        "--hidden-import", "google.ai",
        "--hidden-import", "google.api_core",

        # ── Hidden imports: Finance ──
        "--hidden-import", "yfinance",

        # ── SSL certificates (required for yfinance/requests HTTPS in frozen exe) ──
        "--hidden-import", "certifi",
        "--collect-data", "certifi",

        # ── Hidden imports: Other ──
        "--hidden-import", "dotenv",
        "--hidden-import", "pydantic",

        # Entry point
        "launcher.py",
    ]

    # Step 2: Run the build
    print("\n[2/2] Running PyInstaller (this takes 2-5 minutes)...\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[ERROR] Build failed! Check the output above for details.")
        sys.exit(1)

    # Post-build info
    dist_dir = Path("dist") / "launcher"
    exe_path = dist_dir / "launcher.exe"

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print("  BUILD SUCCESSFUL!")
        print("=" * 60)
        print(f"  Location:  {dist_dir.resolve()}")
        print(f"  EXE size:  {size_mb:.1f} MB")
        print()
        print("  To distribute:")
        print(f"    1. Zip the entire '{dist_dir}' folder")
        print("    2. Send the zip to your users")
        print("    3. They unzip and double-click launcher.exe")
        print()
        print("  On first run, a config folder is created at:")
        print("    %LOCALAPPDATA%\\MarketMinds\\.env")
        print()
        print("  Users must configure:")
        print("    • Gemini: Set GEMINI_API_KEY in .env (get key at aistudio.google.com)")
        print("=" * 60)
    else:
        print("\n[WARNING] Build completed but launcher.exe was not found.")


if __name__ == "__main__":
    build()
