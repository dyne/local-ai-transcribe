# -*- mode: python ; coding: utf-8 -*-
"""Freeze the file-only CLI and its dynamic inference/download dependencies."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

runtime_packages = ['openvino', 'openvino_genai', 'openvino_tokenizers', 'hf_xet']
datas = []
binaries = []
for package in runtime_packages:
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)

a = Analysis(
    ['local-ai-transcribe.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['av', 'noisereduce', 'webrtcvad', 'hf_xet', 'tqdm'],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    # Optional backends discovered through Hugging Face are outside this CLI.
    excludes=['fastapi', 'uvicorn', 'websockets', 'webview', 'faiss',
              'sounddevice', 'torch', 'tensorflow', 'matplotlib', 'tkinter', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='local-ai-transcribe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
