SHELL := /bin/sh
ifeq ($(OS),Windows_NT)
HOST_PYTHON ?= py -3.11
VENV_PYTHON := .venv/Scripts/python.exe
else
HOST_PYTHON ?= python3.11
VENV_PYTHON := .venv/bin/python
endif
PYTHON ?= $(VENV_PYTHON)
PIP := $(PYTHON) -m pip
APP_NAME := local-ai-transcribe
SCRIPT := $(APP_NAME).py
SPEC := $(APP_NAME).spec
REQUIREMENTS := numpy noisereduce webrtcvad-wheels av huggingface_hub[hf_xet] hf_xet
OPENVINO_PACKAGES := openvino openvino-genai openvino-tokenizers

.PHONY: all install install-build test build run clean
all: install

$(VENV_PYTHON):
	$(HOST_PYTHON) -m venv .venv

install: $(VENV_PYTHON)
	$(PIP) install $(REQUIREMENTS) $(OPENVINO_PACKAGES)

install-build: $(VENV_PYTHON)
	$(PIP) install pyinstaller

test:
	$(PYTHON) -m pytest

build: install install-build
	$(PYTHON) -m PyInstaller --clean $(SPEC)

run:
	$(PYTHON) $(SCRIPT) $(ARGS)

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('build', ignore_errors=True); shutil.rmtree('dist', ignore_errors=True)"
