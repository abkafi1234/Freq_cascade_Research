# Setting up a new machine

Tested on the original machine: Windows 11, Python 3.12.10, NVIDIA RTX 4070 (12 GB), CUDA 12.8.
Linux works too; the commands are the same except where noted.

## 1. Python

Python **3.12** (3.10+ works, but 3.12 matches the results). Use a fresh virtual environment:

```bash
cd KBS
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. PyTorch, matched to the GPU driver

The results were produced with `torch==2.11.0+cu128`. Check the driver with `nvidia-smi`
(top-right shows the highest CUDA version it supports), then:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

If the driver is older than CUDA 12.8, pick the matching index from https://pytorch.org
(e.g. `cu126`) and note the change in `setup/environment.txt`. The reproduction check (step 5)
tells you whether neural results still match.

## 3. Pinned libraries and the package

```bash
pip install -r setup/requirements.txt
pip install -e freqcascade
```

## 4. Verify

```bash
python run/resume.py --verify
```

It must end with `VERIFY: ALL OK`. It loads all eight corpora **offline** from
`freqcascade/data/raw/`, so no dataset downloads are needed. Model weights (DeBERTa-v3-base for R7,
about 740 MB) download on first use into `freqcascade/data/raw/hf_home/`, so that step needs
internet.

## 5. Reproduction check

```bash
python run/resume.py --check
```

Compares two recomputed cells with stored values. The policy on a mismatch is in `KBS/CLAUDE.md`.

## Running scripts by hand

`run/resume.py` sets these environment variables for every step. If you run a script directly from
`freqcascade/`, set them yourself so the bundled data is used:

```bash
# Linux / Git Bash, from the KBS folder
export FREQCASCADE_DATA="$PWD/freqcascade/data/raw/freqcascade_cache"
export SCIKIT_LEARN_DATA="$PWD/freqcascade/data/raw/scikit_learn_data"
export NLTK_DATA="$PWD/freqcascade/data/raw/nltk_data"
export HF_HOME="$PWD/freqcascade/data/raw/hf_home"
```

```powershell
# Windows PowerShell, from the KBS folder
$env:FREQCASCADE_DATA  = "$PWD\freqcascade\data\raw\freqcascade_cache"
$env:SCIKIT_LEARN_DATA = "$PWD\freqcascade\data\raw\scikit_learn_data"
$env:NLTK_DATA         = "$PWD\freqcascade\data\raw\nltk_data"
$env:HF_HOME           = "$PWD\freqcascade\data\raw\hf_home"
```

## Building the paper

TeX Live or MiKTeX with `latexmk`: `latexmk -pdf main.tex` in the KBS folder.
