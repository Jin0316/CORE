# =============================================================================
# Docker image for CORE_2026_CVPR_v2 (Which Concepts to Forget? and How to Refuse? Continual Unlnearing for Large Vision-Language Models, CVPR 2026).
#
# Builds the `core` conda environment (Python 3.9, torch 1.13.0+cu117 and all
# pip/conda deps) from environment.yml. torch's wheel bundles its own CUDA 11.7
# runtime, so only the host NVIDIA driver is needed at run time (--gpus all).
#
# All hardcoded paths in the code now resolve under /workspace, which is the
# repo root (2026_CVPR_CORE) bind-mounted at run time:
#   /workspace/CORE_2026_CVPR_v2   <- code (working dir)
#   /workspace/datasets         <- datasets
#   /workspace/pretrained       <- pretrained MiniGPT-4 weights
#   /workspace/Vicuna-7b        <- Vicuna-7B weights
#
# Build:  docker build -t core2026:core CORE_2026_CVPR_v2
# Run:    docker run --gpus all -it -v /path/to/2026_CVPR_CORE:/workspace core2026:core
# =============================================================================
FROM continuumio/miniconda3:24.1.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# System libraries required by opencv / decord / ffmpeg / matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create the conda environment from the project's environment.yml.

## We split environment.yml into a conda-only env (drop the `pip:` subsection)
## and the pip requirements (the 6-space-indented entries).

## The only real dependency conflict is salesforce-lavis 1.0.2, which pins
## transformers<4.27 while the env uses transformers==4.28.0. So we install
## everything *except* lavis with pip's normal resolver (this also fills in
## transitive deps the export dropped, e.g. `click`), then add lavis with
## --no-deps to keep the pinned transformers. Two more entries can't be resolved
## from PyPI by version and are handled separately:
##   - en-core-web-sm==3.7.1  -> spaCy model, fetched via `spacy download`
##   - clip==1.0              -> OpenAI CLIP, installed from its GitHub repo
COPY environment.yml /tmp/environment.yml
RUN grep -vE '^      - |^  - pip:' /tmp/environment.yml > /tmp/environment.conda.yml && \
    grep -E '^      - ' /tmp/environment.yml | sed 's/^      - //' \
        | grep -vE 'en-core-web-sm|clip==1\.0|salesforce-lavis' > /tmp/requirements.txt && \
    conda env create -f /tmp/environment.conda.yml && \
    conda run -n core pip install -r /tmp/requirements.txt && \
    conda run -n core pip install --no-deps salesforce-lavis==1.0.2 && \
    conda run -n core pip install --no-deps git+https://github.com/openai/CLIP.git && \
    conda run -n core python -m spacy download en_core_web_sm && \
    conda clean -afy

# Activate the env by default for every shell / command.
ENV PATH=/opt/conda/envs/core/bin:$PATH \
    CONDA_DEFAULT_ENV=core

# Make interactive shells activate `core` instead of `base` (the miniconda
# .bashrc activates base otherwise, shadowing the env above).
RUN conda init bash && echo "conda activate core" >> /root/.bashrc

# Code lives here at run time (paths in the code are relative to this dir).
WORKDIR /workspace/CORE_2026_CVPR_v2
CMD ["bash"]
