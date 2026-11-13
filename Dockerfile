# Environment for rl-lead-optimization.
#
# requirements.txt pins the libraries and data/egfr_chembl.parquet pins the
# input. This file pins the two things neither of those can: the interpreter
# and the OS-level shared libraries underneath it.
#
# It does NOT pin the CPU. A container shares the host's processor, and NumPy
# still selects kernels by the instruction set it finds there, so the same
# image on two different hosts is not guaranteed to produce identical
# floating-point results. See the README's reproducibility notes.

# The published run used CPython 3.13.15. Tags are mutable; for a base that
# cannot move underneath you, replace the tag with the digest reported by
#   docker image inspect --format '{{index .RepoDigests 0}}' python:3.13.15-slim
FROM python:3.13.15-slim

LABEL org.opencontainers.image.title="rl-lead-optimization" \
      org.opencontainers.image.description="RL exploration policies for molecular lead optimization (EGFR / CHEMBL203)" \
      org.opencontainers.image.source="https://github.com/farabhi/rl-lead-optimization" \
      org.opencontainers.image.licenses="MIT"

# RDKit's manylinux wheel leaves its X11-linked modules (the Cairo drawing
# code paths) dynamically linked, and the slim base ships neither library.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      libxrender1 \
      libxext6 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /work

# Copied on its own so a change to the notebook does not re-resolve the
# dependency layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The notebook runner. Deliberately not in requirements.txt: that file lists
# what the notebook imports, which is what the run's environment fingerprint
# reports. These three only execute it.
RUN pip install --no-cache-dir \
      jupyterlab==4.6.6 \
      nbconvert==7.17.2 \
      ipykernel==7.4.0

# Includes data/egfr_chembl.parquet, so a run needs no network at all.
COPY . ./

RUN useradd --create-home --uid 1000 runner \
 && chown -R runner:runner /work
USER runner

EXPOSE 8888

# Default: execute the notebook end to end (~2.5 h) and write the executed
# copy plus the three figures beside the source. Bind-mount the checkout over
# /work to get them out onto the host:
#
#   docker run --rm -v "$PWD:/work" rl-lead-optimization
#
# Interactive instead:
#
#   docker run --rm -it -p 8888:8888 -v "$PWD:/work" rl-lead-optimization \
#     jupyter lab --ip 0.0.0.0 --no-browser
CMD ["jupyter", "nbconvert", \
     "--to", "notebook", \
     "--execute", \
     "--ExecutePreprocessor.timeout=-1", \
     "--output", "rl_lead_optimization.executed.ipynb", \
     "rl_lead_optimization.ipynb"]
