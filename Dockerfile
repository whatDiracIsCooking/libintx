# The CPU image: everything Dockerfile.cuda has except CUDA.
#
# It is not a lesser image in the way the equivalent file in the project this
# devtools set came from is -- there, the light image could not build the C++
# tree at all. Here it builds and tests the whole HOST half of libintx: boys,
# md2/md3/md4, the host K engine and their doctest suites, with
#
#   cmake --preset workstation && cmake --build --preset workstation
#   ctest --preset workstation
#
# What it cannot do is anything under src/libintx/gpu -- libintx.gpu, the
# device md3/md4 engines, the J engine and the device K engine all need nvcc
# and, for the `default` preset, a live device at configure time. Use
# Dockerfile.cuda for those (DEVCONTAINER_CONFIG in devtools/config.sh already
# points there).
FROM ubuntu:24.04

# Stamped on the image so devtools/worktree.sh can tell THIS project's
# per-worktree build images from another repo's. The devcontainer CLI tags them
# `vsc-<worktree>-<hash>`, which is not project-scoped; without this label a gc
# in one checkout would happily delete another repo's. Must match PROJECT_NAME
# in devtools/config.sh.
ARG PROJECT_NAME=libintx
LABEL devcontainer.project="${PROJECT_NAME}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# liblapacke-dev is what keeps libintx.blas from failing at LINK time on
# LAPACKE_dsygvd; see the longer note in Dockerfile.cuda.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        ninja-build \
        ccache \
        git \
        curl \
        ca-certificates \
        less \
        vim \
        sudo \
        libopenblas-dev \
        liblapacke-dev \
        python3 \
        python3-venv \
        python3-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/trusted.gpg.d/githubcli.gpg && \
    echo "deb [arch=$(dpkg --print-architecture)] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    ( apt-get update && apt-get install -y --no-install-recommends gh && \
      rm -rf /var/lib/apt/lists/* ) || \
    echo "WARN: gh unavailable; the PR flow will not work in this image"

RUN install -d -o 1000 -g 1000 \
        /workspace \
        /home/ubuntu/.ccache \
        /home/ubuntu/.claude/plugins \
        /home/ubuntu/.claude/backups && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu

ENV CCACHE_DIR=/home/ubuntu/.ccache

WORKDIR /workspace
USER ubuntu

RUN g++ --version | head -1 && \
    cmake --version | head -1 && \
    ninja --version && \
    test -f /usr/include/lapacke.h

CMD ["/bin/bash"]
