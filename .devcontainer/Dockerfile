ARG type=base

FROM blwalker/devcon as env-base
WORKDIR /tmp
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV CONDA_PKGS_DIRS=/opt/conda/pkgs
SHELL ["bash", "-lc"] 
RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=bind,source=environment.yml,target=environment.yml \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    micromamba install -n base -y -f environment.yml git openssh -c conda-forge 
RUN eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    Rscript -e 'devtools::install_github("sqjin/CellChat")'


FROM env-base as env-gpu
RUN --mount=type=cache,target=/tmp/.cache/pip \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    XDG_CACHE_DIR=/tmp/.cache/ pip install spagcn


FROM env-${type} as env


FROM blwalker/devcon as dev
COPY --from=env --chmod=777 /opt/conda /opt/conda
USER vscode
SHELL ["bash", "-lc"] 
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN /usr/local/bin/micromamba shell init --shell=bash && \
    echo "micromamba activate" >> /home/vscode/.bashrc


FROM dev as repo
USER root
SHELL ["bash", "-lc"] 
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV CONDA_PKGS_DIRS=/opt/conda/pkgs
WORKDIR /tmp/nest-dev
RUN --mount=type=bind,source=${pwd},target=/tmp/nest-dev \
    eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    mkdir /workspaces && \
    cd /workspaces && \
    git clone --no-hardlinks /tmp/nest-dev && \
    cd nest-dev && \
    git checkout --force


FROM dev
COPY --from=repo --chmod=777 /workspaces /workspaces
WORKDIR /workspaces/nest-dev
RUN eval "$(micromamba shell hook --shell bash)" && \
    micromamba activate && \
    pip install -e .