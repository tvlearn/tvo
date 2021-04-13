FROM continuumio/miniconda3

COPY env_for_gitlab_img.yml environment.yml
RUN conda env create
SHELL ["conda", "run", "-n", "tvem", "/bin/bash", "-c"]
RUN conda install -c conda-forge sphinx
RUN conda clean --all -y -q
