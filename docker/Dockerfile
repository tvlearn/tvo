FROM continuumio/miniconda3

COPY env_for_gitlab_img.yml environment.yml
RUN conda env create
RUN conda clean --all -y -q
