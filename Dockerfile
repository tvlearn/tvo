FROM continuumio/miniconda3

COPY environment.yml .
RUN conda update --all -y -q
RUN conda env create
RUN conda clean --all -y -q
