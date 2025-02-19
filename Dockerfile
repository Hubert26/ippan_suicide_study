FROM continuumio/miniconda3

WORKDIR /workspace

COPY environment.yml /workspace

RUN conda env update --name base --file environment.yml

CMD ["/bin/bash"]
