FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml /app

RUN conda env update --name base --file environment.yml

CMD ["/bin/bash"]
