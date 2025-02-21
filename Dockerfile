FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml /app

RUN conda env update --name base --file environment.yml && conda clean -afy

RUN conda install pip

COPY pyproject.toml /app

RUN pip install .

CMD ["/bin/bash"]
