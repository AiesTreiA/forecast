.PHONY: help install run data test clean

PYTHON ?= $(shell which python3)
VENV ?= .venv
STREAMLIT ?= $(VENV)/bin/streamlit

help:
	@echo "Comandos disponibles:"
	@echo "  make install  - Instala dependencias en el entorno virtual"
	@echo "  make data     - Genera dataset sintético retail en data/"
	@echo "  make run      - Inicia la aplicación Streamlit (Holt-Winters+)"
	@echo "  make test     - Ejecuta las pruebas unitarias y de integración"
	@echo "  make clean    - Elimina archivos temporales y cache"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

data:
	$(VENV)/bin/python -m core.data_generator --dataset retail

castano:
	$(VENV)/bin/python -m core.data_generator --dataset castano

run:
	@if [ -f "$(STREAMLIT)" ]; then \
		$(STREAMLIT) run app/Home.py --server.port=8501 --server.address=0.0.0.0; \
	else \
		streamlit run app/Home.py --server.port=8501 --server.address=0.0.0.0; \
	fi

test:
	$(VENV)/bin/python -m pytest tests/ -v

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
