.PHONY: setup setup-dev download pipeline test

setup:
	python3 src/setup/setup_env.py

setup-dev:
	python3 src/setup/setup_env.py --dev

download:
	python3 src/ingestion/download_dataset.py

pipeline:
	python3 src/pipeline/data_pipeline.py

test:
	python3 -m pytest tests
