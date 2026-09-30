.PHONY: setup setup-dev download pipeline

setup:
	python3 src/setup/setup_env.py

setup-dev:
	python3 src/setup/setup_env.py --dev

download:
	python3 src/setup/download_dataset.py

pipeline:
	python3 src/pipeline/pipeline.py
