.PHONY: setup setup-dev download pipeline test label compact-labels

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

label:
	python3 src/labeling/server.py $(if $(LABELER),--labeler $(LABELER))

compact-labels:
	python3 src/labeling/labels.py --compact
