IMAGE   ?= visao-computacional
LABELER ?=
SPLIT   ?= val
FLAGS   ?= --hand-only

# Volume mounts reused across targets
DATA_VOL   = -v $(CURDIR)/data:/app/data
LABELS_VOL = -v $(CURDIR)/labels:/app/labels

.PHONY: setup setup-dev download pipeline test label compact-labels \
        docker docker-build docker-down docker-download docker-pipeline \
        docker-test docker-label docker-compact-labels \
        clean clean-docker clean-all

# Local targets

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

# Removes local runtime artifacts; leaves labels/ intact (versioned)
clean:
	rm -rf data/ .venv/

# Docker targets

# Convenience: build -> download -> pipeline in one shot
docker: docker-build docker-download docker-pipeline

docker-build:
	docker build -t $(IMAGE) .

docker-download:
	docker run --rm $(DATA_VOL) $(IMAGE) \
		python src/ingestion/download_dataset.py

docker-pipeline:
	docker run --rm $(DATA_VOL) $(LABELS_VOL) $(IMAGE) \
		python src/pipeline/data_pipeline.py

docker-test:
	docker run --rm $(IMAGE) python -m pytest tests

docker-label:
	docker run --rm -p 8765:8765 $(DATA_VOL) $(LABELS_VOL) $(IMAGE) \
		python src/labeling/server.py --host 0.0.0.0 $(if $(LABELER),--labeler $(LABELER))

docker-compact-labels:
	docker run --rm $(LABELS_VOL) $(IMAGE) \
		python src/labeling/labels.py --compact

docker-eval-back:
	docker run --rm $(DATA_VOL) $(LABELS_VOL) $(IMAGE) \
		python src/evaluation/back.py --split $(SPLIT) $(FLAGS)

# Removes the image and any dangling build cache for it
clean-docker: docker-down
	docker rmi -f $(IMAGE)
	docker builder prune -f --filter label=image=$(IMAGE)

# Removes everything: local artifacts + docker image/cache
clean-all: clean clean-docker
