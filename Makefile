# Shortcuts for the pixi tasks (pixi.toml), for those used to make.
.PHONY: setup setup-dev download pipeline test label compact-labels

setup:
	pixi install

setup-dev:
	pixi install --environment dev
	pixi run --environment dev hooks

download:
	pixi run download

pipeline:
	pixi run pipeline

test:
	pixi run test

label:
	pixi run label $(if $(LABELER),--labeler $(LABELER))

compact-labels:
	pixi run compact-labels
