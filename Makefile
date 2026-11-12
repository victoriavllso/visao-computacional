.PHONY: help setup setup-dev download pipeline test label compact-labels

help: ## Exibe os comandos disponíveis
	@echo ""
	@echo "Uso: make <alvo>"
	@echo ""
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup: ## Instala as dependências do projeto
	pixi install

setup-dev: ## Instala dependências de desenvolvimento e configura hooks
	pixi install --environment dev
	pixi run --environment dev hooks

download: ## Baixa os dados necessários
	pixi run download

pipeline: ## Executa o pipeline principal
	pixi run pipeline

test: ## Executa os testes
	pixi run test

label: ## Rotula os dados. Exemplo: make label LABELER=nome
	pixi run label $(if $(LABELER),--labeler $(LABELER))

compact-labels: ## Compacta os rótulos
	pixi run compact-labels

ocr: ## Avalia o modelo OCR (split val, apenas mãos)
	pixi run eval --split val --hand-only