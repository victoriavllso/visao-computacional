# Rotulagem manual da tabela nutricional

Os valores do Open Food Facts (OFF) que usamos como gabarito para o pipeline do verso têm erros (em uma checagem manual, 1 em cada 4 divergências era erro do OFF, não da leitura). Esta ferramenta permite corrigir o gabarito à mão: para cada produto, você vê a foto da tabela nutricional e digita os valores corretos.

Os rótulos manuais têm prioridade: se um produto foi rotulado à mão, a avaliação usa esses valores; se não, usa os do OFF.

## 1. Preparar (uma vez)

Você precisa do código, do ambiente e das imagens. Enquanto a ferramenta não estiver na `main`, ela está no branch `phase1-data`:

```bash
git fetch
git checkout phase1-data
```

Na raiz do projeto:

```bash
make setup
source .venv/bin/activate
make download
make pipeline
```

`make download` baixa ~8 GB e `make pipeline` baixa ~5 GB de imagens. Se alguém do grupo já tem a pasta `data/`, copiar dela é mais rápido.

Todos precisam usar o mesmo snapshot do OFF (fixado em `HF_REVISION`, 2026-10-18), senão a lista de produtos muda. Se você baixou o dataset antes desta versão, veja "Downloading the Dataset" no README.

## 2. Abrir a ferramenta

```bash
git pull
source .venv/bin/activate
make label
```

Abra <http://localhost:8765> no navegador. Os rótulos de todos vão para um único arquivo, `labels/back/hand-labeled.jsonl`. Cada salvamento acrescenta uma linha; o `git pull` junta as linhas de todos sem conflito (`merge=union` no `.gitattributes`).

## 3. Rotular

Escolha a fila no topo:

| Fila | O que é | Prioridade |
| :--- | :--- | :--- |
| `val` | Conjunto de validação (395 produtos): a nota final honesta do pipeline | **1ª** |
| `dev disagreements` | Produtos do dev em que o pipeline e o OFF discordam | 2ª |
| `dev` | Todo o dev | 3ª |

Para cada produto:

1. Olhe a imagem. **Table crop** mostra a tabela recortada pelo pipeline; se o recorte estiver errado, use **Full photo**. Clique na imagem para ampliar.
2. Marque a base da **primeira coluna numérica**: `100 g` ou `100 ml`.
3. Digite **açúcares adicionados (g)**, **gorduras saturadas (g)** e **sódio (mg)** dessa coluna. Vírgula ou ponto, tanto faz (`8,7`).
   - O valor do OFF aparece embaixo de cada campo; se estiver certo, clique **= OFF**.
   - Linha que não existe na tabela, ou que você não consegue ler: deixe **vazio** (não chute).
   - Exceção: se a tabela diz *"Não contém quantidades significativas de ..."*, os nutrientes listados valem **0** (*açúcares* inclui os adicionados; *gorduras totais* inclui as saturadas).
4. Escolha a situação da foto:

| Situação | Quando | Valores |
| :--- | :--- | :--- |
| **table readable** | Tabela nova (com 100 g/100 ml), legível | Preencha os três |
| **partly readable** | Parte ilegível (reflexo, dobra, corte) | Preencha o que der, deixe o resto vazio |
| **linear text, no table** | Embalagem pequena com o texto corrido ("Por 100 g: ... Açúcares adicionados 23 g ...") | Preencha os três |
| **old format** | Tabela antiga, só por porção, sem coluna de 100 g/100 ml | Deixe vazio |
| **no nutrition table / unreadable** | A foto não é da tabela (código de barras, frente...) ou é ilegível | Deixe vazio |

5. **Enter** salva e vai para o próximo. Cada produto é salvo na hora; pode fechar a página quando quiser.

Casos especiais vão no campo **Note**. Exemplo: tabela com colunas de 100 g **e** 100 ml (produto para preparar): preencha a coluna do produto **como vendido** e anote.

"skip already labeled" pula o que você ou outra pessoa já rotulou (de quem você já deu `git pull`). Se o mesmo produto for rotulado por duas pessoas, vale o rótulo mais recente.

## 4. Enviar

Envie com frequência, para os outros não repetirem os mesmos produtos:

```bash
git pull
git add labels/back/
git commit -m "labels: +N produtos"
git push
```

De vez em quando, para tirar as linhas repetidas (produto salvo mais de uma vez), rode `make compact-labels`: reescreve o arquivo com um rótulo por produto (o mais recente) e junta nele outros arquivos `.jsonl` que existirem na pasta.

## Como os rótulos são usados

- `src/labeling/labels.py`: lê os rótulos (linhas corrompidas são ignoradas com um aviso) e decide o gabarito de cada produto (manual se existir, OFF se não).
- `python src/evaluation/back.py --split dev`: avaliação do pipeline do verso; `--hand-only` avalia só os produtos rotulados à mão.
- `make pipeline` (etapa 4): os avisos esperados de cada produto também usam os rótulos manuais.
