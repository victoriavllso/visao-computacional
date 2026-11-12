# Rotulagem manual da tabela nutricional

Os valores do Open Food Facts (OFF) que usamos como gabarito para o pipeline do verso têm erros (em uma checagem manual, 1 em cada 4 divergências era erro do OFF, não da leitura). Esta ferramenta permite corrigir o gabarito à mão: para cada produto, você vê a foto da tabela nutricional e digita os valores corretos.

Os rótulos manuais têm prioridade: se um produto foi rotulado à mão, a avaliação usa esses valores; se não, usa os do OFF.

## 1. Preparar (uma vez)

Você precisa do código, do ambiente e das imagens. O ambiente é criado pelo [pixi](https://pixi.sh) (instalação no README). Na raiz do projeto:

```bash
pixi install
pixi run download
pixi run pipeline
```

`pixi run download` baixa ~8 GB e `pixi run pipeline` baixa ~5 GB de imagens. Se alguém do grupo já tem a pasta `data/`, copiar dela é mais rápido.

Todos precisam usar o mesmo snapshot do OFF (fixado em `HF_REVISION`, 2026-10-18), senão a lista de produtos muda. Se você baixou o dataset antes desta versão, veja "Downloading the Dataset" no README.

## 2. Abrir a ferramenta

```bash
git pull
pixi run label
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

## 3b. Frente: avisos "ALTO EM"

No topo da página, **front labels →** (ou <http://localhost:8765/front>) abre a rotulagem da foto da frente. Os rótulos vão para `labels/front/hand-labeled.jsonl`. As filas são as mesmas (`val`, `dev`); comece pela `val`.

Para cada produto:

1. Marque os avisos que aparecem na foto: **açúcar adicionado** (`1`), **gordura saturada** (`2`), **sódio** (`3`). Nenhum marcado = a foto não tem aviso.
2. **Arraste sobre a foto** um retângulo em volta de cada painel de aviso (o retângulo preto com a lupa, incluindo o "ALTO EM"). Se os avisos estão num painel só, é uma caixa só. `Z` amplia a foto; **Undo box** desfaz a última caixa. As caixas servem para treinar o detector, então capriche no enquadramento.
3. Escolha a situação da foto:

| Situação | Quando |
| :--- | :--- |
| **front of the package** | A frente da embalagem aparece (com ou sem aviso) |
| **a warning is there, but which one can't be read** | Dá para ver o painel, mas não ler quais avisos (borrado, cortado, pequeno demais). Desenhe a caixa mesmo assim |
| **not the front / package not visible** | A foto não é da frente (verso, lateral, outro objeto) |

4. **Enter** salva e vai para o próximo.

Marque só o que está **na foto**, mesmo que você saiba (ou o verso indique) que o produto deveria ter aviso: é exatamente isso que vamos comparar. Por isso a página não mostra nada do verso.

## 4. Enviar

Envie com frequência, para os outros não repetirem os mesmos produtos:

```bash
git pull
git add labels/
git commit -m "labels: +N produtos"
git push
```

De vez em quando, para tirar as linhas repetidas (produto salvo mais de uma vez), rode `pixi run compact-labels`: reescreve o arquivo com um rótulo por produto (o mais recente) e junta nele outros arquivos `.jsonl` que existirem na pasta.

## Como os rótulos são usados

- `src/labeling/labels.py`: lê os rótulos (linhas corrompidas são ignoradas com um aviso) e decide o gabarito de cada produto (manual se existir, OFF se não).
- `pixi run eval --split dev`: avaliação do pipeline do verso; `--hand-only` avalia só os produtos rotulados à mão.
- `pixi run pipeline` (etapa 4): os avisos esperados de cada produto também usam os rótulos manuais.
- `pixi run eval-front --split dev [--method yolo]`: avaliação do pipeline da frente contra os rótulos da frente (avisos marcados).
- `pixi run -e detector train-detector`: as caixas dos painéis (só da `dev`) treinam o detector de avisos; fotos da frente sem caixa são exemplos negativos, por isso um aviso visível precisa de caixa.
