<h1 align="center">O.X.T.A Pathology</h1>
<h4 align="center">Diagnóstico Automatizado de Neoplasias Cutâneas Caninas via Deep Learning</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Pesquisa_e_Desenvolvimento-blue.svg" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-Multi--Stage-2496ed.svg" alt="Docker">
  <img src="https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg" alt="License">
</p>

## Sumário
1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Dataset CATCH e Arquitetura Baseline](#dataset-catch-e-arquitetura-baseline)
3. [Otimizações Implementadas na Baseline](#otimizações-implementadas-na-baseline)
4. [Abordagem do Estado-da-Arte: Multiple Instance Learning](#abordagem-do-estado-da-arte-multiple-instance-learning)
5. [Reprodutibilidade e Arquitetura Docker](#reprodutibilidade-e-arquitetura-docker)
6. [Estrutura do Repositório](#estrutura-do-repositório)
7. [Referências e Agradecimentos](#referências-e-agradecimentos)

---

## 1. Visão Geral do Projeto

Este repositório documenta a pesquisa em patologia digital veterinária, com foco na implementação de algoritmos de aprendizado profundo (Deep Learning) para o diagnóstico automatizado de neoplasias cutâneas em cães.

O diagnóstico histopatológico, considerado o padrão-ouro na medicina veterinária, baseia-se na análise microscópica de tecidos corados por Hematoxilina e Eosina (H&E). A avaliação manual de Imagens de Lâminas Inteiras (Whole Slide Images - WSIs) é um processo exaustivo e sujeito à variabilidade inter e intra-observador. O objetivo deste projeto é desenvolver um fluxo computacional reprodível e de alto desempenho que sirva como ferramenta de suporte à decisão para patologistas veterinários na detecção, segmentação e classificação de tumores.

A arquitetura do repositório está subdividida em duas metodologias:
* **Baseline Supervisionada (`CATCH/`)**: Baseada na metodologia original do dataset CATCH, utilizando segmentação pixel a pixel.
* **Estado-da-Arte (`modern/`)**: Pesquisa atual focada no uso de *Multiple Instance Learning* (MIL) e *Foundation Models* visando mitigar a dependência de anotações minuciosas.

## 2. Dataset CATCH e Arquitetura Baseline

A abordagem tradicional em visão computacional médica exige anotações poligonais detalhadas por especialistas, demarcando regiões específicas nas WSIs. Em 2022, o **Dataset CATCH** (CAnine cuTaneous Cancer Histology) foi disponibilizado publicamente, fornecendo 350 WSIs com resoluções elevadas e anotações rigorosas correspondentes a sete subtipos tumorais caninos: Melanoma, Mastocitoma (MCT), Carcinoma de Células Escamosas (SCC), Tumor da Bainha do Nervo Periférico (PNST), Tricoblastoma, Histiocitoma e Plasmocitoma.

A adoção dos scripts originais do dataset CATCH como baseline metodológica justifica-se por:
1. **Validação na Literatura**: A metodologia já possui métricas aferidas e revisadas por pares, estabelecendo um ponto de controle para a experimentação atual.
2. **Arquitetura em Duas Etapas**: O modelo divide-se logicamente em segmentação de tecido e tumor (através de uma arquitetura UNet com ResNet18) e classificação subsequente de subtipos tumorais em *patches* utilizando EfficientNet-B0.
3. **Plataforma Estruturada**: Baseada na biblioteca `fast.ai`, provendo laços de treinamento padronizados para patologia digital.

Qualquer nova proposta arquitetural, como a implementada no diretório `modern`, deve demonstrar superioridade estatística em relação às métricas estabelecidas por esta baseline.

## 3. Otimizações Implementadas na Baseline

Visando garantir a estabilidade e eficiência computacional na avaliação comparativa, a implementação original baseada em `fast.ai` foi refatorada. Intervenções técnicas foram aplicadas na estrutura original (`CATCH/`) para mitigar lentidões e falhas de treinamento identificadas no decorrer das experimentações de controle.

### 3.1 Correção Numérica Estrita (Função DiceLoss)
A segmentação de áreas teciduais desbalanceadas frequentemente faz uso da métrica Dice (Coeficiente de Sorensen-Dice). Observou-se que na implementação inicial do CATCH a alocação do termo protetor em frações (valor de *epsilon* para prevenir divisões por zero) estava alocada incorretamente no cômputo matricial, propiciando *Gradient Exploding* e consequentemente estabilizando o gradiente em mínimos locais. A correção readequou o cálculo baseando-se no modelo canônico consolidado na literatura (e.g. Milletari et al., 2016):
`dice_score = (2. * (weights * numerator).sum() + eps) / ((weights * denominator).sum() + eps)`

### 3.2 Eliminação de Gargalos CPU-GPU nas Métricas
As métricas IoU (Intersecção sobre União) são fundamentais para avaliar o desempenho do modelo na delimitação do tumor. A implementação original alocava a validação IoU utilizando funções nativas da biblioteca `scikit-learn` (`jaccard_score`). Como consequência, requeria-se a transferência massiva e iterativa de tensores originados na GPU (VRAM) para a CPU (RAM) em cada mini-batch, resultando em um forte gargalo no barramento que elevava exponencialmente os tempos de treinamento. A reestruturação reescreveu todas as funções métricas (background_iou, dermis_iou, tumor_iou) operando lógicas e matrizes puramente nativas e in-place em código PyTorch, proporcionando ganhos de eficiência significativos no tempo computacional por época e eliminando estagnações causadas pelo overhead serial.

### 3.3 Mitigação de Retenção de Memória (Memory Leaks)
Devido ao fatiamento WSI em centenas de milhares de *patches*, os *DataLoaders* empregavam um loop sem declarações estritas para a liberação das pilhas temporais originadas (`temp_rows` e `seg_preds`), conduzindo frequentemente a erros fatais de dimensionamento gráfico (`CUDA Out of Memory`). Implementou-se rotinas determinísticas de deleção em variáveis intermediárias iterativas, estabilizando e contendo a expansão predatória da alocação de memória gráfica durante longas rodadas de inferência.

## 4. Abordagem do Estado-da-Arte: Multiple Instance Learning

No atual estágio evolutivo da pesquisa (`modern/`), buscou-se superar as fortes limitações da supervisão estrita (pixel-a-pixel), cuja dependência em anotações custosas asfixia a escalabilidade de treinamentos massivos com milhares de dados histológicos de lâminas fechadas. Como rota científica principal, explora-se a arquitetura de *Multiple Instance Learning* (MIL) aliada a *Foundation Models*.

### Paradigma MIL
No modelo MIL, as lâminas histológicas imensas não são dissecadas sob rótulos espaciais de cada agrupamento epitelial ou estromal, mas unificadas no escopo analítico como aglomerados de instâncias. Isso traduz-se em repassar todos os pedaços microscópicos em um "Saco" e estampar apenas o diagnóstico final (o Rótulo Patológico do animal) e permitir ao modelo, matematicamente, estimar os pesos relativos e quais regiões atestam a malignidade na área global sem serem ensinados previamente onde se encontravam.

### Extração Analítica via Foundation Models (Swin Transformers)
O pipeline executado em `modern/extract_features.py` realiza a quebra das WSIs, submete os recortes padronizados em algoritmos de unificação visual como a normalização Macenko, e envia as matrizes atômicas para camadas atencionais do **Swin Transformer**. Modelos generalistas com treinamento prévio (CtransPath Proxy) absorveram padrões biológicos estruturais amplos e conseguem retornar uma *feature* vetorizada descritiva (e.g 768 variáveis de alta dimensionalidade).

### Rede CLAM (Clustering-constrained Attention MIL)
O orquestramento de classificação em `modern/train.py` usa os vetores representativos não como pontos únicos em uma Rede Neural Convolucional isolada, mas através de redes de Portão e Atenção definindo classificações de pontuação de significância por mecanismo linear (`Gated Attention Network`). Baseada nas definições de Lu et al. (2021), a rede CLAM correlaciona a atenção gerada sobre conjuntos específicos de estroma neoplásico perante lâminas saudáveis, ponderando por regressão as áreas que consolidam o prognóstico tumoral perante o background microscópico de gorduras ou cortes de epidermes saudáveis. O método exibe grande promessa para superação analítica das redes primitivas.

## 5. Reprodutibilidade e Arquitetura Docker

A adoção de boas práticas na ciência de dados requer ferramentas que garantam reprodutibilidade semântica sob as exatas dependências arquitetadas no período da experimentação, eliminando variáveis em máquinas *host*.

O repositório está configurado utilizando instâncias em contêineres e um arquivo `Dockerfile` na raiz utilizando a estratégia de **Multi-Stage Build**:
1. O estágio primário (**Builder**) instancia ambientes isolados equipados com todas as ferramentas de compilação complexas do sistema Debian/Ubuntu (Build-Essential, Compiladores GNU) instalando sem atritos os *wheels* requeridos para o suporte do OpenSlide, CUDA Runtime Modules, as estruturas da Scipy e Numpy acopladas no Pytorch Vision dentro de um diretório Virtual (VENV).
2. O estágio subsequente (**Runtime**) retira toda a matriz excedente empregadada pra compilação, instalando no container final apenas utilitários de sistema puros e carregando o `VENV` recém purificado originário do *Builder*. Isso garante tamanhos dimensionais de arquivos limitados, máxima segurança e rápida replicação estrutural perante instâncias provisionadas na nuvem para patologia e instâncias de inferência para laboratórios parceiros da pesquisa.

## 6. Estrutura do Repositório

```text
OXTA-Pathology/
├── CATCH/                              # Diretriz Retrospectiva de Segmentação Basal Patológica
│   ├── annotation_conversion/          # Mapeadores conversores de XML para bases SQL e Arrays formatados
│   ├── classification/                 # Pipeline supervisionado Classificador de Imagens UNet 
│   ├── evaluation/                     # Inferência Piramidal e algoritmos de fatiador WSI
│   ├── models/                         # Local de Salvamento Físico Persistido para Instâncias .pth
│   └── segmentation/                   # Pipeline Detetor Tecidual Primário 
├── modern/                             # Diretriz MIL (SOTA - Aprendizado Analítico de Lote Inteiro)
│   ├── models/
│   │   └── clam.py                     # Implementações atencionais (Gated Networks CLAM)
│   ├── utils/
│   │   └── preprocessing.py            # Técnicas e limiares de limpezas de lâminas
│   ├── extract_features.py             # Redutor Multidimensional e extrator proxy Swin-Transformer
│   └── train.py                        # Rede Gated de validação do prognóstico
├── Registro_Otimizacoes_CATCH.md       # Logs de intervenção aplicados contra gargalos de Fast.AI
└── Dockerfile                          # Build Multi-Stages reproduzíveis a arquitetura operacional C/Python
```

## 7. Referências e Agradecimentos

1. **Dataset Original:** Wilm, Frauke, et al. *Pan-tumor CAnine cuTaneous Cancer Histology (CATCH) dataset*. Scientific Data, 2022. Arquétipo central de lâminas avaliadas que fundamentam essa pesquisa retrospectiva.
2. **Arquitetura Base MIL:** Lu, Ming Y., et al. *Data-efficient and weakly supervised computational pathology on whole-slide images*. Nature Biomedical Engineering, 2021. Fundamentação teórica que impulsiona a rota moderna metodológica e as derivações para agrupamentos de atenção e CLAM implementados neste projeto investigativo para diagnósticos da pele epidérmica canina.
