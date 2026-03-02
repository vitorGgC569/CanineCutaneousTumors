# Discriminação do Projeto

## Resumo

As neoplasias cutâneas em cães representam um desafio diagnóstico significativo na clínica veterinária, exigindo métodos precisos para o manejo adequado e bem-estar animal. A Inteligência Artificial (IA), particularmente o Aprendizado Profundo (Deep Learning - DL), tem demonstrado um potencial transformador no diagnóstico por imagem em patologia. Este projeto visa investigar a aplicação e validar o desempenho de arquiteturas de IA como ferramenta de suporte ao diagnóstico e classificação de tumores cutâneos caninos, fundamentado nos achados e metodologias aplicadas ao dataset CATCH. A pesquisa se concentrará na utilização de imagens histopatológicas de lâminas inteiras (WSIs) do dataset CATCH, que abrange sete subtipos comuns de tumores cutâneos caninos. A pesquisa utilizará como base os modelos originais (UNet com ResNet18 e EfficientNet-B0) descritos no estudo do dataset CATCH, evoluindo para a exploração de abordagens do estado-da-arte, especificamente Aprendizado de Múltiplas Instâncias (Multiple Instance Learning - MIL) utilizando a arquitetura CLAM combinada com extração de características por Foundation Models (ViT/Swin Transformer). Buscar-se-á validar e comparar o desempenho destas diferentes abordagens na identificação e classificação de subtipos tumorais cutâneos.

**Palavras-Chaves**: histopatologia; cães; IA; CNN; detecção; câncer cutâneo

## Introdução

As afecções do sistema tegumentar estão entre as causas mais comuns de consulta na clínica veterinária, podendo chegar até 15,23% dos atendimentos, atrás apenas de sistema reprodutivo (NASCIMENTO et al., 2022). Condições ligadas ao sistema tegumentar englobam um vasto espectro, vão desde processos inflamatórios até desordens alérgicas, autoimunes e neoplasias.

Os tumores tegumentares são de particular importância devido a sua prevalência, e também ao seu impacto na saúde e expectativa de vida dos animais. Um estudo retrospectivo da Universidade Federal de Santa Maria, revisou 761 tumores cutâneos analisados entre 1964 e 2003. Os tumores mais prevalentes foram: mastocitoma (20,9%), carcinoma de células escamosas (7,0%), adenoma perianal ( 5,8%) e lipoma (5,5%) (MELLO DE SOUZA et al., 2006). De forma complementar, uma análise de 727 casos de neoplasias cutâneas em cães, realizada em Fortaleza, CE, entre 2003 e 2010, identificou o carcinoma de células escamosas como o segundo mais frequente, representando 15,26% dos tumores, seguido pelo mastocitoma com 11,69%, o mais frequente foi o lipoma com 18,56 (BASTOS et al., 2017).

O diagnóstico preciso destes tumores, é fundamental para o tratamento adequado, sendo que a histopatologia, que é o exame microscópico de tecidos biológicos, é considerado o padrão ouro, com exceção do caso de linfomas (FONTI & MILLANTA, 2025). E no processo de análise das células, a Inteligência Artificial (IA), especialmente o Aprendizado Profundo, tem emergido como uma tecnologia promissora. Algoritmos como as Redes Neurais Convolucionais (CNNs), têm demonstrado boa capacidade na análise de imagens médicas, incluindo lâminas histopatológicas completas (WSIs), que são fundamentais para o diagnóstico oncológico (DANG et al., 2025). A aplicação da IA a estas tarefas visa diminuir o tempo de análise, aumentar a objetividade e auxiliar na identificação de características prognósticas, servindo como uma ferramenta de suporte importante para patologistas. Trabalhos como os de Salvi et al. (2021) demonstram avanços na classificação de tumores de células redondas, e o dataset CATCH e os modelos nele baseados propostos por Wilm et al. (2022) expandem essa capacidade para um espectro mais amplo de tumores cutâneos.

Assim, este projeto propõe a investigação da aplicação e a validação do desempenho de sistemas de IA para apoiar o diagnóstico e a classificação de tumores cutâneos caninos. Especificamente, serão adotados os sete subtipos representados no dataset CATCH (melanoma, mastocitoma, carcinoma de células escamosas, tumor da bainha do nervo periférico, tricoblastoma, histiocitoma e plasmocitoma). Embora a IA também venha sendo explorada em outras áreas da dermatologia veterinária, este projeto concentrar-se-á especificamente na sua aplicação para a análise histopatológica de tumores cutâneos. Por fim, é importante salientar que a IA deve ser vista como uma ferramenta de auxílio à decisão, potencializando a capacidade do patologista veterinário.

## Justificativa

O Instituto Federal Goiano (IF Goiano), por meio de seu curso de Medicina Veterinária e do Hospital Veterinário, consolida-se como polo de excelência em ensino, pesquisa e extensão. O estudo e melhoria de tecnologias para auxílio de diagnóstico contribuirá não apenas na formação de profissionais qualificados, mas futuramente poderá ser utilizada no cuidado aos animais assistidos pela instituição, principalmente do programa cão guia, reforçando o compromisso com a inovação e a responsabilidade social.

Outro ponto importante que justifica a realização do presente projeto, é que a precisão no diagnóstico de tumores cutâneos caninos é fundamental para o planejamento terapêutico e o prognóstico (CRMVRJ, 2025; DANG et al., 2025). A interpretação histopatológica pode ser impactada pela variabilidade entre patologistas, especialmente em casos complexos, ou na distinção de subtipos com características morfológicas sobrepostas, em lâminas histopatológicas completas o desafio é ainda maior (NORTHRUP et al., 2005). Rissi e Oliveira (2022) destacam a importância desse tipo de aprendizado para o residente em medicina veterinária. Dessa forma, a Inteligência Artificial surge como uma tecnologia promissora para auxiliar nesses desafios, oferecendo o potencial para análises mais rápidas, objetivas e padronizadas. Para além do diagnóstico, a IA também pode tornar possível prever desfechos clínicos e até mesmo descobrir novos biomarcadores morfológicos (DANG et al., 2025).

A investigação de modelos de IA treinados no dataset CATCH já demonstrou a capacidade de segmentar regiões tumorais e classificar subtipos com alta acurácia (WILM et al., 2022). No entanto, trata-se de uma base relativamente nova, proposta em 2022, contando com poucos estudos e aprofundamento por parte de outros pesquisadores. Parâmetros técnicos chave no treinamento incluem a escolha da resolução dos patches de imagem (diferentes resoluções para segmentação e classificação para otimizar contexto e detalhe, respectivamente), estratégias de amostragem para lidar com desequilíbrio de classes, e funções de perda adequadas para cada tarefa podem ser melhoradas com maiores estudos.

Wilm et al. (2022) utiliza arquitetura UNet com backbone ResNet18 (pré-treinado no ImageNet) para segmentação de Tecidos e Regiões Tumorais, EfficientNet-B0 (também pré-treinado no ImageNet) para classificação de Subtipos Tumorais, com isso, é construído um Pipeline de Inferência otimizado baseado duas etapas, consistindo em segmentação em baixa resolução seguida por classificação em alta resolução apenas nas áreas tumorais identificadas, de forma que novas arquiteturas, e novos pipelines, possivelmente incluindo pré-processamento de imagens possam produzir resultados ainda melhores, assim como observado por Dos Santos et al. (2020).

## Fundamentação Teórica

São diversos os tipos de tumores em cães, sendo que o diagnóstico desses tumores cutâneos depende principalmente da avaliação histopatológica, que envolve a análise da arquitetura tecidual e das características citomorfológicas das células neoplásicas em amostras coradas por Hematoxilina & Eosina (H&E) (FRAGOSO-GARCIA et al., 2023). Um tipo importante é o de tumores de células redondas (RCNs), que referem-se a um grupo de neoplasias que apresentam células de forma arredondada ou oval, geralmente com um núcleo grande e citoplasma escasso. segundo Rissi et al. (2022) as características histológicas dos principais tumores dessa categoria são:

**Tumor de Mastócitos (MCT)**
- Localização: Pode ser cutâneo (na derme) ou subcutâneo (na gordura sob a pele).
- Arranjo celular: As células se organizam em pequenos grupos, cordões ou lâminas que separam as fibras de colágeno da derme ou invadem a gordura subcutânea.
- Características celulares: Células redondas com citoplasma abundante, fracamente corado e com grânulos distintos. O núcleo é redondo com cromatina densa.
- Outras características: Presença de eosinófilos e figuras de "chama" (fibras de colágeno envolvidas por degranulação eosinofílica)

**Plasmocitoma Cutâneo (PCT)**
- Localização: Geralmente bem delimitado na derme, mas pode invadir a gordura subcutânea
- Arranjo celular: As células formam lâminas e agregados sustentados por um delicado estroma fibrovascular
- Características celulares: Células plasmáticas com citoplasma moderado a abundante, geralmente eosinofílico (cor-de-rosa) e homogêneo ou granular. Núcleo redondo a oval, excêntrico, com cromatina densa e, às vezes, com um ou mais nucléolos. Presença comum de células binucleadas ou multinucleadas

**Linfoma Cutâneo**
- Localização: Há dois tipos principais: Epiteliotrópico (infiltra a derme superficial e a junção dermoepidérmica) e Não epiteliotrópico (forma lâminas de células que invadem profundamente a derme e o tecido subcutâneo).
- Arranjo celular: Epiteliotrópico: Agregados distintos ou infiltração difusa; Não epiteliotrópico: Lâminas e agregados celulares.
- Características celulares: Varia muito de acordo com o tipo de linfoma. Os linfócitos neoplásicos geralmente têm pouco citoplasma, margens celulares pouco definidas e núcleos redondos a médios com cromatina densa.

**Histiocitoma**
- Localização: Bem delimitado na derme e, às vezes, no tecido subcutâneo
- Arranjo celular: Células neoplásicas formam cordões soltos perto da junção dermoepidérmica e podem invadir a epiderme
- Características celulares: Células com moderada quantidade de citoplasma eosinofílico e núcleos redondos com cromatina finamente granular e 1-2 nucléolos.
- Outras características: É comum a presença de pequenos linfócitos na base do tumor

**Sarcoma Histiocítico**
- Localização: Pode ser localizado na pele ou ser parte de uma doença disseminada
- Arranjo celular: Células dendríticas neoplásicas se organizam em lâminas ou nódulos sustentados por um estroma fibrovascular escasso
- Características celulares: As células têm abundante citoplasma redondo a poligonal, eosinofílico e homogêneo, ou com vacúolos, e limites celulares distintos. Núcleos redondos a ovais ou indentados, com cromatina finamente granular e um a vários nucléolos.

**Tumor Venéreo Transmissível (TVT)**
- Localização: Geralmente invade a derme e o tecido subcutâneo, podendo raramente invadir a epiderme
- Arranjo celular: As células neoplásicas formam lâminas suportadas por um delicado estroma fibrovascular
- Características celulares: Células redondas a poliédricas com moderada quantidade de citoplasma eosinofílico e finamente vacuolizado. Os núcleos são uniformes, redondos e centrais, com cromatina finamente granular e um nucléolo proeminente

O dataset CATCH fornece uma base de dados pública e robusta para o treinamento e validação de modelos de IA, com 350 WSIs, obtidas por meio de biópsia, de sete subtipos tumorais, e anotações detalhadas de 13 classes histológicas, sendo elas: melanoma, mastocitoma (MCT), carcinoma de Células Escamosas (SCC), tumor da Bainha do Nervo Periférico (PNST), tricoblastoma, histiocitoma, plasmocitoma, osso, cartilagem, derme, epiderme, hipoderme, inflamação/necrose. Osso e cartilagem apresentam pouca relevância no dataset (WILM et al., 2022).

A IA, especificamente o Aprendizado Profundo (DL), utiliza Redes Neurais Convolucionais (CNNs) para aprender automaticamente características hierárquicas a partir de dados visuais. As CNNs são otimizadas para o processamento de dados com estrutura de grade, tais como imagens, e têm demonstrado desempenho do estado da arte em diversas tarefas de visão computacional. A arquitetura hierárquica das CNNs, inspirada no córtex visual biológico, permite a aprendizagem automática de representações de características cada vez mais complexas e abstratas a partir dos dados brutos (DATACAMP, 2024). Conforme destacado por LeCun et al. (1998), no desenvolvimento da LeNet-5, as camadas convolucionais utilizam o compartilhamento de pesos e a conectividade local para extrair características espaciais, seguidas por camadas de subamostragem (pooling) para reduzir a dimensionalidade e conferir invariância a pequenas translações. A introdução de funções de ativação não lineares, como a ReLU (embora LeNet-5 usasse sigmoides) entre essas camadas é crucial para a capacidade do modelo de aprender mapeamentos complexos. Krizhevsky, Sutskever e Hinton (2012, p. 1) com o desenvolvimento da AlexNet, demonstraram o poder das CNNs profundas em tarefas de classificação de imagens em larga escala, como o ImageNet, popularizando seu uso e impulsionando avanços subsequentes na área. A combinação de camadas convolucionais, pooling e camadas totalmente conectadas ao final da rede permite a classificação ou regressão baseada nas características aprendidas.

A eficácia das CNNs reside na sua capacidade de aprender filtros relevantes diretamente dos dados, mitigando a necessidade de engenharia manual de características. Arquiteturas como UNet são eficazes para tarefas de segmentação semântica em imagens biomédicas, enquanto arquiteturas como ResNet (utilizada como backbone) e EfficientNet são poderosas para classificação de imagens. O pré-treinamento em grandes datasets como o ImageNet (transfer learning) é uma técnica comum para melhorar o desempenho e acelerar a convergência dos modelos (WILM et al., 2022; XIAO et al., 2025).

## Objetivo Geral e Objetivos Específicos

Investigar, validar e aprimorar o uso de Inteligência Artificial como ferramenta de suporte para o diagnóstico e classificação dos principais tumores cutâneos em cães, utilizando imagens histopatológicas do dataset CATCH e metodologias de aprendizado profundo.

Objetivos específicos:

a) Analisar e utilizar o dataset CATCH, compreendendo suas 350 WSIs e 12.424 anotações poligonais dos sete subtipos de tumores cutâneos caninos e classes de tecido não neoplásico.

b) Implementar, treinar e validar os modelos baseline (UNet com ResNet18 e EfficientNet-B0) e investigar a aplicação de técnicas no estado-da-arte, com foco em Multiple Instance Learning (MIL) utilizando a arquitetura CLAM e extração de características por Foundation Models (ViT/Swin Transformer).

## Metodologia da Execução do Projeto

Inicialmente será realizada uma pesquisa exploratória buscando uma melhor compreensão sobre datasets, arquiteturas de CNN, histologia e tumores caninos. Após essa etapa, serão realizadas fases sequenciais e interligadas, aproveitando os dados e metodologias associados ao dataset CATCH.

**Fase 1: Aquisição e Preparação do Dataset CATCH**
- Obtenção do dataset CATCH, incluindo as 350 WSIs e as anotações associadas (disponíveis no The Cancer Imaging Archive).
- Familiarização com a estrutura do dataset, as 13 classes histológicas anotadas (7 tumorais, 6 não neoplásicas) e os metadados disponíveis.
- Configuração do ambiente computacional com as bibliotecas necessárias (OpenSlide, fastai, PyTorch/TensorFlow, Numpy, etc).

**Fase 2: Implementação, Treinamento e Investigação de Arquiteturas de IA**
- Pré-processamento de Imagens: Extração de patches das WSIs do CATCH conforme as especificações do estudo original:
    - Para segmentação: Patches de 512x512 pixels a uma resolução de 4px/µm.
    - Para classificação: Patches de 1024x1024 pixels na resolução original de 0.25px/µm.
    - Aplicação de aumento de dados online (rotações, zoom) e normalização de patches.
- Implementação de Arquiteturas de IA de Referência e Exploração de Novas Abordagens:
    - Abordagem Original (Baseline): Segmentação com UNet (backbone ResNet18) e Classificação com EfficientNet-B0, utilizando pesos pré-treinados do ImageNet.
    - Abordagem Moderna (Estado-da-Arte): Investigação da aplicação de Multiple Instance Learning (MIL) através da arquitetura CLAM (Clustering-constrained Attention Multiple instance learning).
    - Extração de Características (Feature Extraction): Utilização de Foundation Models (como CTransPath/Swin Transformer) para geração de embeddings a partir de patches das lâminas, reduzindo a dependência de anotações poligonais finas.
- Treinamento dos Modelos:
    - Utilização da divisão padrão do CATCH: 245 WSIs para treinamento, 35 para validação, 70 para teste, além de outras configurações.
    - Aplicação da estratégia de amostragem adaptativa de patches para segmentação
    - Implementação de amostragem preferencial de patches tumorais para classificação.
    - Utilização das funções de perda: Dice generalizada + focal categórica para segmentação; entropia cruzada para classificação.
    - Ajuste de hiperparâmetros

**Fase 3: Validação e avaliação dos modelos**
- Avaliação quantitativa no conjunto de teste do CATCH:
    - Métricas para Segmentação: coeficiente de Jaccard por classe
    - Métricas para Classificação: acurácia em nível de lâmina, F1-score por subtipo em nível de patch.
- Análise do pipeline: incluir técnicas de pré-processamento com filtro de Wiener e CLAHE.
- Análise de erros: Investigação de casos de classificação incorreta.

**Fase 4: Disseminação.**
- Relatório Final e Publicações: Compilação detalhada da metodologia, resultados da validação, análise dos desafios e desempenho dos modelos. Submissão de artigos para periódicos científicos e apresentação em eventos.

## Viabilidade

Serão realizadas reuniões semanais para acompanhamento do projeto, além de reuniões e conversas com médicos veterinários do IFGoiano - Campus Urutaí.

## Resultados Esperados

Como resultado, espera-se obter a validação e análise de desempenho de modelos de IA para Diagnóstico de Tumores Cutâneos Caninos, principalmente com relação a delimitação de regiões tumorais e de classes histológicas, utilizando como referência o estudo original do CATCH, sendo que isso será quantificado por meio de dados concretos sobre o desempenho, tais como acurácia, coeficiente de Jaccard e F1-score de cada modelo investigado. Tal análise permitirá significativo impacto Científico e Fortalecimento Institucional, considerando seu forte potencial para publicações em periódicos científicos de qualidade e apresentações em eventos. Serão gerados relatórios ao longo da execução do projeto, principalmente sobre os experimentos realizados.

## Referências Bibliográficas

- BASTOS, R. S. C.; FARIAS, K. M. DE; LOPES, C. E. B.; PACHECO, A. C. L.; VIANA, D. A. Estudo retrospectivo de neoplasias cutâneas em cães da região metropolitana de Fortaleza. Revista Brasileira de Higiene e Sanidade Animal, v. 11, n. 1, p. 39-53, 2017.
- CONSELHO REGIONAL DE MEDICINA VETERINÁRIA DO ESTADO DO RIO DE JANEIRO (CRMVRJ). Inteligência Artificial na Medicina Veterinária: Avanços, Desafios e Tendências Futuras. Rio de Janeiro, 19 jan. 2025.
- DANG, C.; QI, Z.; XU, T.; GU, M.; CHEN, J.; WU, J.; LIN, Y.; QI, X. Deep learning–powered whole slide image analysis in cancer pathology. Laboratory Investigation, v. 105, n. 7, p. 104186, 2025. DOI: 10.1016/J.LABINV.2025.104186.
- DATACAMP. Introdução a Redes Neurais Convolucionais (CNNs). Disponível em: Https://www.datacamp.com/pt/tutorial/introduction-to-convolutional-neural-networks-cnns. 2024. Acesso em: 06 jan. 2025.
- DIETLER, N. et al. A convolutional neural network segments yeast microscopy images with high accuracy. Nature Communications, v. 11, n. 1, p. 5717, 2020.
- DOS SANTOS, J. C. M.; CARRIJO, G. A.; SANTOS, C. F.; FERREIRA, J. C.; SOUSA, P. M.; PATROCÍNIO, A. C. Fundus image quality enhancement for blood vessel detection via a neural network using CLAHE and Wiener filter. Brazilian Journal of Medical and Biological Research, v. 1, p. 107–119, 2020. DOI: 10.1007/s42600-020-00046-y.
- FONTI, N.; MILLANTA, F. Cancer registration in dogs and cats: a narrative review of history, current status, and standardization efforts. Research in Veterinary Science, v. 191, 2025. DOI: 10.1016/j.rvsc.2025.105673.
- FRAGOSO-GARCIA, M. et al. Automated diagnosis of 7 canine skin tumors using machine learning on H&E-stained whole slide images. Veterinary Pathology,1 v. 60, n. 6, p. 865-875, 2023.
- GALLUCCIO, T. et al. A Complete Transfer Learning-Based Pipeline for Discriminating Between Select Pathogenic Yeasts from Microscopy Photographs. Journal of Fungi, v. 14, n. 5, p. 504, maio 2025.
- GAO, Y. et al. Deep Learning-based Trichoscopic Image Analysis: A Review of Current Applications and Future Directions in Male Androgenetic Alopecia. Skin Research and Technology, v. 28, n. 1, p. 1-11, 2022.
- KIM, M. J. et al. Classification of dog skin diseases using deep learning with images captured from multispectral imaging device. Computers and Electronics in Agriculture, v. 194, p. 106793, 2022.
- KRIZHEVSKY, A.; SUTSKEVER, I.; HINTON, G. E. ImageNet Classification with Deep Convolutional Neural Networks. In: PEREIRA, F.; BURGES, C. J. C.; BOTTOU, L.; WEINBERGER, K. Q. (Eds.). Advances in Neural Information Processing Systems 25. Red Hook, NY: Curran Associates, Inc., 2012. p. 1097-1105.
- KRUPIŃSKI, P. et al. Computer-Aided Cytology Diagnosis in Animals: CNN-Based Image Quality Assessment for Accurate Disease Classification. arXiv preprint arXiv:2308.06055, 2023.
- LECUN, Y.; BOTTOU, L.; BENGIO, Y.; HAFFNER, P. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, v. 86, n. 11, p. 2278-2324, nov. 1998.
- MELLO DE SOUZA, T.; ALMEIDA FIGHERA, R.; IRIGOYEN, L. F.; LOMBARDO DE BARROS, C. S. Estudo retrospectivo de 761 tumores cutâneos em cães. Ciência Rural, v. 36, n. 2, p. 555–560, mar./abr. 2006.
- NASCIMENTO, K. K. F. do; NASCIMENTO, K. K. F. do; KNUPP, S. N. R.; FERNANDES, M. M.; NASCIMENTO, K. K. F. do; SANTOS, F. S. dos. Levantamento retrospectivo da rotina no setor de clínica médica de pequenos animais do HV ASA/IFPB nos anos de 2014 a 2019. Revista Principia - Divulgação Científica e Tecnológica Do IFPB, v. 59, n. 4, p. 1327, 2022. DOI: 10.18265/1517 0306a2021id5810.
- NGUYEN, T. T. et al. Automated Fungal Identification with Deep Learning on Time Lapse Images. MDPI-Journal of Imaging, v. 16, n. 2, p. 109, fev. 2025.
- NORTHRUP, N. C. et al. Variation among pathologists in histologic grading of canine cutaneous mast cell tumors. Journal of Veterinary Diagnostic Investigation, v. 17, n. 3, p. 245–248, maio 2005. DOI: 10.1177/104063870501700305
- PICEK, L. et al. Automatic Fungi Recognition: Deep Learning Meets Mycology. arXiv preprint arXiv:2110.14241, 2022. Publicado como: HEJNÁ, K. et al. FungiVision: A citizen science platform for AI-assisted fungi recognition and observation. PLoS ONE, v. 17, n. 1, e02678142, 2022.
- RISSI, D. R.; OLIVEIRA, F. N. Review of diagnostic histologic features of cutaneous round cell neoplasms in dogs. Journal of Veterinary Diagnostic Investigation, v. 34, n. 5, p. 769–779, set. 2022. DOI: 10.1177/10406387221100209.
- SALVI, M. et al. Histopathological Classification of Canine Cutaneous Round Cell Tumors Using Deep Learning: A Multi-Center Study. Frontiers in Veterinary Science, v. 8, p. 640944, 2021.
- SOARES, A. C. S.; SOUSA, J. S.; SILVA, L. A. F.; LUZ, R. A.; COSTA, F. A. L.; ALVES, F. R. Estudo retrospectivo de diagnósticos histopatológicos de tumores em cães em Teresina, Piauí (2008-2018). Arquivo Brasileiro de Medicina Veterinária e Zootecnia, v. 73, n. 1, p. 77-85, 2021.
- WILM, F. et al. Pan-tumor CAnine cuTaneous Cancer Histology (CATCH) dataset. Scientific Data, v. 9, p. 588, 2022. Disponível em: https://doi.org/10.1038/s41597-022 01692-w.
- WILM, F. et al. CAnine CuTaneous Cancer Histology Dataset (Version 1). The Cancer Imaging Archive, 2022. [Data set]. Disponível em: https://doi.org/10.7937/TCIA.2M93-FX66.
- XIAO, M. et al. Review of applications of deep learning in veterinary diagnostics and animal health. Frontiers in Veterinary Science, v. 12, p. 1511522, 2025.
