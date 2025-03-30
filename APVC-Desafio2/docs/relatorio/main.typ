#import "setup/template.typ": *
#include "setup/capa.typ"
#import "setup/sourcerer.typ": code
// #import "@preview/sourcerer:0.2.1": code
#show: project
#counter(page).update(1)
#import "@preview/algo:0.3.3": algo, i, d, comment //https://github.com/platformer/typst-algorithms
#import "@preview/tablex:0.0.8": gridx, tablex, rowspanx, colspanx, vlinex, hlinex, cellx

#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages)
#codly(zebra-fill: none)

#set text(lang: "pt", region: "pt")
#show link: underline
#show link: set text(rgb("#004C99"))
#show ref: set text(rgb("#00994C"))
#set heading(numbering: "1.")
#show raw.where(block: false): box.with(
  fill: luma(240),
  inset: (x: 0pt, y: 0pt),
  outset: (y: 3pt),
  radius: 3pt,
)

#import "@preview/oasis-align:0.2.0": *

//#page(numbering:none)[
//  #outline(indent: 2em, depth: 7)  
//  // #outline(target: figure)
//]
#pagebreak()
#counter(page).update(1)

#show heading.where(level:1): set heading(numbering: (l) => {[Parte #l] })

#set list(marker: ([•], [‣], [–]))

= : Preparação dos dados <prep>

Para a realização deste trabalho, foi utilizado o dataset #link("https://www.kaggle.com/c/dogs-vs-cats/data")[Dogs vs. Cats] do Kaggle, com o objetivo de classificar imagens de cães e gatos, abrangendo uma ampla diversidade de raças e géneros. O conjunto de dados apresenta desafios adicionais, como a ausência de um padrão definido, a presença de ruído, variações na orientação, diferenças na iluminação e inversões, fatores que podem dificultar a tarefa de classificação por redes neuronais convolucionais (CNNs). Além disso, as imagens possuem dimensões variadas e, para garantir um formato padronizado, é realizada uma operação de redimensionamento durante a importação, utilizando interpolação bilinear#footnote[https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory], para assegurar que todas as imagens fiquem com a dimensão de 256×256 pixels. A @imagens_dataset apresenta dois exemplos que podem dificultar a capacidade preditiva dos modelos.


#figure(
  grid(
    align: horizon + center,
    columns: 2,
    image("images/cat.836.jpg", width: 47%, height: 14%),
    image("images/dog.303.jpg", width: 60%, height: 10%),
  ),
  caption: [
    Exemplo de imagens do conjunto de dados - com orientação diferente à esquerda e com ruído e divergência de luminosidade à direita 
  ],
)<imagens_dataset>

As imagens foram fornecidas em 2 pastas/conjuntos: um de treino com 1000 imagens, distribuídas equitativamente entre cães e gatos, e outro de validação também com 1000 imagens, que continham 500 gatos e 500 cães. /*O conjunto de treino contém 1.000 imagens, sendo 500 de cães e 500 de gatos.*/ O conjunto de validação foi dividido em dois para criar um conjunto de teste#footnote[https://keras.io/api/data_loading/image/] com o "mesmo" tamanho do de validação, cuja distribuição das classes pode ser vista na @divisao.

#set text(11pt)
#set rect(
  inset: 15pt,
  fill: rgb("fff"),
  width: 10%,
  stroke: 0.1pt,
)

#align(horizon + center)[
  #figure(
  grid(
    columns: (400pt, 120pt), 
    gutter: 5pt,
    rows: 1,
    image("images/divisao_parte1.png", width: 85%),
    
    rect(
      
      [Nota-se que a distribuição do conjunto de validação/teste não é exatamente 50/50 para as duas classes, pois depende do caráter pseudo-aleatório da seed fornecida na divisão da função do keras. A seed fixada foi 777, neste caso.

], width: 100%, radius: (
    left: 5pt,
    right: 5pt,
  ))

  ),
  caption: [
    Distribuição de cães e gatos nos conjuntos de dados
  ],
)<divisao>
]


= : Rede Neuronal convolucional "custom"<parte2>

Nesta fase do desafio vamos utilizar uma rede convolucional "_custom_", ou seja, com camadas convolucionais e _layers_ de _pooling_ definidas e construídas por nós. Para este efeito, dividimos o treino em 5 modelos com particularidades diferentes, a fim de identificar possíveis melhorias entre experiências e a fazer _tuning_ das _features_ da nossa rede. 

Inicialmente, como modelo de referência (_baseline model_), treinamos uma rede neuronal com duas camadas convolucionais: a primeira com 16 filtros de tamanho 5×5 e a segunda com 32 filtros de tamanho 3×3. Após isso, aplicamos uma camada de _max pooling_, seguida de uma terceira camada convolucional com 64 filtros de tamanho 3×3 e outra camada de _max pooling_ para reduzir ainda mais a dimensionalidade e chegar com a informação mais relevante ao final da rede. Para finalizar transformamos tudo num vetor 1D com #link("https://keras.io/api/layers/reshaping_layers/flatten/")[_Flatten_] e utilizamos 2 camadas densas: uma com 128 neurónios e ativação ReLU, seguida da camada de saída com 1 neurónio e ativação sigmoide para o nosso problema de classificação binária (cão/gato).

A partir deste modelo _baseline_ (1) foram feitos incrementos a este modelo em mais 4 fases do treino, sendo estas:

#set enum(numbering: "(1)", start: 2)


+ *Modelo com mais 1 _layer_ convolucional*: Foi adicionada uma camada com 128 filtros de tamanho 3x3 logo após a camada com 64 filtros,  antes do 2º _max pooling_. A inserção desta camada foi sobretudo para aumentar a robustez e diversidade da rede e tentar filtrar a informação relevante;
+ *Adicionado _Dropout_:* Foi adicionada uma camada de #link("https://keras.io/api/layers/regularization_layers/dropout/")[_Dropout_] com uma taxa de 30% antes da camada de _output_, para tentar reduzir/evitar _overfitting_;
+ *Data Augmentation* (DA): Como já foi mencionado anteriormente (@imagens_dataset), as imagens continham propriedades diferentes no que toca a luminosidade, orientação, entre outras. Por esta razão decidimos fazer DA com os seguintes "_augments_": inversão horizontal, rotação aleatória, zoom in/out, ajuste de contraste e ajuste de brilho, todos com variações de até 10%;
+ *Batch Normalization* (BN): O BN serve essencialmente para reduzir a variação na distribuição das ativações das camadas durante a fase de treino (#link("https://www.geeksforgeeks.org/what-is-batch-normalization-in-deep-learning/")[_internal covariate shift_]). Foi utilizado como uma tentativa de acelerar a convergência e melhorar a generalização do modelo, assim procurando mitigar o risco de _overfitting_.


Para se entender melhor as etapas da fase de treino mencionadas, conseguimos observar a arquitetura base da nossa rede, na @arquitetura. A montagem desta arquitetura baseia-se no _baseline_ e tem os incrementos de forma explícita com (N).

#figure(
image("images/ArquiteturaV4.png", width: 100%),
  caption: [Arquitetura da rede "custom" criada \ (2) - Mais uma _layer_ convolucional, (3) - Adição Dropout, (4) - Data Augmentation, (5) - Batch Normalization]
)<arquitetura>

Para as 5 fases, a rede foi compilada com o otimizador #link("https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam")[Adam], com um `learning_rate` de 0.001; com a função de perda `binary_crossentropy`, ideal para problemas de classificação binária (cão ou gato); e por métricas, entre elas a _accuracy_, _precision_ e _recall_. Posteriormente, as 5 redes foram treinadas para 50 _epochs_ e com 3 _callbacks_, entre elas: 

- `BEST_MODEL_CHECKPOINT`: guarda os pesos do modelo com menor valor da função de perda;
- `EARLY_STOPPING`: interrompe o treino quando não há melhorias na função de perda num período de 5 épocas;
- `REDUCE_LR`: reduz o _learning rate_ quando a função de perda na validação não melhora durante 7 épocas consecutivas. Caso não haja melhoria, é reduzido para 50% do valor atual, mas não pode atingir um valor mínimo de $1 times 10^(-6)$.

//Posteriormente, foram gerados mapas de saliência das imagens mal classificadas. Permitindo fazer a análise da relevância dada pelo modelo a determinadas áreas da imagem, que nestes casos podem não ser as mais adequadas.

Com o treino efetuado, foram ainda gerados mapas de saliência com o intuito de analisar quais regiões da imagem o modelo considerou mais relevantes ao tomar as decisões de classificação. Esta análise é crucial para entender possíveis falhas na interpretação do modelo, pois, em alguns casos, a atenção pode estar mais focada em áreas irrelevantes ou enganadoras, como fundos "ruidosos" ou padrões aleatórios, em vez das características distintivas do objeto principal.


// Explicamos uma beca do modelo... classificação binaria entao acaba com activation relu blablabla

// tabela com todas as tentativas de modelo, resultados e explicar com a imagem as fases de data augmentation, assim iamos justificando

#let numberDigits = 3 // 3 fica melhor

#figure(
  tablex(
    columns: 9,
    align: center + horizon,
    header-rows:1,
    auto-lines: false, 
    rowspanx(2)[*Modelo*], vlinex(stroke:0.2mm), 
    colspanx(2)[#text(11pt)[*Tempo de Execução*]], vlinex(stroke:0.2mm), 
    rowspanx(2)[#text(10pt)[*Nº Epochs*]], vlinex(stroke:0.2mm), 
    colspanx(3)[#text(10pt)[*Métricas de Desempenho*]], vlinex(stroke:0.2mm), 
    colspanx(2)[#text(10pt)[*Nº Parâmetros*]], hlinex(stroke:0.2mm),
    
    (), [#text(10pt)[*GPU*#footnote[Para esta fase do projeto foi implementada uma solução para que os modelos corressem na GPU, algo que diminuiu drasticamente o tempo de execução dos mesmos. Para mais detalhes da implementação ver #link("https://github.com/GothamAnalytics-7/APVC/blob/main/APVC-Desafio2/README.md")[README do projeto]. GPU utilizada para a execução dos modelos "custom": #link("https://www.gigabyte.com/Graphics-Card/GV-N4060WF2OC-8GD#kf")[#text("GeForce RTX™ 4060 WINDFORCE OC 8G")]]]], vlinex(stroke:0.2mm), 
    [#text(10pt)[*CPU*#footnote[CPU utilizada para a execução dos modelos "custom": #link("https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-5-7600x.html")[#text("Ryzen™ 5 7600X")]]]], 
    vlinex(stroke:0.2mm), (), (), (), [#text(8pt)[*Precision*]], hlinex(stroke:0.2mm), 
    [#text(8pt)[*Accuracy*]], vlinex(stroke:0.2mm), 
    [#text(8pt)[*Recall*]], 
    
    vlinex(stroke:0.2mm), 
    [#text(8pt)[*Treináveis*]], vlinex(stroke:0.2mm), [#text(8pt)[*Totais*]], 


    [Baseline_model], [29.2s], [3m 44s], vlinex(stroke:0.2mm), [6], vlinex(stroke:0.2mm), 
    [#calc.round(0.647773, digits: numberDigits)], vlinex(stroke:0.2mm), [#calc.round(0.658, digits: numberDigits)], 
    [#calc.round(0.655738, digits: numberDigits)], vlinex(stroke:0.2mm), [8,413,217], [25,239,653],
    hlinex(stroke:0.2mm),
    
    [#text("+") 1 layer #link("https://keras.io/api/layers/convolution_layers/convolution2d/")[conv2D()]],  
    [37.9s], [5m 22s], vlinex(stroke:0.2mm), [7], vlinex(stroke:0.2mm), 
    [#calc.round(0.42915, digits: numberDigits)], [#calc.round(0.592, digits: numberDigits)], 
    [#calc.round(0.627219, digits: numberDigits)], vlinex(stroke:0.2mm), [16,875,681], [50,627,045],
    hlinex(stroke:0.2mm),
    
    [#text("+") Dropout], 
    [52.5s], [5m 50s], vlinex(stroke:0.2mm), [9], vlinex(stroke:0.2mm), 
    [#calc.round(0.696356, digits: numberDigits)], [#calc.round(0.632, digits: numberDigits)], 
    [#calc.round(0.6121, digits: numberDigits)], vlinex(stroke:0.2mm), [16,875,681], [50,627,045],
    hlinex(stroke:0.2mm),
    
    [#text("+") Data augmentation], 
    [2m 23.8s], [22m 57s], vlinex(stroke:0.2mm), [25], vlinex(stroke:0.2mm), 
    [#calc.round(0.753036, digits: numberDigits)], [#calc.round(0.714, digits: numberDigits)], 
    [#calc.round(0.69403, digits: numberDigits)], vlinex(stroke:0.2mm), [16,875,681], [50,627,045],
    hlinex(stroke:0.2mm),
    
    [#text("+") Batch normalization], 
    [4m 16.4s], [~32m], vlinex(stroke:0.2mm), [19], vlinex(stroke:0.2mm), 
    [#calc.round(0.789474, digits: numberDigits)], [#calc.round(0.728, digits: numberDigits)], 
    [#calc.round(0.698925, digits: numberDigits)], vlinex(stroke:0.2mm), [16,876,417], [50,629,989]
  ),
  caption: [Resultados dos modelos com o treino da rede convolucional (_training on 2 components_)],
  kind:table
)<table_results_custom>


Pela @table_results_custom, observamos que o melhor modelo é o que utiliza BN, em que alcança as melhores métricas de desempenho. Destaca-se também o impacto positivo do DA, que melhora significativamente a performance, talvez devido à correção da diversidade das imagens. Além disso, os tempos de execução na GPU são consideravelmente menores que na CPU, reforçando a eficiência do uso de aceleração por _hardware_. Analisaremos o modelo final (5) com BN em mais detalhes:

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_custom_models/final_model-evolution.png", width: 95%),
    image("images_custom_models/final_model-matrix.png", width: 95%),
  ),
  caption: [
    Resultados (Evolução do Treino e Matriz de confusão no teste) para o modelo com _Batch Normalization_ (5)
  ],
)<Curvas_Loss_Acuracy_Matriz>

Na @Curvas_Loss_Acuracy_Matriz verificamos picos nas curvas de validação para ambas as métricas, ao contrário da curva de treino que decresce ou cresce de forma constante, respectivamente. Este comportamento é justificado pelo pequeno número de observações dos dados de validação @formula1, isto é, as alterações feitas nos pesos do modelo são adequadas para os dados de treino e não para os de validação. No conjunto de teste, vemos que a matriz de confusão faz previsões sólidas, numa taxa de acertos de cerca de 72.8%, valor ligeiramente mais baixo que o mostrado no gráfico da evolução da _accuracy_ na melhor época (13-menor valor de _loss_). Esta diferença pode ser explicada pela disparidade das imagens no conjunto de teste, o que torna a generalização mais "desafiante" para o modelo.

#text(size: 9pt)[$ text("Conj")_"Treino" = (2000)/(3000) times 100 % approx 66.7% gt gt text("Conj")_"Validação" = (500)/(3000) times 100 % approx 16.7% $<formula1>]




#figure(
  grid(
    align: horizon + center,
    columns: 2,
    image("images_custom_models/final_model-badpred(cat).png", width: 95%),
    image("images_custom_models/final_model-badpred(dog).png", width: 95%),
  ),
  caption: [
    Mapa de saliência de classificações incorretas para o modelo final com _Batch Normalization_ (5)
  ],
) <saliency_badpred>

Ao analisar a @saliency_badpred, com classificações incorretas e os mapas de saliência, observamos que o modelo está focado em regiões exteriores ao gato na 1ª imagem (talvez pela luminosidade) e, por isso, classifica como cão com 0.738 de probabilidade. Já na 2ª imagem, apesar de encontrar com sucesso o animal, não consegue extrair adequadamente as características do mesmo para prever na classe correta.


= : Transferência de conhecimento <parte3>

Com o objetivo de melhorar a capacidade de classificação do nosso modelo, é neste segmento aplicada a transferência de conhecimento. Esta abordagem aproveita redes pré-treinadas, que possuem um nível superior de complexidade na extração de características. Desta forma, o treino será focado na otimização do classificador.

Estas transferências foram realizadas com as redes #link("https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Small")[MobileNetV3 Small], #link("https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV3Large")[MobileNetV3 Large], #link("https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16")[VGG16] e #link("https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50")[ResNet50] .


#figure(
  tablex(
    columns: 8,
    align: center + horizon,
    header-rows:1,
    auto-lines: false, rowspanx(2)[*Modelo* \ *Pré-treinado*], vlinex(stroke:0.2mm), hlinex(stroke:0.2mm),
    rowspanx(2)[#text(10pt)[*Tempo de Execução GPU*]#footnote[GPU utilizada para execução dos modelos pré treinados: #link("https://www.gigabyte.com/Graphics-Card/GV-N4060WF2OC-8GD#kf")[#text("GeForce RTX™ 4060 WINDFORCE OC 8G")]]], vlinex(stroke:0.2mm), rowspanx(2)[#text(10pt)[*Nº Epochs*]], vlinex(stroke:0.2mm),
    colspanx(3)[#text(10pt)[*Métricas de Desempenho*]], vlinex(stroke:0.2mm), colspanx(2)[#text(10pt)[*Nº Parâmetros*]],
    (), (), (), [#text(8pt)[*Precision*]], hlinex(stroke:0.2mm), [#text(8pt)[*Accuracy*]], hlinex(stroke:0.2mm), vlinex(stroke:0.2mm), [#text(8pt)[*Recall*]],
    vlinex(stroke:0.2mm), [#text(8pt)[*Treináveis*]], vlinex(stroke:0.2mm), [#text(8pt)[*Totais*]],
    [MobileNetV3 Small], [20.9s], [8], [#calc.round(0.983806, digits: numberDigits)], vlinex(stroke:0.2mm), [#calc.round(0.984, digits: numberDigits)], [#calc.round(0.983806, digits: numberDigits)], vlinex(stroke:0.2mm), [4 718 849], /* non treinable + optimizer */ [15 095 669],
    hlinex(stroke:0.2mm),

    [MobileNetV3 Large], [1m 29.9s], [13], [#calc.round(1, digits: numberDigits)], [#calc.round(0.996, digits: numberDigits)], [#calc.round(0.991968, digits: numberDigits)], vlinex(stroke:0.2mm), [7 864 577], /* non treinable + optimizer */ [26 590 085],
    hlinex(stroke:0.2mm),

    [VGG16], [1m 13.8s], [10], [#calc.round(0.983806, digits: numberDigits)], [#calc.round(0.968, digits: numberDigits)], [#calc.round(0.952941, digits: numberDigits)], vlinex(stroke:0.2mm), [3 211 521], [24 349 253],
    hlinex(stroke:0.2mm),

    [ResNet50], [1m 19.9s], [13], [#calc.round(0.979757, digits: numberDigits)], [#calc.round(0.982, digits: numberDigits)], [#calc.round(0.98374, digits: numberDigits)], vlinex(stroke:0.2mm), [12 845 313], [62 123 653]
  ),
  caption: [Resultados dos modelos pré-treinados com o treino da rede convolucional],
  kind:table
)<tabela_pretrained>

Com base na @tabela_pretrained conseguimos perceber que o MobileNetV3 Large destaca-se com 99.6% de Accuracy e 100% de Precision, seguido pelo ResNet50 e VGG16, que também apresentam métricas acima de 96%. Ainda, é possível comparar a versão do modelo MobileNetV3 Large com o Small, onde verificamos que o Large destaca-se por ter um desempenho ligeiramente superior, mas com um custo computacional mais elevado ($approx$4.5x maior), enquanto o Small pode ser mais útil para cenários com restrições de recursos sem comprometer significativamente as métricas. Em geral, os modelos revelam todos muito boas previsões e erram muito pouco (de 2-16 observações $arrow$ @Anexobadpred[]).

É de se notar também que houve uma melhoria bastante significativa deste modelos face aos resultados da rede convolucional "custom" desenvolvida na @parte2[], especialmente nas métricas de desempenho. Talvez esta discrepância de valores deva-se ao facto de que estes modelos pré-treinados já foram treinados em grandes bases de dados, o que permite uma extração mais aprofundada de características e generalização. Além disso, os tempos de execução são bastante satisfatórios, tornando estes modelos em opções viáveis para aplicações práticas como esta.


#align(horizon + center)[
  #figure(
  grid(
    columns: (260pt, 300pt), 
    gutter: -10pt,
    rows: 1,
    rect(
      
      [A imagem mostra a saída de um modelo pré-treinado MobileNetV3Large, que classificou corretamente a imagem como um gato. A saliência indica que o modelo focou principalmente na região do olho e um pouco no formato da orelha do gato, o que já foi suficiente para garantir uma classificação precisa, com 100% de probabilidade. Estas características fisiológicas são essenciais para diferenciar um gato de um cão, o que reforça a eficiência deste modelo em classificação de imagens.

], width: 100%, radius: (
    left: 5pt,
    right: 5pt,
  )),
    image("images_pretrained_models/MobileNetLarge/1Xpred_MbNLarge.png", width: 90%)

  ),
  caption: [
    Exemplo de imagem que o MobileNetV3 Large prevê corretamente com 100% de probabilidade 
  ],
)<goodpred>
]


/*#figure(
image("images_pretrained_models/MobileNetLarge/1Xpred_MbNLarge.png", height: 15%),
  caption: [Exemplo de imagem que o modelo tem certeza na classe real]
)*/

#pagebreak()

= : Resultados num conjunto out-of-sample

Nesta fase, serão utilizadas imagens de cães e gatos capturadas pelo grupo para realizar previsões e analisar o desempenho dos modelos numa amostra totalmente diferente. Para isso, será dado um maior foco nos dois modelos mais robustos de cada abordagem: o modelo "custom" `Final Model`, desenvolvido na @parte2[] (com métricas apresentadas em @table_results_custom), e o modelo de "transfer learning" `MobileNetV3 Large`, criado na @parte3[] (com métricas disponíveis em @tabela_pretrained).  As imagens utilizadas foram previamente cortadas (_cropped_) para evitar o achatamento na leitura e _rescale_ das imagens e foram no total *16 imagens*, 7 imagens de cães e 9 de gatos. /*Foram garantidos todos os procedimentos necessários para uma previsão correta, obtendo-se os seguintes resultados:*/
#align(center)[*"Custom" Model - `Final Model` (5) :*]
Errou 2 gatos (acertando 7) e 3 cães (acertando 4). Ou seja, esta previsão registou 0.778 de _precision_, 0.625 de _accuracy_ e 0.636 de _recall_. Na @FinalModel_bad_prediction observamos dois exemplos de imagens más classificadas que captam características diferentes. A da Luna (imagem da esquerda) aparenta focar-se em demasia na "portinhola" da gaiola e não no animal. Já a imagem à direita foca-se no animal, contudo, fatores como a luminosidade ou a pequena dimensão do mesmo, podem resultar na previsão errada.


/*A @Tabelaprev1 demonstra as métricas que este modelo apresentou:


#figure(
  tablex(
    columns: (auto, auto, 50pt),
    align: center + horizon,
    header-rows:1,
    auto-lines: true, 
    colspanx(3)[*Final Model*], (), (),
    [*Precision*], [*Accuracy*], [*Recall*],
    [$0.778$], [$0.625$], [$0.636$]
  ),
  caption: [Métricas de previsão do `Final Model` com as fotos do grupo],
  kind:table
)<Tabelaprev1>
*/

#figure(
  grid(
    align: horizon + center,
    columns: 2,
    image("images_predict/final_model-bad_prediction_cat.png", width: 90%),
    image("images_predict/final_model-bad_prediction_dog.png", width: 90%),
  ),
  caption: [
    Fotografias previstas de forma incorreta pelo modelo `Final model`
  ],
)<FinalModel_bad_prediction>


#align(center)[*Transfer Learning Model: `MobileNetV3 Large`*:]

Errou 4 gatos (acertando 5) e 2 cães (acertando 5). Ou seja, esta previsão registou 0.556 de _precision_, 0.438 de _accuracy_ e 0.5 de _recall_. Na @MobileNetLarge_bad_prediction, o modelo pré-treinado demonstra uma alta confiança na previsão, assim como na @goodpred, porém, desta vez, erra na classe original. Foi de facto surpreendente o modelo prever incorretamente, pois a saliência parece estar bastante evidente nos pontos característicos que diferem um gato e um cão. Acreditamos que seja pela forma distinta de como estas fotografias estejam representadas em relação às imagens do conjunto de dados. 

/*
#figure(
  tablex(
    columns: (auto, auto, 50pt),
    align: center + horizon,
    header-rows:1,
    auto-lines: true, 
    colspanx(3)[*MobileNetV3 Large*], (), (),
    [*Precision*], [*Accuracy*], [*Recall*],
    [$0.556$], [$0.438$], [$0.5$]
  ),
  caption: [Métricas de previsão do `MobileNetV3 Large` com as fotos do grupo],
  kind:table
) <QualquerCoisa> */

#figure(
  grid(
    align: horizon + center,
    columns: 2,
    image("images_predict/MobileNetLarge-bad_prediction_cat.png", width: 95%),
    image("images_predict/MobileNetLarge-bad_prediction_dog.png", width: 95%),
  ),
  caption: [
    Fotografias previstas de forma incorreta pelo modelo `MobileNetV3 Large`
  ],
)<MobileNetLarge_bad_prediction>

Embora sejam poucas imagens, temos um conjunto de fotografias de fácil e difícil previsão. De modo geral, o `Final Model` obteve um desempenho superior, possivelmente devido a uma estrutura mais ajustada ao problema. Já o `MobileNetV3 Large`, por ser um modelo mais pesado, pode ter sido impactado pelo excesso de parametrização e _layers_, bem como pelas condições diferentes na leitura das imagens, o que pode ter impactado negativamente os resultados.

O repositório do projeto pode ser encontrado no seguinte #link("https://github.com/GothamAnalytics-7/APVC/tree/main/APVC-Desafio2")[link].

#pagebreak()
#set heading(numbering: none)
#show heading.where(level:1): set heading(numbering: none)

= Anexos <Anexos> // IF NEEDED
#set heading(numbering: (level1, level2,..levels ) => {
  if (levels.pos().len() > 0) {
    return []
  }
  ("Anexo", str.from-unicode(level2 + 64)/*, "-"*/).join(" ")
}) // seria so usar counter(heading).display("I") se nao tivesse o resto
//show heading(level:3)

== - Evolução do treino e matriz de confusão nas várias fases do modelo "custom" 

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_custom_models/baseline_model-evolution.png", width: 100%),
    image("images_custom_models/baseline_model-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo _baseline_ (1)
  ],
)

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_custom_models/mais1conv_model-evolution.png", width: 100%),
    image("images_custom_models/mais1conv_model-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo com mais uma camada convolucional (2)
  ],
)

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_custom_models/dropout_model-evolution.png", width: 100%),
    image("images_custom_models/dropout_model-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo com uma _layer_ de _dropout_ (3)
  ],
)

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_custom_models/augmented_model-evolution.png", width: 100%),
    image("images_custom_models/augmented_model-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo com Data Augmentation (4)
  ],
)


#pagebreak()

== - Evolução do treino e matriz de confusão nos modelos pré-treinados 

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_pretrained_models/MobileNetSmall/MobileNetV3Small-evolution.png", width: 100%),
    image("images_pretrained_models/MobileNetSmall/MobileNetV3Small-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo MobileNetV3 Small
  ],
)

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_pretrained_models/MobileNetLarge/MobileNetV3Large-evolution.png", width: 100%),
    image("images_pretrained_models/MobileNetLarge/MobileNetV3Large-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo MobileNetV3 Large
  ],
)

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_pretrained_models/VGG16/VGG16-evolution.png", width: 100%),
    image("images_pretrained_models/VGG16/VGG16-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo VGG16
  ],
)

#figure(
  grid(
    align: horizon + center,
    columns: (7fr, 3.5fr),
    image("images_pretrained_models/ResNet50/ResNet50-evolution.png", width: 100%),
    image("images_pretrained_models/ResNet50/ResNet50-matrix.png", width: 100%),
  ),
  caption: [
    Resultados para o modelo ResNet50
  ],
)

#pagebreak()

== - Todas as classificações incorretas nos modelos pré-treinados <Anexobadpred>

#figure(
image("images_pretrained_models/MobileNetSmall/badpreds_MbNSmall.png", height: 92%),
  caption: [Classificações incorretas e mapa de saliência para o modelo *MobileNetV3 Small*]
)

#pagebreak()

#figure(
image("images_pretrained_models/MobileNetLarge/badpreds_MbNLarge.png", width: 90%),
  caption: [Classificações incorretas e mapa de saliência para o modelo *MobileNetV3 Large*]
)

#pagebreak()

#figure(
  grid(
    align: horizon + center,
    columns: 2,
    image("images_pretrained_models/VGG16/badpreds_VGG16-1.png", height: 95%),
    image("images_pretrained_models/VGG16/badpreds_VGG16-2.png", height: 95%),
  ),
  caption: [
    Classificações incorretas e mapa de saliência para o modelo *VGG16*
  ],
)


#pagebreak()

#figure(
image("images_pretrained_models/ResNet50/badpreds_ResNet50.png", height: 98%),
  caption: [Classificações incorretas e mapa de saliência para o modelo *ResNet50*]
)

#pagebreak()

== - Matrizes de confusão para as imagens out_of_sample

#figure(
image("images_predict/final_model-matrix_predict.png", width: 70%),
  caption: [Matriz de confusão para o `Final Model`]
)

#figure(
image("images_predict/MobileNetV3Large-matrix_predict.png", width: 70%),
  caption: [Matriz de confusão para o `MobileNetV3 Large`]
)

#pagebreak()

== - Tentativa de ajuste dos hiperparâmetros do modelo com Optuna

#figure(
image("images/optuna_try.jpg", width: 98%),
  caption: [Código de _tuning_ dos modelos desenvolvidos]
)