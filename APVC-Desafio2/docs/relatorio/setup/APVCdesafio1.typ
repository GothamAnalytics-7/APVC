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


= : Classificação Multi-classe <1.Multiclass>

Pretende-se fazer a classificação de imagens do _dataset_ FASHION_MNIST#footnote[https://en.wikipedia.org/wiki/Fashion_MNIST], que é composto por imagens, a preto e branco, de 10 tipos de peças de roupa (60000 imagens no conjunto de treino e 10000 imagens no conjunto de teste). Contudo, para validar e testar o modelo foi utilizado ainda um conjunto de validação com 10000 observações retiradas dos dados de treino.

Para construir a rede foi utilizada uma camada de input de 28 por 28 _features_/píxeis da imagem e depois transformada num vetor (1D), ou seja, com $28 times 28 = #(28*28)$ _features_; uma camada escondida com 128 neurónios e função de ativação ReLu e uma camada de _output_ com 10 neurónios, correspondente ao número de classes. Para além destas camadas foi utilizado um Dropout com um fator de 20% para evitar _overfitting_ e aumentar a robustez do modelo. A rede foi ainda compilada com o otimizador Adam#footnote[Valor padrão da biblioteca: https://keras.io/api/optimizers/adam/] (valor de `learning_rate` de 0.001 - ver outros valores testados no @tentativas_lrate[]); com a função de perda `categorical_crossentropy`, ideal para problemas de classificação multi-classe; e por métricas, entre elas a _accuracy_, _precision_ e _recall_.

Construída a rede, o modelo foi posteriormente treinado com _batch size_ de 64 e com "50 _epochs_", mas com duas _callbacks_: BEST_MODEL_CHECKPOINT (guarda os pesos do modelo com menor valor da função de perda) e EARLY_STOPPING (quando não há melhorias na função de perda).

Com isto, o nosso modelo terá $(784 times 128) + 128 = #(784*128+128)$ neurónios na camada escondida e $(128 times 10) + 10 = #(128*10+10)$ neurónios na camada de _output_. No total são #((128*10+10)+(784*128+128)) parâmetros treináveis e 203542 não treináveis.

Os resultados obtidos foram os seguintes:

#figure(
image("images/multiclasse_loss.png", width: 85%),
  caption: [Evolução da Perda e da Acurácia ao longo das Épocas: Classificação multi-classe]
)<multi_loss>



Na @multi_loss podemos observar a evolução da função de perda e da acurácia ao longo das épocas do modelo. No gráfico da perda vemos que, a partir de uma determinada época, a perda começa a estagnar para o conjunto de validação e não acompanha o conjunto de treino. Após um certo número de épocas, o modelo acaba por parar pela _callback_ de `early_stopping`, pois o valor da perda para a validação começa a decrescer pouco/a aumentar significativamente e a distanciar-se bastante da perda no conjunto de treino. Já no gráfico com a evolução da _accuracy_, observamos a mesma tendência, onde o modelo acaba por parar com  de diferença entre o conjunto de treino e validação. 

Após treinado o modelo e este ter sido validado fazemos a previsão para o conjunto de teste. Para avaliar a performance utilizamos a matriz de confusão da @multi_CM, juntamente com as métricas tradicionais. Entre estas destacamos a _accuracy_, as métricas individuais de cada classe e as métricas macro, que correspondem à média das métricas individuais de todas as classes, de forma a garantir que cada classe tenha o mesmo peso na avaliação.

#set par(justify: false)
#set text(hyphenate: false)
#set text(8pt)

#align(center + horizon)[
#oasis-align(int-dir:-1, swap:true, max-iterations: 1, int-frac: 0.7,
  
  figure(
image("images/multiclasse_CM.png", width: 85%),
  caption: [Matriz de confusão: Classificação multi-classe]
),
  [
#figure(
  tablex(
    columns: 4,
    align: center + horizon,
    header-rows: 1,
    header-columns: 1,
    auto-lines: true,
    stroke: 0.1mm,
    repeat-header: true,

    map-rows: (row, cells) => cells.map(c =>
    if c == none {
      c
    } else {
      (..c, fill: if row == 0 { rgb("#BAD6EB") } else { none })
    },
  ),

    map-column: (column, cells) => cells.map(c =>
    if c == none {
      c
    } else {
      (..c, fill: if column == 1 { rgb("#BAD6EB") } else { none })
    },
  ),
    
    // Cabeçalhos
    [*Classe*], [*Precision*], [*Recall*], [*F1-Score*],
    hlinex(stroke: 0.5mm),
    
    // Linhas com os valores
    [#text(6.5pt)[T-Shirt/Top]], [0.86], [0.79], [0.82],
    [Trouser], [0.98], [0.97], [0.98],
    [Pullover], [0.81], [0.76], [0.78],
    [Dress], [0.86], [0.90], [0.88],
    [Coat], [0.78], [0.80], [0.79],
    [Sandal], [0.97], [0.96], [0.97],
    [Shirt], [0.68], [0.71], [0.69],
    [Sneaker], [0.94], [0.95], [0.94],
    [Bag], [0.95], [0.98], [0.96],
    [Boot], [0.96], [0.96], [0.96],
    hlinex(stroke: 0.5mm),
    [#text(7.5pt)[*Macr. avg*]], [0.88], [0.88], [0.88],
    [*Accuracy*], colspanx(3)[0.88],
  ), caption: [Classificação multi-classe], kind: table
)<multi_CM>
 ]
)]

#set par(justify: true)
#set text(hyphenate: true)
#set text(11pt)

Analisando os resultados do modelo de forma mais detalhada, é evidente que este apresenta um desempenho global bastante robusto. O Accuracy atinge um valor de $0.88$, um resultado bastante positivo na identificação de peças de vestuário e acessórios. No entanto, ao observar a @multi_CM, verificamos que o modelo demonstra dificuldades na previsão de algumas categorias específicas. Um exemplo claro é a classe Shirt, onde a taxa de acerto é inferior à de outras peças. Em contrapartida, no que diz respeito a acessórios como Bag e Boot, o modelo apresenta um desempenho consistente, sem dificuldades notáveis. Na @3.CompResults[], este modelo será binarizado de forma a responder à questão central deste trabalho.


// #figure(
// image("images/multiclasse_CM.png", width: 60%),
//   caption: [Matriz de confusão: Classificação multi-classe]
// )<multi_CM123>

= : Classificação Binária <2.Binary>

Para esta parte foram divididas as 10 classes originais do _dataset_ para 2 grupos distintos: um de peças de Vestuário e outro de Calçado/Malas. As classes  "T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat" e "Shirt" foram consideradas Vestuário, enquanto que as classes "Sandal", "Sneaker", "Bag" e "Boot" foram incluídas em Calçado/Malas. Estas classes foram utilizadas num modelo com os mesmos argumentos e _callbacks_ da @1.Multiclass[], com as seguintes diferenças: a camada de _output_ passou a ter um único neurónio com função de ativação _sigmoid_, enquanto que a função de perda foi alterada para `binary_crossentropy`. Ambas as alterações foram com o intuito de servir melhor um problema de classificação binária. Com a alteração na camada de _output_, o número de parâmetros treináveis nesta camada passará a ser $(128 times 1) + 1 = #(128*1+1)$ e o número de parâmetros na camada escondida permanecerá igual (#(784*128+128)).

#figure(
image("images/binary_loss.png", width: 85%),
  caption: [Evolução da Perda e da Acurácia ao longo das Épocas: Classificação binária]
)<bin_loss>

Pela @bin_loss podemos perceber que o modelo em poucas épocas de treino chega a uma acurácia no conjunto de validação de cerca de 0.992, acabando de treinar na época 13 devido a dar _overfit_ no conjunto de treino em compromisso do conjunto de validação. Foram utilizados os pesos obtidos na época 8 como os finais. 

#figure(
  grid(
    columns: (250pt, 270pt), gutter: -1pt,
    rows: 1,
    align: (horizon + center),
    figure(
  tablex(
    columns: 4,
    align: center + horizon,
    header-rows: 1,
    header-columns: 1,
    auto-lines: true,
    stroke: 0.1mm,
    repeat-header: true,

    map-rows: (row, cells) => cells.map(c =>
    if c == none {
      c
    } else {
      (..c, fill: if row == 0 { rgb("#BAD6EB") } else { none })
    },
  ),

    map-column: (column, cells) => cells.map(c =>
    if c == none {
      c
    } else {
      (..c, fill: if column == 1 { rgb("#BAD6EB") } else { none })
    },
  ),
    
    // Cabeçalhos
    [*Classe*], [*Precision*], [*Recall*], [*F1-Score*],
    hlinex(stroke: 0.5mm),
    
    // Linhas com os valores
    [Calçado/Malas], [0.99], [1.00], [0.99],
    [Vestuário], [1.00], [0.99], [1.00],
    hlinex(stroke: 0.5mm),
    [*Macro avg*], [0.99], [0.99], [0.99],
    [*Accuracy*], colspanx(3)[0.99],
  ), caption: [Relatório de Classificação Binary], kind: table
),image("images/binary_CM_but_better.png", width: 95%)
  ),
  caption: [
    Classificação binária
  ],
  kind:image
)<bin_CM>

/* 
Relatório de Classificação:
               precision    recall  f1-score   support

Calçado/Malas       0.99      1.00      0.99      4000
    Vestuário       1.00      0.99      1.00      6000

     accuracy                           0.99     10000
    macro avg       0.99      0.99      0.99     10000
 weighted avg       0.99      0.99      0.99     10000 
*/
// PRECISION: no vestuário, todos os que o modelo diz que são vestuário são de facto vestuário;
// RECALL: no vestuário, alguns dos vestuários do conjunto são identificados como calçado/malas.

Neste paradigma, o modelo pouco confunde as duas classes do problema: 35 imagens de Vestuário do conjunto de teste são classificadas como Calçado/Malas, enquanto que 19 imagens de Calçado/Malas do conjunto de teste são classificadas como Vestuário. Desta forma, o modelo apresenta uma acurácia de 0.99 na tarefa de classificação.

#figure(
image("images/vestuario_as_calcado.png", width: 80%),
  caption: [Exemplo de imagem de Vestuário classificada como Calçado/Malas. À direita, vemos um mapa de saliência do modelo para o exemplo. Os píxeis mais vermelhos são aqueles cujos valores o modelo mais importância deu para o resultado que obteve.]
)<vestuario_as_calcado>







= : Comparação de resultados <3.CompResults>

// TODO: Binarizar as classes da Parte 1



Após terem sido realizadas as classificações multi-classe e binária nas partes 1 e 2, respetivamente, prossegue-se à comparação de resultados e a resposta à seguinte questão:


#quote[*Para  a  classificação  binária  Vestuário/Calçado  e  Malas  será  melhor  usar  uma  rede  para classificação multiclasse e depois binarizar as predições da rede neuronal, ou será melhor 
usar uma rede neuronal projetada para realizar diretamente a classificação binária?*]

Primeiramente, é necessário binarizar o modelo da Multi-classe (@1.Multiclass[]). Para isso, iremos utilizar como base a matriz de confusão do modelo multi-classe original, que pode ser visualizada na @multi_CM, e iremos demonstrar os cálculos que serão realizados para a binarizar. A @multi_CM_to_BIN demonstra os cálculos que vão ser realizados.



#figure(
image("images/multiclasse_CM_to_BIN.png", width: 75%),
  caption: [Matriz de confusão: Classificação multi-classe para Binarizar]
)<multi_CM_to_BIN>

Na matriz de confusão $M[i,j]$ da @multi_CM_to_BIN, podemos identificar quatro blocos de cores distintas, que são somados para a construção da matriz de confusão binarizada.
Para uma breve explicação da transformação, vamos utilizar os `True Negative`, que correspondem aos elementos $[i, j]$ com um fundo esverdeado na matriz. Esta /*TODO BOTAS*/ vai corresponder aos casos em que o modelo previu classes do novo grupo Calçado/Malas (classe 0) corretamente, ou seja, $sum_(i in"Calçado/Malas") sum_ (j in"Calçado/Malas") M[i, j] = #(963+946+981+957+22+1+13+29+4+35+25+1+6)$. Este valor corresponderá aos `True Negatives` da nova matriz de confusão binarizada, na @ConfMatrix_Compare. Para os cálculos completos pode-se consultar o @calcs_binar[].



//#align(center)[
//  #figure(
//image("images/cm_parte3.png", width: 75%),
//  caption: []
//)
//]

#figure(
  grid(
    columns: (180pt, 380pt), gutter: -1pt,
    rows: 1,
    align: (horizon + center),
    rect(stroke: 0.5pt,[
      
      A matriz de confusão dos modelos mostra resultados semelhantes. A @Tabela_MC_to_Bin apresenta a matriz dos modelos multi-classe após binarização, e a @bin_CM apresenta valores próximos, com uma ligeira melhoria no F1-score. Os resultados indicam que o desempenho dos dois modelos é quase idêntico.
      
], width: 95%, radius: (
    left: 5pt,
    right: 5pt,
  )),
    image("images/cm_parte3.png", width: 90%)
  ),
  caption: [
    Comparação dos modelos: Matriz de Confusão
  ],
) <ConfMatrix_Compare>


#figure(
  tablex(
    columns: 4,
    align: center + horizon,
    header-rows: 1,
    header-columns: 1,
    auto-lines: true,
    stroke: 0.1mm,
    repeat-header: true,

    map-rows: (row, cells) => cells.map(c =>
    if c == none {
      c
    } else {
      (..c, fill: if row == 0 { rgb("#BAD6EB") } else { none })
    },
  ),

    map-column: (column, cells) => cells.map(c =>
    if c == none {
      c
    } else {
      (..c, fill: if column == 1 { rgb("#BAD6EB") } else { none })
    },
  ),
    
    // Cabeçalhos
    [*Classe*], [*Precision*], [*Recall*], [*F1-Score*],
    hlinex(stroke: 0.5mm),
    
    // Linhas com os valores
    [Calçado/Malas], [0.99], [1.00], [0.99],
    [Vestuário], [1.00], [0.99], [0.99],
    hlinex(stroke: 0.5mm),
    [*Macro avg*], [0.99], [0.99], [0.99],
    [*Accuracy*], colspanx(3)[0.99],
  ), caption: [Relatório de Classificação Binary], kind: table
)<Tabela_MC_to_Bin>

/*#align(center)[
  #figure(
image("images/", width: 90%),
  caption: []
)
] */

Para além dos valores da tabela, ambos os métodos possuem ROC-AUC de 0.99 (ver @roc-auc[]).

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

== - Tentativas com outros valores de `learning_rate` no Otimizador Adam do modelo de classificação multi-classe<tentativas_lrate>


#figure(
image("images/lrate_01.png", width: 85%),
  caption: [Modelo de classificação multi-classe com valor de *0.01* para o `learning_rate`]
)

#figure(
image("images/lrate_0001.png", width: 85%),
  caption: [Modelo de classificação multi-classe com valor de *0.0001* para o `learning_rate`]
)

== - Cálculos para binarização do modelo de classificação multi-classe<calcs_binar>


$ #text(rgb("#FF0000"))[*FN*] = sum_(i in"Vestuário") sum_ (j in"Calçado/Malas") M[i, j] = \ = 19 + 5 + 6 + 10 + 3 + 12 = #(19 + 5 + 6 + 10 + 3 + 12) $
$ #text(rgb("#FFA500"))[*FP*] = sum_(i in"Calçado/Malas") sum_ (j in"Vestuário") M[i, j] = \ = 1 + 1 + 1 + 2 + 5 + 3 + 2 + 1 + 1 = #(1 + 1 + 1 + 2 + 5 + 3 + 2 + 1 + 1) $
$ #text(rgb("#008000"))[*TN*] = sum_(i in"Calçado/Malas") sum_ (j in"Calçado/Malas") M[i, j] = \ = 963+946+981+957+22+1+13+29+4+35+25+1+6 = #(963+946+981+957+22+1+13+29+4+35+25+1+6) $

$ #text(rgb("#800080"))[*TP*]= sum_(i in"Vestuário") sum_ (j in"Vestuário") M[i, j] = sum_(i=0)^9 sum_(j=0)^9 M[i, j] - ("FN" + "FP" + "TN") = \ = 10000 - (#(963+946+981+957+22+1+13+29+4+35+25+1+6) + #(1 + 1 + 1 + 2 + 5 + 3 + 2 + 1 + 1) + #(19 + 5 + 6 + 10 + 3 + 12)) = #(10000 - ((963+946+981+957+22+1+13+29+4+35+25+1+6) + (1 + 1 + 1 + 2 + 5 + 3 + 2 + 1 + 1) + (19 + 5 + 6 + 10 + 3 + 12))) $


== - ROC-AUC para os dois métodos realizados<roc-auc>

#figure(
image("images/roc-auc.png", width: 85%),
  caption: [Comparação dos modelos com curva ROC]
)
