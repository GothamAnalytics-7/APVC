#import "template.typ": *
#set text(lang: "pt", region: "pt")
#let titulo = "Desafio 2 - Redes neuronais convolucionais para distinguir cães de gatos"
#let subtitulo = "Trabalho de Grupo realizado no âmbito da Unidade Curricular de Aprendizagem Profunda para Visão por Computador do 1º ano do Mestrado em Ciência de Dados"
#let indice= false
// TODO change to grid
// let versao
// let data
// let autores para ter for loop
#set align(center)
#set page(
  margin: (x: 3cm, y: 2.5cm),
)
#set text(
  hyphenate: false
)
#set par(
  first-line-indent: 0pt,
  justify: false,
)
#par(leading: 0.15cm)[
  #show: smallcaps

  #text(13pt)[ISCTE-IUL] \
  #text(12.5pt)[Mestrado em Ciência de Dados]
]
#v(0.1cm)
#line(length: 100%, stroke: 0.5pt)

#v(0.4cm)
#par(leading: 0.22cm)[
  #text(24pt)[#titulo]
  #linebreak()
  #v(0.1cm)
  #text(16pt)[#subtitulo]
]
#v(0.4cm)
#line(length: 100%, stroke: 1pt)
#if indice {
  v(0.4cm)
} else {
  v(1.4cm)
}

#v(1cm)
#par(leading: 0.3cm)[
  #text(20pt)[Diogo Freitas, 104841, MCD-LCD-A1]\
  #link("mailto:Diogo_Alexandre_Freitas@iscte-iul.pt")
  #v(0.12cm)
  #text(20pt)[João Francisco Botas, 104782, MCD-LCD-A1]\
  #link("mailto:Joao_Botas@iscte-iul.pt")
  #v(0.12cm)
  #text(20pt)[Miguel Gonçalves, 105944, MCD-LCD-A1]\
  #link("mailto:Miguel_Goncalves_Pereira@iscte-iul.pt")
  #v(0.12cm)
  #text(20pt)[Ricardo Galvão, 105285, MCD-LCD-A1]\
  #link("mailto:Araujo_Galvao@iscte-iul.pt")
]
#if indice {
  v(1cm)
} else {
  v(1.4cm)
}
#par(leading: 0.2cm)[
  #text(16pt)[30 de março 2025]\
  #text(16pt)[Versão 1.0.0] 
  // 1st number: versão "final" para entregar
  // 2nd number: incremento de conteúdo e tópicos
  // 3rd number: correções de escrita e + info/comments
  // ao aumentar um número, da direita desse passam a 0
]
#v(0.1cm)
#set align(left)
 #outline(
   title: text(weight: 600)[Índice],
   fill: text(15pt, spacing: 220%)[#repeat(" . ")], depth: 1
 )

