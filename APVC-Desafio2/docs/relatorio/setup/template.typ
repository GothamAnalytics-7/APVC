// #import "@preview/algo:0.3.3": algo, i, d, comment, code
#set text(lang: "pt", region: "pt")
#import "@preview/tablex:0.0.5": vlinex, hlinex, tablex, gridx
#import "@preview/algo:0.3.3": algo, i, d, comment, code
#let to = box(width: 8pt, baseline: 10%)[..]

#let media(it) = {
  $#h(-1pt)<#h(0pt)#text(baseline: 0.5pt, it)#h(0pt)>#h(1pt)$
}

#let grau = {
  $#h(-1pt)<#h(0pt)#text(baseline: 0.5pt)[k]#h(0pt)>#h(1pt)$
}
#let grau2 = {
  $#h(-1pt)<#h(0pt)#text(baseline: 0.5pt)[k]^2#h(0pt)>#h(1pt)$
}
#let Algoritmo = "Algoritmo"
#let algoKeywords = ("repetir", "até", "if", "then", "else", "vezes")
#let project(fontsize:11pt, doc) ={

  // set figure(placement: auto) TODO considerar

  // set algo(keywords: _algo-default-keywords) // doesnt work
  
  let calculated_leading = 10.95pt
  set heading(
    bookmarked: true
  )
  set raw(tab-size: 2)

  
  
  show raw: set text(
    font: "Fira Code"
  )

  show raw.where(block:true): it => [
    #block(fill: luma(230), inset: 6pt, radius: 1pt, it)
  ]

  
  
  // set text(
  //   font: "Libertinus Serif",
  //   fallback: false,
  //   size: fontsize,
  //   hyphenate: true,
  //   lang: "pt",
  //   region: "pt"
  // )
  set page(
    numbering: "1 / 5", //"1 / 24"
    margin: (left: 12mm, right: 12mm, top: 13mm, bottom: 13mm)
  )
  set par(
    justify: true,
    leading: calculated_leading, // calculo manual
    first-line-indent: 1cm,
  )
  show par: set block(spacing: calculated_leading)
  set math.equation(
    numbering: "(1)"
  )
  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(
        el.numbering,
        ..counter(eq).at(el.location())
      )
    } else {
      it
    }
  }
  show figure.where(
    kind: table
  ): set figure.caption(position: top)

  show figure.where(
    kind: algo
  ): set figure(supplement: "Algoritmo", placement: none)
  show figure.where(
    kind: algo
  ): set block(breakable: true)

  show figure.where(
    kind: Algoritmo
  ): set figure(supplement: "Algoritmo") // n funciona?

  show figure.where(
    kind: Algoritmo
  ): it => [
    #set align(center)
    #set enum(numbering: it => [#text(fill:gray)[$it$]])
    #set list(marker: [--], indent: 0pt)
    #set par(leading: 0.5em)
    #rect(stroke: 0.5pt,align(left, it.body))
    #v(-7pt)
    #it.caption
  ]
  show figure.caption: set text(size: fontsize - 2pt)
  show figure.where(
    kind: grid
  ): set figure(kind: image) // n funciona n sei pq

  set list(indent: 0pt)
  // show figure.where(
  //   kind: tablex
  // ): set figure(kind: table) // n funciona n sei pq
  doc
}

#let vline(start:none, end:none) = {
  vlinex(start:start, end:end, stroke:0.5pt)
}

#let hline(start:none, end:none) = {
  hlinex(start:start, end:end, stroke:0.5pt)
}

#let hline_header(start:none, end:none) = {
  hlinex(start:start, end:end, stroke:0.5pt, expand:-3pt)
}

#let hline_before(start:none, end:none) = {
  // hlinex(start:start, end:end, stroke:0.5pt, expand:-3pt)
}
