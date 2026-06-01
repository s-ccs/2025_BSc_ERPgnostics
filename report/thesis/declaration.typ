// Standalone declaration page rebuilt from the official PDF.
// Compile directly with: typst compile declaration.typ

#set page(
  paper: "a4",
  margin: 0pt,
  header: none,
  footer: none,
  numbering: none,
)

#set text(
  font: ("Calibri", "Liberation Sans", "Noto Sans"),
  size: 10.8pt,
  fill: rgb("#111111"),
  hyphenate: false,
)
#set par(justify: true, linebreaks: "optimized", leading: 0.58em)

#let content-width = 154mm
#let left-offset = 28mm

#let section-title(body) = text(size: 13pt, weight: "bold", body)
#let signature-line(body) = text(size: 10.8pt, body)
#let declaration-block(body) = block(width: content-width, body)

#place(top + left, dx: left-offset, dy: 30mm)[
  #declaration-block[
    #section-title[Erklärung]

    #v(4.5mm)

    Hiermit erkläre ich, dass ich die vorliegende Arbeit selbstständig verfasst und dabei keine anderen als die angegebenen Quellen und Hilfsmittel benutzt habe. Sämtliche wörtlichen oder sinngemäßen Übernahmen und Zitate aus anderen Werken sind kenntlich gemacht und nachgewiesen. Ich versichere, dass ich keine Hilfsmittel verwendet habe, deren Nutzung die Prüferin oder der Prüfer explizit ausgeschlossen hat.

    Im Anhang „Nutzung von KI-Tools“ habe ich dokumentiert, inwiefern KI-Tools zur Erstellung wesentlicher Teile dieser Arbeit verwendet wurden. Mit Abgabe der Arbeit übernehme ich die volle Verantwortung für alle Inhalte. Die #emph[Grundsätze zur KI-Nutzung für Studierende an der Universität Stuttgart] habe ich zur Kenntnis genommen und befolgt.

    Weder diese Arbeit noch wesentliche Teile daraus waren bisher Gegenstand eines anderen Prüfungsverfahrens. Ich habe diese Arbeit bisher weder teilweise noch vollständig veröffentlicht. Das elektronische Exemplar stimmt mit allen eingereichten Exemplaren überein.

    #v(4mm)
    #signature-line[Datum, Unterschrift der Studentin / des Studenten]
  ]
]

#place(top + left, dx: left-offset, dy: 151mm)[
  #line(length: content-width, stroke: 0.5pt + rgb("#777777"))
]

#place(top + left, dx: left-offset, dy: 184mm)[
  #declaration-block[
    #section-title[Declaration]

    #v(4.5mm)

    I hereby declare that the work presented in this thesis is entirely my own. I did not use any other sources and aids than the listed ones. I have marked all direct or indirect transfers from other sources contained therein as referenced quotations. I assure that I have not used any aids that were explicitly ruled out by the examiner.

    In Appendix "Use of AI Tools" I have documented to what extent AI tools were used for generating any significant parts of this work. By submitting this work, I assume full responsibility for all content. I have taken notice of and I comply with the #emph[Principles for the Use of Artificial Intelligence by Students at the University of Stuttgart.]

    Neither this work nor significant parts of it were part of another examination procedure. I have not published this work in whole or in part before. The electronic copy is consistent with all submitted hard copies.

    #v(4mm)
    #signature-line[Date and signature of the student]
  ]
]
