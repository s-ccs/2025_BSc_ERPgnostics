// Standalone title page rebuilt from the official PDF form.
// Compile directly with: typst compile titlepage.typ

#set page(
  paper: "a4",
  margin: 0pt,
  header: none,
  footer: none,
  numbering: none,
)

#set text(
  font: ("Arial", "Liberation Sans", "Noto Sans"),
  size: 12pt,
  fill: rgb("#101010"),
)
#set par(justify: false)

#let logo = "assets/uni-stuttgart-logo.svg"

#let title-line(body) = text(size: 17pt, weight: "bold", body)

#place(top + center, dy: 16mm)[
  #image(logo, width: 72mm)
]

#place(top + center, dy: 35mm)[
  #align(center)[
    #text(size: 12.8pt)[Institut für Visualisierung und Interaktive Systeme (VIS)]
    #linebreak()
    #text(size: 12.8pt)[Computational Cognitive Science]
  ]
]

#place(top + center, dy: 64mm)[
  #align(center)[
    #text(size: 12.8pt)[Universitätsstraße 32]
    #linebreak()
    #text(size: 12.8pt)[70569 Stuttgart]
  ]
]

#place(top + center, dy: 116mm)[
  #align(center)[
    #stack(
      spacing: 7mm,
      text(size: 14pt, weight: "bold")[Bachelorarbeit],
      stack(
        spacing: 4mm,
        title-line[Automated Pattern Detection in],
        title-line[ERP Images Using Convolutional],
        title-line[Neural Networks (CNN)],
      ),
    )
  ]
]

#place(top + center, dy: 177mm)[
  #text(size: 14pt)[Benjamin Borchert]
]

#place(top + center, dy: 224mm)[
  #box(width: 112mm)[
    #set text(size: 12.5pt)
    #table(
      columns: (34mm, 71mm),
      stroke: none,
      inset: (x: 0pt, y: 2.1mm),
      align: left,
      [*Studiengang:*], [Data Science],
      [*Prüfer:*], [Jun.-Prof. Dr. Benedikt Ehinger],
      [*Betreuer:*], [Vladimir Mikheev, M.Sc.],
    )
    #v(5mm)
    #table(
      columns: (34mm, 71mm),
      stroke: none,
      inset: (x: 0pt, y: 2.1mm),
      align: left,
      [*begonnen am:*], [01.12.2025],
      [*beendet am:*], [01.06.2026],
    )
  ]
]
