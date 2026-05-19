#import "@preview/glossarium:0.5.6": print-glossary

// Only print short and long, disregard rest
#let custom-print-title(entry) = {
  let short = entry.at("short")
  let long = entry.at("long", default: "")
  [#strong(short) #h(0.5em) #long]
}

#let custom-print-reference(
  entry,
  show-all: false,
  disable-back-references: false,
  minimum-refs: 1,
  description-separator: ": ",
  user-print-gloss: none,
  user-print-title: none,
  user-print-description: none,
  user-print-back-references: none,
) = {
  if show-all == true {
    block(
      above: 0pt,
      below: .4em,
    )[
      #user-print-gloss(
        entry,
        show-all: show-all,
        disable-back-references: disable-back-references,
        minimum-refs: minimum-refs,
        description-separator: description-separator,
        user-print-title: user-print-title,
        user-print-description: user-print-description,
        user-print-back-references: user-print-back-references,
      )
    ]
  }
}

#let abbreviations-page(abbreviations) = {
  // --- List of Abbreviations ---
  align(left)[
    = List of Abbreviations
    #v(1em)
    #print-glossary(
      abbreviations,
      user-print-title: custom-print-title,
      user-print-reference: custom-print-reference,
      show-all: true,
      disable-back-references: true,
    )
  ]
}
