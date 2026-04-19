# Model Writing Instructions for `thesis.typ`

These instructions are for future models editing this thesis. They summarise
the user's preferences, the local writing guide, and the lessons from Gernot
Heiser's technical writing advice. They apply when writing thesis prose in
`report/thesis/thesis.typ` and adding sources to `report/thesis/refs.bib`.

## Scope

- Edit thesis prose in `report/thesis/thesis.typ`.
- Add or fix bibliography entries in `report/thesis/refs.bib`.
- Do not generate the PDF unless the user explicitly asks for it.
- Do not touch unrelated files for pure writing tasks.
- Keep existing structure, terminology, and citation style unless there is a
  clear reason to change them.

## Language and Tone

- Write the thesis in British English.
- Prefer clear, human prose over exhaustive academic padding.
- Keep the text short enough that the reader does not lose the point.
- Avoid repetition. State a caveat once, then move on.
- Use precise technical terms, but do not make the prose sound like a glossary.
- Avoid hype words such as "novel", "innovative", "groundbreaking", and
  similar filler.

## Paragraphs and Flow

- Explain top-down: first say what the paragraph is about, then give the
  needed detail.
- Lead the reader from the general idea to the specific thesis topic. Assume a
  reader with good general knowledge, but no prior knowledge of EEG, ERPs, or
  ERP images.
- Introduce concepts in a slow, steady order. For example: measurement problem,
  EEG, ERP averaging, what averaging hides, ERP images, pattern recognition,
  simulation, and only then the specific experiment.
- Do not start a section with specialised labels, model names, or internal
  project details before the reader knows why they matter.
- Each paragraph should have one clear job.
- Prefer 2--4 sentences per paragraph.
- Use section openings to orient the reader, not to advertise the section.
- Do not assume the reader remembers every earlier definition. Add a short
  reminder when a term returns after a long gap.
- Do not over-explain. A thesis section should be complete, but not a lecture
  transcript.

## Style Rules

- Use active voice where practical.
- Use present tense for what the thesis does: "This section defines...",
  "The model receives...", "The experiment compares...".
- Use past tense only for historical facts or specific completed work.
- Use inclusive "we" only when it genuinely takes the reader along. Avoid royal
  "we" for work the author performed alone.
- Define acronyms before first use in the body text.
- Keep terminology consistent. If the thesis uses "ERP image", do not switch to
  "ERP heatmap" unless the difference matters.
- Avoid long chains of subordinate clauses. Split the sentence if the reader
  has to hold too much state.
- Use lists sparingly in prose sections. Prefer readable paragraphs unless a
  list genuinely improves scanning.

## Citations and `refs.bib`

- Cite concrete claims that come from the literature.
- Prefer primary papers, official documentation, or well-established reviews.
- Do not cite sources that have not been checked well enough to support the
  claim.
- Reuse existing BibTeX keys where possible.
- Add only bibliography entries that are actually cited in `thesis.typ`.
- Use Typst citations in the local style, for example:
  `... single-trial variability @Jung2001.`
- Keep BibTeX entries complete enough for a thesis: authors, title, venue,
  year, DOI or URL where available.
- Preserve acronym capitalisation in titles with braces, for example `{ERP}`,
  `{EEG}`, `{P3}`, or `{CNN}`.

## ERP Image Pattern Naming

Use readable names in prose:

- sigmoid
- diverging bar
- one-sided fan
- hourglass
- two-sided fan
- tilted bar

Use the code labels only when referring to labels, files, or model classes:

- `sigmoid`
- `diverging_bar`
- `one_sided_fan`
- `hourglass`
- `two_sided_fan`
- `tilted_bar`
- `no_class`

Do not treat these pattern names as names of separate neural generators. They
are operational labels for visible ERP image morphology. Explain likely origins
briefly, but keep the distinction clear: detecting a pattern is not the same as
explaining the underlying brain process.

## How to Write New Thesis Text

1. Read the surrounding section before editing.
2. Identify the one point the new text must make.
3. Decide what the reader already knows at that point in the thesis. If the
   reader needs a bridge from everyday intuition to the technical topic, write
   that bridge first.
4. Draft the shortest version that still defines the terms, gives the reason,
   and supports the claim with citations.
5. Remove repeated caveats and phrases.
6. Check naming consistency against nearby text.
7. Add or update `refs.bib` only for cited sources.
8. Verify that all citation keys used in `thesis.typ` exist in `refs.bib`.
9. Do not compile the PDF unless explicitly requested.

## Quick Quality Check

Before finishing a writing task, check:

- Does each paragraph have one clear purpose?
- Does the section guide a non-specialist reader from the basic idea to the
  specific technical topic?
- Can a reader understand the point without opening the source files?
- Is the text concise enough, or does it repeat the same warning?
- Are the claims supported by citations?
- Are all citation keys present in `refs.bib`?
- Is the text in British English?
- Did the edit stay limited to `thesis.typ` and `refs.bib`, unless the user
  asked for something else?
