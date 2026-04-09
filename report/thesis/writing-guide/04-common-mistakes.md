# Common Mistakes

## Capitalisation

- Do NOT randomly capitalise words.
- Capitals are used for: sentence starts, proper nouns, acronyms, and certain
  words in top-level headings.
- Definitions do NOT turn words into proper nouns — don't capitalise them.
- **Important exception**: numbered section/figure references ARE capitalised
  in scientific writing:
  - "in the next section we introduce..." (generic, lowercase)
  - "in Section 3.1 we demonstrate..." (numbered, capitalised)
  - Sub-sections are still called "Section": "Section 3.1.2", not
    "Subsection 3.1.2"

## Headings

- Only top-level (and maybe second-level) headings should be title-capitalised.
- Lower-level headings: capitalise first word only (plus proper nouns).
- When capitalising headings: only nouns, adjectives, pronouns, verbs, adverbs.

## Acronyms

- Define ALL acronyms on first use (except universally known ones like CPU).
- Standard form: full term first, then abbreviation in parentheses:
  "translation look-aside buffer (TLB)"
- Do NOT introduce acronyms in headings. Define on next paragraph appearance.
- Do NOT introduce in abstract (or if you do, re-introduce in the body).
- If an acronym hasn't been used for many pages, gently re-introduce it.
- Plurals: "CPUs" (NOT "CPU's"). "OSes" (NOT "OS's").
- Acronyms are normally ALL CAPS or mixed case (QoS). Almost never all
  lowercase.

## Definitions / New Terms

- Use *italics* to introduce new terms.
- Do NOT use quotation marks for definitions.
- Do NOT capitalise the defined term.

## Footnotes

- Use sparingly (about one every few pages at most).
- Footnotes must be complete, standalone sentences.
  - BAD: "5 KiB" as a footnote
  - GOOD: "The buffer size is defined to be 5 KiB."
- Place footnote markers AFTER punctuation, not before.

## Spelling

- There is NO excuse for not running a spell checker.
- Be consistent: either British/Australian OR American spelling, not mixed.
- If using American spelling, use ALL American rules consistently.

## Spaces

- No space before colons.
- No space after opening or before closing parentheses.
- Yes space before opening and after closing parentheses.
- Half-space between number and unit: "100 Hz" (use `100\,Hz` in LaTeX).

## Pseudo-Accuracy (Excess Digits)

- Do NOT report "improvements of 27.81%" when accuracy is only a few percent.
- Round to meaningful precision. What matters is the order of magnitude.
- Standard deviations are second-order — report to ONE significant digit only.
- In tables, show digits consistent with actual accuracy.
- Do not suppress trailing zeros if they carry information.

## Percentage vs Percentage Points

- Going from 20% to 30% is a **50% increase** (or **10 percentage points**).
- It is absolutely NOT a "10% increase".
- Use terminology correctly.

## Equations

- Equations are part of the prose, not floats. Make them part of the sentence:
  "The dynamic power is given as P = c f V^2, where f is the frequency..."
- Equation numbers are only needed if you cross-reference them.
