// A central place where libraries are imported (or macros are defined)
// which are used within all the chapters:
#import "utils/global.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes


// Fill me with the Abstract
#let abstract = [#lorem(150)]

// Fill me with acknowledgments
#let acknowledgements = [#lorem(50)]

#let appendix-term-table(..args) = table(
  columns: (1.15fr, 4.15fr),
  inset: (x: 5pt, y: 4pt),
  stroke: .4pt + luma(180),
  fill: (_, y) => if y == 0 {
    rgb("#eaf1f8")
  } else if calc.odd(y) {
    rgb("#f8fafc")
  } else {
    white
  },
  align: (x, y) => if y == 0 { center } else { left + top },
  ..args,
)

#let appendix-result-table(..args) = table(
  columns: (1.2fr, 1.15fr, 3.55fr, 1.1fr),
  inset: (x: 5pt, y: 4pt),
  stroke: .4pt + luma(180),
  fill: (_, y) => if y == 0 {
    rgb("#eaf1f8")
  } else if calc.odd(y) {
    rgb("#f8fafc")
  } else {
    white
  },
  align: (x, y) => if y == 0 { center } else { left + top },
  ..args,
)

// if you have appendices, add them here
#let appendix = [
  = Appendices

  #set par(justify: false)

  This appendix collects short definitions for the main experimental terms and
  compact comparison tables for the model results discussed in the thesis.
  The goal is to document which model, which data split, which preprocessing
  pipeline, and which measured outcome correspond to each result claim.

  == Short Definitions of the Main Experimental Terms

  #set text(size: 9pt)
  #appendix-term-table(
    [*Term*], [*Meaning in this thesis*],

    [Supervised learning], [
      Train the classifier only on real labelled ERP images. No pseudo-labels
      are added.
    ],
    [Self-supervised learning (SSL)], [
      Pretrain the encoder without class labels. In this project that stage is
      SimCLR-style contrastive learning on ERP images.
    ],
    [Linear probe], [
      Freeze the SSL encoder and train only a new classifier head on labelled
      ERP images.
    ],
    [Fine-tuning], [
      Update the full pretrained model on labelled ERP images so the learned
      representation adapts to the task.
    ],
    [Semi-self-supervised learning], [
      Use SSL first, then extend later training with pseudo-labelled examples
      from an unlabeled pool.
    ],
    [Pseudo-labeling], [
      Predict labels for unlabeled ERP images and keep only the high-confidence
      predictions as temporary training labels.
    ],
    [Trainfold images], [
      Use only ERP images from the current training folds as SSL input. The
      validation fold stays unseen.
    ],
    [Same-dataset mod-4 pool], [
      A larger unlabeled ERP-image pool from the same fixation dataset,
      generated with the same mod-4 policy.
    ],
    [Mod-4 split], [
      Split the sorted trials into four modulo parts. In week 18, pattern
      contributes 4 parts and no pattern contributes 1 part.
    ],
    [Grouped 5-fold cross-validation], [
      Samples from the same original ERP source stay in the same fold to avoid
      train/validation leakage.
    ],
    [Balanced accuracy], [
      Mean recall across both classes. It is preferred here because the labels
      are imbalanced.
    ],
    [Macro-F1], [
      F1 averaged equally across both classes, so the majority class does not
      dominate the score.
    ],
  )
  #set text(size: 11pt)

  #pagebreak()

  == Real-Data Supervised Baseline Candidates

  The table below focuses on experiments that use only the real fixation ERP
  data for the actual classification objective. This is the most relevant
  reference set when later methods are compared against a real-data baseline.
  All rows use the project default: mod-4 split and grouped 5-fold
  cross-validation. The numbers come from the saved preprocessing notebook
  output and the exported augmentation summaries.

  #set text(size: 8.2pt)
  #appendix-result-table(
    [*Experiment*], [*Model*], [*Key setup*], [*Outcome*],

    [Preprocessing], [
      ResNet18, pretrained
    ], [
      Best overall supervised run. Sort, z-score, Gaussian, filter, resize.
      Dilation filter, default radius, Gaussian high 100, 8 epochs.
    ], [
      BAcc 0.907.\ Macro-F1 0.903.
    ],

    [Preprocessing], [
      ResNet18, pretrained
    ], [
      Best Gaussian-only reference. Sort, z-score, Gaussian, resize. No
      filter. Gaussian mid 50, 8 epochs.
    ], [
      BAcc 0.904.\ Macro-F1 0.898.
    ],

    [Preprocessing], [
      ResNet18, random init
    ], [
      Best random-init run. Sort, z-score, filter, resize. Opening filter,
      low setting, no Gaussian, 8 epochs.
    ], [
      BAcc 0.875.\ Macro-F1 0.881.
    ],

    [Sup. augmentation], [
      ResNet18, random init
    ], [
      Trial dropout with threshold tuning. Class-aware augmentation: 4 pattern
      views and 1 no-pattern view.
    ], [
      BAcc 0.902.\ Macro-F1 0.880.
    ],

    [Sup. augmentation], [
      ResNet18, random init
    ], [
      Standard cross-entropy baseline. No SSL, no pseudo-labels, no
      augmentation, tuned threshold.
    ], [
      BAcc 0.551.\ Macro-F1 0.505.
    ],
  )
  #set text(size: 11pt)

  The strongest supervised-only result in the current code base came from the
  preprocessing challenge with the pretrained ResNet18. The Gaussian-only
  reference was already very strong, and the best filter-augmented pipeline
  improved it only slightly. The best random-initialised baseline was weaker
  than the pretrained baseline but still clearly above the raw supervised
  baseline from the SSL notebook.

  #pagebreak()

  == Supervised Augmentation and Imbalance Handling

  This table isolates the supervised-only comparison from the dedicated
  augmentation notebook. All runs use only labelled fixation ERP images with
  the shared mod-4 split and grouped 5-fold CV.
  The numbers come from data_augmentation_tests_ranked_summary.csv.

  #set text(size: 8.2pt)
  #appendix-result-table(
    [*Experiment*], [*Model*], [*Key setup*], [*Outcome*],

    [Augmentation], [
      ResNet18, random init
    ], [
      Trial dropout, threshold-tuned, class-aware augmentation:
      4 pattern views and 1 no-pattern view.
    ], [
      BAcc 0.902.\ Macro-F1 0.880.
    ],

    [Augmentation], [
      ResNet18, random init
    ], [
      Pink-noise augmentation, threshold-tuned, class-aware augmentation:
      4 pattern views and 1 no-pattern view.
    ], [
      BAcc 0.886.\ Macro-F1 0.877.
    ],

    [Augmentation], [
      ResNet18, random init
    ], [
      Time jitter, threshold-tuned, class-aware augmentation:
      4 pattern views and 1 no-pattern view.
    ], [
      BAcc 0.886.\ Macro-F1 0.882.
    ],

    [Augmentation], [
      ResNet18, random init
    ], [
      Safe combination of ERP augmentations, threshold-tuned, class-aware:
      4 pattern views and 1 no-pattern view.
    ], [
      BAcc 0.883.\ Macro-F1 0.867.
    ],

    [Imbalance], [
      ResNet18, random init
    ], [
      Class-weighted cross-entropy, no augmentation, tuned threshold.
    ], [
      BAcc 0.516.\ Macro-F1 0.446.
    ],

    [Imbalance], [
      ResNet18, random init
    ], [
      Focal loss, no augmentation, tuned threshold.
    ], [
      BAcc 0.538.\ Macro-F1 0.427.
    ],

    [Imbalance], [
      ResNet18, random init
    ], [
      Balanced batches, no augmentation, tuned threshold.
    ], [
      BAcc 0.532.\ Macro-F1 0.475.
    ],
  )
  #set text(size: 11pt)

  These results should be read carefully. In the augmentation study the gain is
  not a pure augmentation effect, because the training set was also rebalanced
  by generating four augmented views for each labelled pattern example and one
  augmented view for each labelled no-pattern example.

  #pagebreak()

  == SSL and Semi-SSL Comparison

  The SSL notebook compares representation learning, transfer mode, and
  pseudo-labeling. Unlike the supervised baselines above, these rows use either
  an unlabeled pretraining pool, pseudo-labels, or both.
  All rows are still evaluated with the project default: mod-4 split and
  grouped 5-fold CV.
  The numbers come from semi_supervised_learing2_summary.csv.

  #set text(size: 8.2pt)
  #appendix-result-table(
    [*Experiment*], [*Model*], [*Key setup*], [*Outcome*],

    [Semi-SSL + pseudo-labels], [
      ResNet18, SSL fine-tune + student stage
    ], [
      SSL pool: same-dataset mod-4 pool. Same SSL stage as above, then
      confidence-based pseudo-labeling with
      threshold 0.9; about 823 pseudo-labels kept per fold.
    ], [
      BAcc 0.837.\ Macro-F1 0.849.
    ],

    [Semi-SSL], [
      ResNet18, SSL fine-tune
    ], [
      SSL pool: same-dataset mod-4 pool. SimCLR-style pretraining on the
      larger same-dataset mod-4 pool, then
      full fine-tuning on true labels.
    ], [
      BAcc 0.776.\ Macro-F1 0.788.
    ],

    [Self-supervised], [
      ResNet18, SSL fine-tune
    ], [
      SSL pool: trainfold images. SimCLR-style pretraining on the current
      fold's training images, then
      full fine-tuning on true labels.
    ], [
      BAcc 0.676.\ Macro-F1 0.697.
    ],

    [Linear probe], [
      ResNet18, SSL linear probe
    ], [
      SSL pool: same-dataset mod-4 pool. SSL pretraining on the mod-4 pool,
      frozen encoder, and a newly trained
      classifier head only.
    ], [
      BAcc 0.505.\ Macro-F1 0.428.
    ],

    [SSL baseline], [
      ResNet18, supervised
    ], [
      No SSL; no pseudo-labels; direct supervised training only.
    ], [
      BAcc 0.500.\ Macro-F1 0.377.
    ],

    [Linear probe], [
      ResNet18, SSL linear probe
    ], [
      SSL pool: trainfold images. SSL pretraining, frozen encoder, and a
      newly trained classifier head only.
    ], [
      BAcc 0.495.\ Macro-F1 0.388.
    ],
  )
  #set text(size: 11pt)

  The difference between fine-tuning and semi-SSL is important. Fine-tuning
  still learns only from true labels after the SSL stage. Semi-SSL adds a
  second student-training stage in which high-confidence predictions from an
  unlabeled pool are reused as pseudo-labels. In the current experiments that
  extra stage improved balanced accuracy from 0.776 to 0.837.

  == Behaviour on New Unlabeled ERP Candidates

  These rows are not performance metrics in the strict sense, because the
  candidate ERP images have no ground-truth labels. They are included here only
  as evidence for how the trained models behave when screening new data.
  The trained models come from the same mod-4 grouped 5-fold evaluation
  protocol as above.
  The numbers come from semi_supervised_learing2_unlabeled_summary.csv.

  #set text(size: 8.2pt)
  #appendix-result-table(
    [*Experiment*], [*Model*], [*Key setup*], [*Outcome*],

    [Screening], [
      ResNet18, supervised
    ], [
      Raw supervised baseline on the screened pool.
    ], [
      Pattern rate 0.000.\ Mean confidence 0.935.
    ],

    [Screening], [
      ResNet18, SSL fine-tune
    ], [
      SSL source: trainfold images. No pseudo-labeling in the final stage.
    ], [
      Pattern rate 0.065.\ Mean confidence 0.934.
    ],

    [Screening], [
      ResNet18, SSL fine-tune
    ], [
      SSL source: same-dataset mod-4 pool. No pseudo-labeling in the final
      stage.
    ], [
      Pattern rate 0.043.\ Mean confidence 0.964.
    ],

    [Screening], [
      ResNet18, SSL fine-tune + pseudo-labeling
    ], [
      Semi-SSL with the same-dataset mod-4 pool and a pseudo-label student
      stage.
    ], [
      Pattern rate 0.069.\ Mean confidence 0.965.
    ],
  )
  #set text(size: 11pt)
]

// Put your abbreviations/acronyms here.
// 'key' is what you will reference in the typst code
// 'short' is the abbreviation (what will be shown in the pdf on all references except the first)
// 'long' is the full acronym expansion (what will be shown in the first reference of the document)
//
// In the text, call @eeg or @uniS to reference  the shortcode
#let abbreviations = (
  (
    key: "eeg",
    short: "EEG",
    long: "Electroencephalography",
  ),
  (
    key: "uniS",
    short: "UoS",
    long: "University of Stuttgart",
  ),
)

#show: thesis.with(
  author: "Benjamin Borchert",
  title: "Automated Pattern Detection in ERP Images Using Convolutional Neural Networks",
  degree: "Bachelor of Science",
  faculty: "Faculty of Electrical Engineering and Computer Science",
  department: "Computational Cognitive Science",
  major: "Data Science",
  supervisors: (
    (
      title: "Examiner",
      name: "Benedikt Ehinger",
      affiliation: [Computational Cognitive Science \
        Faculty of Electrical Engineering and Computer Science, \
        Department of Computer Science
      ],
    ),
    (
      title: "Supervisor",
      name: "Vladimir Mikheev",
      affiliation: [Computational Cognitive Science \
        Faculty of Electrical Engineering and Computer Science, \
        Department of Computer Science
      ],
    ),
  ),
  epigraph: none,
  abstract: abstract,
  appendix: appendix,
  acknowledgements: acknowledgements,
  preface: none,
  figure-index: false,
  table-index: false,
  listing-index: false,
  abbreviations: abbreviations,
  date: datetime(year: 2026, month: 6, day: 1),
  bibliography: bibliography("refs.bib", title: "References", style: "ieee"),
)

// Code blocks
#codly(
  languages: (
    rust: (
      name: "Rust",
      color: rgb("#CE412B"),
    ),
    // NOTE: Hacky, but 'fs' doesn't syntax highlight
    fsi: (
      name: "F#",
      color: rgb("#6a0dad"),
    ),
  ),
)

// If you wish to use lining figures rather than old-style figures, uncomment this line.
// #set text(number-type: "lining")

// import custom utilities
#import "utils/general-utils.typ": *

#let pattern-decision-tree = figure(
  align(center)[
    #block(width: 124mm, height: 118mm)[
      #place(top + center, dx: -35mm, dy: 49mm)[
        #block(width: 66mm, height: 62mm, fill: rgb("#f5efe5"), radius: 2.5pt)[]
      ]
      #place(top + center, dx: 35mm, dy: 49mm)[
        #block(width: 66mm, height: 62mm, fill: rgb("#eaf1f8"), radius: 2.5pt)[]
      ]
      #place(top + center, dx: 2.5mm, dy: 1mm)[
        #set text(size: 6.8pt)
        #scale(72%, reflow: true)[
          #diagram(
            cell-size: 3.6mm,
            node-stroke: .6pt,
            edge-stroke: .6pt,
            node-fill: luma(98%),

            node((0, 0), align(center)[Observed\ ERP image], width: 17mm, corner-radius: 1.5pt),
            edge((0, 0), (0, 1.8), "-|>"),
            node(
              (0, 1.8),
              align(center)[Connected\ pattern\ visible?],
              shape: shapes.diamond,
              width: 20mm,
            ),

            edge((0, 1.8), (-2.8, 3.8), "-|>", [No], label-pos: 0.48),
            node((-2.8, 3.8), align(center)[No class], width: 18mm, corner-radius: 1.5pt),

            edge((0, 1.8), (0, 4.1), "-|>", [Yes], label-pos: 0.6),
            node(
              (0, 4.1),
              align(center)[Narrow\ in time?],
              shape: shapes.diamond,
              width: 18mm,
            ),

            edge((0, 4.1), (-2.3, 6.5), "-|>", [Yes], label-pos: 0.55),
            node(
              (-2.3, 6.5),
              align(center)[Opposed V\ shapes?],
              shape: shapes.diamond,
              width: 19mm,
            ),
            edge((-2.3, 6.5), (-3.1, 8.8), "-|>", [Yes], label-pos: 0.55),
            node((-3.1, 8.8), align(center)[Two-sided fan], width: 22mm, corner-radius: 1.5pt),
            edge((-2.3, 6.5), (-1.3, 8.8), "-|>", [No], label-pos: 0.55),
            node(
              (-1.3, 8.8),
              align(center)[Pinched at\ centre?],
              shape: shapes.diamond,
              width: 18mm,
            ),
            edge((-1.3, 8.8), (-2.1, 11.1), "-|>", [Yes], label-pos: 0.55),
            node((-2.1, 11.1), align(center)[Hourglass bar], width: 21mm, corner-radius: 1.5pt),
            edge((-1.3, 8.8), (-0.5, 11.1), "-|>", [No], label-pos: 0.55),
            node((-0.5, 11.1), align(center)[Diverging bar], width: 21mm, corner-radius: 1.5pt),

            edge((0, 4.1), (2.3, 6.5), "-|>", [No], label-pos: 0.45),
            node(
              (2.3, 6.5),
              align(center)[Single S-shaped\ band?],
              shape: shapes.diamond,
              width: 20mm,
            ),
            edge((2.3, 6.5), (1.3, 8.8), "-|>", [Yes], label-pos: 0.55),
            node((1.3, 8.8), align(center)[Sigmoid], width: 14mm, corner-radius: 1.5pt),
            edge((2.3, 6.5), (3.3, 8.8), "-|>", [No], label-pos: 0.55),
            node(
              (3.3, 8.8),
              align(center)[Opens to\ one side?],
              shape: shapes.diamond,
              width: 18mm,
            ),
            edge((3.3, 8.8), (2.8, 11.1), "-|>", [Yes], label-pos: 0.55),
            node((2.8, 11.1), align(center)[One-sided fan], width: 21mm, corner-radius: 1.5pt),
            edge((3.3, 8.8), (3.9, 11.1), "-|>", [No], label-pos: 0.55),
            node((3.9, 11.1), align(center)[No class], width: 18mm, corner-radius: 1.5pt),
          )
        ]
      ]
      #place(top + left, dx: 3mm, dy: 53mm)[
        #box(
          inset: (x: 1.6mm, y: 0.8mm),
          fill: rgb("#f5efe5"),
          radius: 1.5pt,
        )[
          #text(size: 7pt, weight: "semibold", fill: rgb("#6f5f43"))[Mostly vertical]
        ]
      ]
      #place(top + right, dx: -3mm, dy: 53mm)[
        #box(
          inset: (x: 1.6mm, y: 0.8mm),
          fill: rgb("#eaf1f8"),
          radius: 1.5pt,
        )[
          #text(size: 7pt, weight: "semibold", fill: rgb("#46627a"))[Extends across time]
        ]
      ]
    ]
  ],
  caption: [
    First draft of a manual decision tree for ERP image patterns. .
  ],
)

// ============================================================================
// Main Content
// NOTE: Explicit #pagebreak() between each chapter is required.
//
// INITIAL OUTLINE:
// Keep the structure broad for now. We can split chapters into more specific
// sections later once the experiments and text have stabilized.
// ============================================================================


// ----------------------------------------------------------------------------
// Chapter 1 - Introduce the problem, the thesis goal, and the high-level idea.
// ----------------------------------------------------------------------------
= Introduction <chp:introduction>

== Motivation
When a person sees a stimulus, reads a word, moves their eyes, or presses a
button, the brain response changes within fractions of a second. To study such
fast processes, we need a measurement that follows brain activity on the same
time scale. Electroencephalography (EEG) does this by recording electrical
activity from the scalp. For this introduction, this simple idea is enough;
Chapter 2 defines the terms more carefully.

The raw EEG signal is difficult to interpret because each single trial contains
both the response of interest and a large amount of unrelated activity. A common
solution is to repeat the same type of event many times, align the EEG to that
event, and average the trials. The result is an event-related potential (ERP).
This averaging approach is still central in human neuroscience, and ERP
research now spans many components, tasks, and application areas
@Kappenman2021 @Donoghue2022.

Averaging is useful, but it also hides information. Two datasets can have a
similar average even if the individual trials behave very differently. A
component may be present on most trials but appear blurred because its timing
changes. A response may be linked more closely to a later button press or the
next fixation than to the event used for alignment. Different trial subtypes
can also cancel each other when they are averaged too early
@Jung2001 @Ouyang2017LatencyVariabilityReview.

ERP images keep this missing structure visible. Instead of reducing all trials
to one waveform, they show the data as an image: rows are trials, columns are
time points, and colour represents signal amplitude. If the rows are sorted by
a meaningful variable, such as reaction time, fixation duration, or trial
order, structure across trials can become visible. A curved band may indicate
response-locked activity, a fan may indicate increasing timing variability, and
a vertical split may indicate a condition difference.

This visual structure is useful, but it still needs care. In eye-tracking EEG,
for example, neighbouring fixations and saccades can overlap with the analysed
response. A pattern in an ERP image can therefore be a clue to a cognitive
effect, to event overlap, or to a preprocessing issue. This is why ERP images
are best treated as a first diagnostic view, often followed by more explicit
overlap modelling or regression-based analysis @Dimigen2011Coregistration
@Ehinger2019Unfold.

The practical problem is scale. A researcher can inspect a few ERP images by
hand, but the number grows quickly with subjects, channels, conditions, and
sorting variables. Manual judgement is also hard to reproduce unless the
criteria are written down and applied consistently. Automated pattern
recognition is attractive because it can turn this slow visual screening step
into a consistent first pass. It does not replace interpretation; it helps
decide where interpretation should start.

This thesis focuses on that first pass. The goal is to detect interpretable
ERP-image morphology, not to explain the underlying brain process
automatically. The main experiments narrow the task to `sigmoid` versus
`no_class`, while the broader pattern vocabulary remains important for
simulation and annotation. Deep learning is a plausible tool for this
image-based task, but it needs labelled examples, and those labels are costly
to create manually @Roy2019.

The project therefore uses simulation to create labelled training data under
controlled assumptions. UnfoldSim.jl provides a principled basis for simulating
continuous event-based EEG-like time series, which can then be rendered as ERP
images @Schepers2025. The central question follows directly from this setup:
can a model trained on simulated ERP-image structure recognise the same kind of
sigmoid pattern in manually labelled real data?

== Research Questions and Contributions
// State the main research question, sketch the sim-to-real idea at a high
// level, and list the concrete contributions of the thesis.
// Mention the reduced scope here as part of the framing.

== Thesis Structure
// Briefly guide the reader through the remaining chapters.


// ----------------------------------------------------------------------------
// Chapter 2 - Provide the conceptual basis and position the thesis briefly in
// the literature without overloading the early draft.
// ----------------------------------------------------------------------------
#pagebreak()
= Background and Related Work <chp:background>

== EEG, ERPs, and ERP Images
Electroencephalography (EEG) is a method for recording electrical brain
activity from the scalp. In a typical human EEG experiment, an electrode cap is
placed on the participant's head, the electrodes are connected to an amplifier,
and each electrode records a voltage time series relative to a reference
electrode @Light2010. These voltages are small, indirect measurements. The
signal at one scalp electrode is not the activity of a single neuron, but a
macroscopic field signal shaped by many synchronised postsynaptic currents,
the geometry of the head, the chosen reference, and volume conduction through
brain tissue, skull, and scalp @Nunez2006 @Cohen2014. The strength of EEG is
therefore temporal rather than spatial: it can follow neural activity on the
scale of milliseconds. Its weakness is that each single trial mixes the
event-related response of interest with ongoing brain activity, eye movements,
muscle artefacts, line noise, and other measurement noise @Luck2014.

An event-related potential (ERP) is not recorded as a separate signal. It is
estimated from the continuous EEG by using the event markers of an experiment.
An event marker defines a meaningful time point, for example stimulus onset,
the start of a fixation, or a participant's response. Around each event, the
continuous EEG is cut into a short epoch, such as a pre-stimulus baseline
period followed by several hundred milliseconds after the event. Each epoch is
one trial. Standard ERP preprocessing then usually includes steps such as
filtering, artefact handling, and baseline correction, where the mean voltage
in the pre-event interval is subtracted from the epoch @Luck2014 @Light2010.
Classical ERP analysis aligns all epochs at time zero and averages them. Signal
components that occur consistently after the event remain visible in the
average, whereas activity that is not consistently time-locked is reduced by
averaging @Luck2014. ERP components such as P100, N170, and P300 are then
described by their polarity, latency, amplitude, and scalp distribution
@Luck2014 @Kappenman2021.

The average ERP is useful, but it deliberately removes single-trial variation.
Two datasets can have almost the same average waveform even if one contains a
systematic latency drift across trials and the other contains a stable response
with only random noise. Jung et al. make this issue explicit in their work on
single-trial ERP analysis: relevant event-related dynamics may be hidden by the
average because individual trials differ in latency, amplitude, and artefact
contamination @Jung2001. An ERP image keeps this single-trial information. It
is not a photograph of the brain and not a scalp topography. It is a
two-dimensional representation of one EEG channel or one extracted ERP signal:
the horizontal axis is time, the vertical axis is trial number, and the colour
at each cell encodes the voltage amplitude for one trial at one time point
@Jung2001.

The construction of an ERP image in this thesis follows the same conceptual
pipeline as the project proposal: EEG/ERP recording first, ERP images second,
pattern detection third. Starting from continuous EEG, the pipeline selects a
channel, extracts event-locked epochs, applies the same preprocessing choices
to all images, keeps the analysis window, and arranges the resulting data as a
trial-by-time matrix. The matrix can then be normalised, smoothed, cropped, or
resized so that it matches the input format of the CNN. The crucial step is
the ordering of the rows. If rows are left in recording order, the image mainly
shows the chronological sequence of trials. If rows are sorted by an
experimental variable, a behavioural variable such as reaction time, or a
simulated event parameter, the same voltage matrix can reveal whether the
response changes systematically with that variable. A latency shift can become
a slanted or curved band; an amplitude change can become a gradual colour
change; a categorical split can become two visually distinct blocks. ERP images
therefore preserve more information than a single averaged ERP waveform while
still giving a compact visual summary.

The next section uses this idea to define the ERP image patterns that appear in
the simulation and annotation workflow. The later experiments narrow the
classification task to `sigmoid` versus `no_class`, but that narrower task is
easier to understand after the broader pattern vocabulary is clear.

== ERP Image Patterns
The six pattern names used in this thesis are a practical vocabulary for visible
structure in ERP images. They are not six separate brain components. Sorting can
make different mechanisms visible: a component may shift in time, spread out
with variable duration, change polarity, or vary non-linearly across the trial
stack @Jung2001 @Delorme2004EEGLAB @Delorme2015GrandERPImage. The six labels
were chosen because they cover these main cases while staying distinct enough
for manual annotation.

In prose, the thesis uses readable names: sigmoid, diverging bar, one-sided
fan, hourglass, two-sided fan, and tilted bar. In code and annotation files,
the same labels are written as `sigmoid`, `diverging_bar`, `one_sided_fan`,
`hourglass`, `two_sided_fan`, and `tilted_bar`. This keeps the simulator,
manual labels, and classifier outputs consistent. It also makes `no_class`
clearer: it means that no coherent target pattern is visible, not merely that
the image is not sigmoid-like.

A `sigmoid` is a smooth, curved diagonal band. It often appears when epochs are
aligned to one event, but the relevant activity follows a later event, such as
a response, decision, or next fixation. In reaction-time-sorted ERP images,
early sensory activity can stay vertical while later positive activity follows
the response @Jung2001. In fixation-related data, a similar curve may also come
from overlap with neighbouring fixations, so the shape is a clue rather than a
finished interpretation @Dimigen2021RegressionEyeTrackingEEG.

A `tilted_bar` is a straight diagonal band. It has the same general origin as a
sigmoid, but the relation between trial order and component latency is roughly
linear. This may reflect response latency, trial order, fatigue, practice, or a
neighbouring event that shifts through the epoch. The visual label therefore
says that timing changes systematically; it does not decide why.

A `one_sided_fan` is narrow at one end of the trial stack and wider at the
other. This is expected when one boundary of the response is anchored, while
the other boundary depends on duration or on the timing of a following event.
Eye-tracking EEG is a typical case: variable fixation durations can create
duration-dependent overlap between successive fixation-related responses
@Dimigen2011Coregistration @Ehinger2019Unfold.

A `two_sided_fan` widens toward both extremes of the sorted stack. The component
usually keeps the same sign, but its timing is less stable for low and high
values of the sorting variable than near the middle. This fits the broader ERP
literature on trial-to-trial latency jitter, where variable component timing can
broaden averaged ERPs and hide sharper single-trial structure
@Ouyang2017LatencyVariabilityReview.

A `diverging_bar` is a mostly vertical band whose colour reverses across the
ordered trials. The timing is therefore fairly stable, but the sign changes
with the sorting variable. This can happen when trials are ordered by a signed
covariate, by prestimulus phase, or by conditions with opposite scalp polarity
@Haig1998AlphaPhaseP3 @Ritter2009PhaseSortingCaveats. In visual ERPs, polarity
can also depend on stimulus position and source geometry, as shown for early
retinotopic components @Capilla2016RetinotopicMapping.

An `hourglass` is a pinched or crossover-like pattern. It is usually not a
single canonical ERP mechanism, but a warning that the relation may be
non-linear, mixed across subgroups, or affected by overlap. This is the kind of
case where regression-based ERP or fixation-related potential analysis is more
appropriate than interpretation by eye alone @Ehinger2019Unfold
@Dimigen2021RegressionEyeTrackingEEG.

The later experiments focus on `sigmoid` versus `no_class`, but the full
six-pattern vocabulary matters. It defines what counts as meaningful ERP image
morphology before the classifier is trained. The model is therefore evaluated
as a detector of visible ERP image structure, not as an automatic explanation
of the underlying neural process.

== CNN-Based Pattern Detection in ERP Data
Convolutional neural networks are designed to learn local filters and combine
them across layers into increasingly abstract image features @LeCun2015. For
ERP images, this inductive bias is useful because the target is visibly
spatial: the model must detect short temporal segments, local contrast changes,
neighbouring trial rows, and finally larger connected shapes. A CNN is therefore
a plausible model for the same type of judgement made during manual labelling:
does the image contain a coherent pattern rather than unrelated local patches?

The limitation is that an ERP image is not a natural photograph. The horizontal
axis is time, the vertical axis is a sorted trial index, and the pixel values
come from a preprocessed electrophysiological signal. Sorting, z-scoring,
filtering, cropping, colour scaling, and resizing can all change what the CNN
sees. Roy et al. review deep-learning work on EEG and emphasise that
performance depends heavily on preprocessing, available labels, validation
design, and reproducibility @Roy2019. For this thesis, this means that a strong
CNN score is not automatically evidence of a neurophysiological pattern. It may
also indicate that the model has learned a simulator artefact, a preprocessing
artefact, or a dataset-specific shortcut.

The empirical task is therefore deliberately narrow. The model receives one
single-channel ERP image and predicts one of two labels: `sigmoid` or
`no_class`. The binary framing keeps the main transfer question clear: can a
shape learned from simulated sigmoid images still be recognised when the input
comes from manually labelled real ERP images?

== Data Simulation, Sim-to-Real Transfer, and Semi-Supervised Learning
The starting point for the simulation work is the label bottleneck. Manual
labelling requires a person to inspect an ERP image, decide whether a connected
pattern is visible, and separate the target pattern from noise or from other
visual structures. Simulation changes this trade-off: we choose the event
design, component timing, covariates, noise process, sorting variable, and
label before the image is generated. UnfoldSim is well suited to this role
because it simulates continuous event-based time series for EEG and related
signals, rather than only isolated averaged waveforms @Schepers2025.

The first project prototype used this idea to define a library of ERP-image
patterns. One simulated time-by-trial matrix was rendered several times, each
time with a different row ordering. The visible class was therefore determined
by the sorting variable:

- `sigmoid`: trials sorted by the latency difference to the next event, written
  as `Delta latency` in the event table;
- `one_sided_fan`: trials sorted by a duration covariate and combined with a
  lognormal time-varying basis;
- `two_sided_fan`: trials sorted by a second duration covariate and combined
  with a Hanning-shaped time-varying basis;
- `diverging_bar`: trials sorted by a categorical condition such as `car`
  versus `face`;
- `hourglass`: trials sorted by a continuous covariate that modulates the N170
  component;
- `tilted_bar`: trials sorted by a linear-duration covariate and combined with
  a shifted Hanning basis.

This design choice is important because the class is not encoded by a separate
image generator for each pattern. Instead, the same simulated ERP activity can
be made to reveal different structures through sorting. The later `no_class`
label follows the same logic. It is not an empty or all-noise image. It is an
ERP image created from the same simulated activity, but with a random trial
order, so that the systematic relation between trials and time is removed.

The later `ERPGen` implementation turns the prototype into a configurable
pipeline. A single repetition first samples global simulation settings:
lognormal onset parameters, trial count, sampling rate, and epoch duration.
It then samples P100-, N170-, and P300-like component parameters, including
widths, relative gaps, peak locations within each component window, and
amplitudes. These component names should be read as simulation anchors rather
than as claims that the generated signal is a full physiological model of each
ERP component @Luck2014. Additional time-varying components generate the
non-sigmoid pattern families. The noise pool contains pink, white, red, and
exponential noise, with a separate sampled level for each noise type.

The image part of `ERPGen` is equally important. After raw simulation, the
pipeline can drop trials, crop the time window, sort trials with a
pattern-specific rule, apply a Gaussian anti-aliasing low-pass filter before
downsampling, resize the matrix to the model input resolution, and apply
timepoint-wise z-scoring. It also creates four deterministic variants of each
image: normal trial order, reversed trial order, inverted polarity, and reversed
plus inverted. These variants are meant to discourage the model from treating a
single polarity or row direction as the definition of the class.

The empirical target was then narrowed to the binary task `sigmoid` versus
`no_class`. In the real-data matching configuration, the simulation was aligned
with the fixation-data preprocessing as closely as the current generator
allowed: the sampling rate was fixed at 512 Hz, the trial count was fixed to
2508 trials in the small-scale workflows, and the epoch duration was kept close
to one second so that the generated time axis matched the post-stimulus image
construction. The simulator still loaded the other pattern components, but only
the sigmoid sort and the random no-class sort became classifier labels. This
gives a cleaner test of the intended transfer question than a six-class
experiment would: can a model trained on simulated sigmoid structure recognise
the same visual concept in real ERP images?

The central risk is the sim-to-real gap. In simulation-to-real transfer, strong
synthetic performance does not guarantee real-data performance, because the
model may learn properties that are specific to the simulator. Domain
randomisation addresses this risk by varying simulator parameters so that the
real world appears as one possible variant within a broad synthetic
distribution @Tobin2017. In this thesis, the same principle is applied to ERP
images, but with a stricter constraint: if component widths, latency gaps,
amplitudes, basis shapes, or noise levels are varied too aggressively, the
target pattern itself becomes implausible or disappears.

The simulator search therefore had two goals: increase variation and remain
close enough to the real sigmoid morphology. The exposed search space contains
24 parameter specifications. Because each specification has a mean and a
standard deviation, the calibration problem becomes 48-dimensional. Three
sampling strategies were explored. Broad random search directly samples this
space and serves as a baseline. Latin hypercube sampling spreads candidates
more evenly across dimensions than ordinary random sampling @McKay1979. Monte
Carlo random search gives a simple and defensible comparison because random
search is a strong baseline for high-dimensional hyperparameter spaces
@Bergstra2012.

Two scoring ideas were then tested. The simulator-first workflow scores
candidates before full model training by comparing real and simulated sigmoid
images in a hand-designed feature space. The feature vector includes global
image standard deviation, mean absolute gradients along time and trial axes,
local patch-variance statistics, row autocorrelation, gradient concentration,
interquartile-range trends, and row/column energy variation. The distance score
combines z-normalised feature differences with a radial-basis-function maximum
mean discrepancy term, which is a kernel-based way to compare two sample
distributions @Gretton2012. The model-first workflow skips this proxy and
scores candidate simulator settings directly by real-data balanced accuracy.

Additional attempts were included because the first simulator settings did not
close the gap. A heterogeneity experiment increased parameter standard
deviations to 50%, 100%, and 1000% of the corresponding means to see whether
broader variation would produce more realistic image diversity. A dense neural
network with low-pass preprocessing was used as a model-first direct-search
alternative. These runs are important for the thesis even when they do not
produce the final model, because they show that the problem is not merely a
matter of sampling more synthetic examples. The shape and variability of the
simulated distribution matter.

Self-supervised and semi-supervised learning enter the project as a response to
this limitation. SimCLR-style contrastive learning trains an encoder by making
two augmented views of the same image agree while separating representations of
different images @Chen2020. Confidence-based pseudo-labelling then adds
unlabelled examples only when the current model predicts a label with high
confidence, a principle also used in later semi-supervised image methods such
as FixMatch @Sohn2020. In this thesis, these methods do not replace the
simulation question. They provide a second way to reduce dependence on
synthetic labels by learning from unlabelled real ERP images.

== Related Work and Positioning
The closest related work starts with single-trial ERP analysis. Grand averages
remain useful summaries, but they can hide latency jitter, response subtypes,
and systematic links between EEG and behaviour. ERP images and related
single-trial methods address this by keeping trial-wise structure visible
instead of reducing it to one waveform @Jung2001
@Pernet2011SingleTrialWhyBother @Ouyang2017LatencyVariabilityReview. This
thesis uses the same idea, but turns the visual inspection step into a
classification problem.

A second line of related work studies how overlapping events should be handled
in naturalistic EEG. In reading and free viewing, fixations and saccades occur
close together, so the response to one event can overlap with the response to
the next. Regression-based tools such as Unfold model this problem explicitly
@Dimigen2011Coregistration @Ehinger2019Unfold
@Dimigen2021RegressionEyeTrackingEEG. This matters here because a visible ERP
image pattern can reflect cognitive timing, event overlap, or both.

Deep learning provides the modelling background. CNNs have already been used
successfully for EEG decoding and for P300 detection, and EEG-specific models
such as DeepConvNet and EEGNet show that convolutional architectures can learn
useful temporal and spatial filters from electrophysiological data
@Cecotti2011P300CNN @Schirrmeister2017DeepConvNet @Lawhern2018EEGNet. The
present task is different. The model does not predict a stimulus class or a BCI
command, but whether an ERP image contains a named visual morphology.

The data regime is also different from most supervised EEG classification.
Manual labels for ERP image morphology are expensive because a person must
inspect an image and decide whether a pattern is present. Label Studio provides
a practical environment for that workflow, but the labels still depend on human
judgement @LabelStudio. This makes annotation quality important: agreement
statistics are needed when several raters label ambiguous visual categories
@Artstein2008InterCoderAgreement @Hallgren2012InterRaterKappa.

Simulation is one way to reduce this label bottleneck. EEG simulation tools
such as SEREEGA and UnfoldSim show that event-related EEG-like data can be
generated under controlled assumptions @Krol2018SEREEGA @Schepers2025. The
central risk is that a model trained on synthetic images may learn simulator
shortcuts instead of a pattern that also exists in real recordings. This is the
same broad problem addressed by sim-to-real transfer and domain adaptation:
the source and target distributions must be close enough for the learned
representation to transfer @Tobin2017 @Ganin2016DANN.

Self-supervised and semi-supervised learning offer a complementary path. Instead
of relying only on labelled synthetic images, an encoder can learn from
unlabelled ERP images and then use the smaller labelled set more efficiently.
This idea is well established in computer vision through contrastive learning
and pseudo-labelling, and it has also been explored directly for EEG
representations @Chen2020 @Sohn2020 @Banville2021SelfSupervisedEEG.

Taken together, the literature provides the pieces needed for this thesis:
ERP images make trial-wise structure visible, CNNs can learn from
electrophysiological inputs, simulation can provide labelled examples, and
semi-supervised methods can exploit unlabelled data. What is still missing is a
validated path that combines these pieces for ERP-image morphology itself. The
contribution of this thesis is therefore a focused simulation-to-real case
study: train on simulated `sigmoid` and `no_class` ERP images, then test whether
the learned visual concept survives in manually labelled real ERP images.


// ----------------------------------------------------------------------------
// Chapter 3 - Describe the empirical pipeline in one place.
// For an initial draft this is easier to navigate than splitting data and
// methods into separate chapters.
// ----------------------------------------------------------------------------
#pagebreak()
= Data and Methods <chp:data-methods>

== Datasets, Annotation, and Task Definition
At the time of writing, the project uses two real-data sources. The fixation TODO NEEDS SOURCE
dataset remains the main source for manual annotation and direct validation,
while an ERP CORE P3 derivative has been prepared as an unseen cross-domain
dataset with per-subject epochs and reaction times @Kappenman2021. MORE TO COME

The manual labelling workflow was first set up in Label Studio on 100
images and was later expanded to 400 additional unique images @LabelStudio. Labeling images with this Web user interface was more efficient that expected hence more images can be labelled in a shorter time. Moreover having more data to train and evaluate a model against with brings better robustness and genaralisation, as more pattern gestalts and noise can be used.

Figure @fig:pattern-decision-tree shows a first draft of the manual
pattern-labelling tree. It first asks whether a visible pattern is present and
then separates local patterns from patterns that extend across time.

#pattern-decision-tree <fig:pattern-decision-tree>

This is a working aid for annotation and not yet a final rule set.

== Data Simulation and Preprocessing
Several preprocessing were explored, but these runs were not
evaluated with the later reporting setup and are therefore not treated as
model results. The exploratory comparisons include different scalin methods: nearest-neighbour,
linear, quadratic, cubic, and Lanczos resizing, 

pipelines with and without Gaussian smoothing filtering, z-scoring before versus after resizing, value
binning in the ERP matrix, and input resolutions from `16x16` to `256x256`.

The real-data path currently sorts trials, applies per-timepoint z-scoring,
performs Gaussian low-pass filtering, and resizes the image to the model input
size. 

Additional explored data augmentation,
morphological operations, edge detectors, denoising, contrast
normalisation, gradient-based channels, and anti-aliased resizing. These
experiments were useful for narrowing the design space, but they did not
produce a retained improvement in classification performance at that stage.
RESULT TABEL

== Calibration, Models, and Training
Binary CNN baselines with
one, three, and ten convolutional layers, together with a pre-trained
`ResNet18`. 

Parameters to adjust: dependent component latencies, asymmetric window offsets, 38 in total. FOR NOW

broader parameter randomisation in the data simulation setup, together with
broad random search, Latin hypercube sampling, Monte Carlo single-parameter
search, and a two-zone mixture strategy, no gain. TABEL

Performance metrics: accuracy, balanced
accuracy, macro F1, precision, recall, and timing summaries under five-fold
cross-validation. This gives us a consistent framework for comparing model
families, preprocessing variants, and class-balancing strategies, as well as augmenting.

== Evaluation Protocol
Describe how models are evalueted, simulation -> real data path.

 binary labels and
`64x64` single-channel inputs for 500 manually labelled images FOR NOW. Its timing

Runntimes resulst
TABEL  extraction and initial preparation dominate the runtime at
41.87%, followed by low-pass filtering at 25.01% and z-scoring at 17.33%,
while resizing contributes only 2.28%.

Potential Bottelnecks


// ----------------------------------------------------------------------------
// Chapter 4 - Present observations in the same order as the pipeline.
// Keep this chapter descriptive; save interpretation for the discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Results <chp:results>

== Data Simulation and Calibration Results
TABEL

For now no result, model labels randomly

== Classification Performance Real Data
TABEL
This part already produced the clearest quantitative results. A first direct
validation of the current `64x64` binary CNN against the 500 manual labels
reached an accuracy of 0.284 and an F1 score of 0.175 for the pattern class,
which shows that the initial sim-to-real transfer is still weak.

Later five-fold experiments on the manually labelled data gave balanced
accuracy values of 0.500 for both `cnn_1conv` and `cnn_3conv`, 0.600445 for
`cnn_10conv`, and 0.849564 with `macro_f1 = 0.857006` for
`resnet18_pretrained` at `64x64`. A resolution sweep then showed that
`128x128` improved the same model to `balanced_accuracy = 0.905729` and
`macro_f1 = 0.896564`, whereas `256x256` was slower and less stable at
`balanced_accuracy = 0.868976`.

The real-data processing comparison also produced a first ranking of practical
changes. Per-image clipping to q01/q99 and scaling to `[-1, 1]` gave only
small gains over the `64x64` baseline, while balancing the classes improved the
current `resnet18_pretrained` setup to `balanced_accuracy = 0.897182` and
`macro_f1 = 0.895754`. By contrast, the best current two-channel filter
variant, based on a Laplacian second channel, remained slightly below the
simpler one-channel baseline with `balanced_accuracy = 0.841661` and
`macro_f1 = 0.85117`.

== Cross-Dataset Generalization
TODO MORE DATA


// ----------------------------------------------------------------------------
// Chapter 5 - Interpret the findings, state limitations honestly, and derive
// realistic next steps.
// ----------------------------------------------------------------------------
#pagebreak()
= Discussion <chp:discussion>

== Interpretation of the Main Findings


== Limitations
The main limitation of this thesis is the sim-to-real gap. The simulator can
generate labelled ERP images in large numbers, but it can only generate the
kind of variability that was built into it. Real EEG also contains
subject-specific responses, non-stationary noise, artefacts, imperfect event
timing, and preprocessing effects. This means that strong performance on
synthetic images is not enough to show that the model has learned a real
ERP-image pattern @Krol2018SEREEGA @Schepers2025. The weak first direct
transfer result should therefore be read as a genuine limitation of the setup,
not just as a tuning problem.

A related risk is shortcut learning. CNNs often use whichever visual regularity
is easiest for the training objective, even if that regularity is not the
intended concept @Geirhos2020ShortcutLearning. In this project, such shortcuts
could come from simulator-specific smoothness, noise texture, colour scaling,
or unusually clean pattern boundaries. Domain randomisation is meant to reduce
that risk by varying the synthetic world, but it cannot remove it completely
@Tobin2017. If randomisation is too narrow, the model overfits to the
simulator. If it is too broad, the target pattern itself becomes unrealistic.

The manual labels are another limitation. The real-data evaluation uses only a
limited number of manually labelled ERP images, and visual pattern labels are
not as objective as event markers or stimulus classes. Borderline cases can
reasonably be judged differently, especially when a weak sigmoid, a tilted bar,
or noisy overlap structure appear in the same image. Reliability measures are
normally used to quantify such disagreement when several raters label the same
material @Artstein2008InterCoderAgreement @Hallgren2012InterRaterKappa. Without
that type of agreement analysis, disagreement between the classifier and the
labels cannot always be interpreted as model error alone.

The interpretation of a detected pattern is also limited. An ERP image can show
a clear visual structure without revealing its cause. A sigmoid-like band may
reflect a response-locked component, overlap from a neighbouring fixation, a
sorting artefact, or a preprocessing choice. Work on fixation-related potentials
and regression-based overlap correction shows that event overlap can change ERP
morphology substantially @Dimigen2011Coregistration @Ehinger2019Unfold
@Dimigen2021RegressionEyeTrackingEEG. The classifier in this thesis sees only
the final image. It can detect morphology, but it cannot explain the underlying
neurocognitive mechanism.

The results are also conditional on the chosen preprocessing pipeline. Sorting,
z-scoring, smoothing, resizing, cropping, and channel construction all change
the image seen by the CNN. This is not a minor technical detail: ERP methods
research shows that reasonable processing choices can lead to different
measurements and conclusions @Clayson2021ERPMultiverse. In the experiments,
resolution, filtering, and channel variants also changed performance. The
reported metrics therefore describe one concrete representation pipeline, not
an intrinsic property of ERP images in general.

Finally, the empirical scope is deliberately narrow. The main task is binary:
`sigmoid` versus `no_class`. This makes the sim-to-real question easier to test,
but it does not establish performance for the full six-pattern vocabulary. The
negative class also mixes many different cases: random trial order, weak
patterns, noisy images, and possible non-sigmoid structures. As a result, the
current results cannot tell us whether the model would separate sigmoid from
hourglass, fan, or tilted-bar patterns in a multiclass setting.

These limitations do not make the study uninformative. They define what its
results can support. The thesis evaluates whether a simulated visual concept
can survive a first transfer to real labelled ERP images. It does not yet prove
general ERP-image pattern recognition across subjects, sessions, datasets, or
all possible pattern families.

== Future Work

// ----------------------------------------------------------------------------
// Chapter 6 - End with direct answers, not a second discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Conclusion <chp:conclusion>

// Summarize the core takeaway in a few sentences and answer the research
// question directly.
