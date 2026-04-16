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
If we want to understand how the brain reacts to what a person sees, hears, or
does, we need measurements that are fast, practical, and repeatable. EEG is
one such method: it records electrical activity at the scalp, and ERPs are
patterns that become visible when we align those recordings to repeated events.
For the purpose of this introduction, this rough intuition is enough. Chapter 2
explains both concepts more carefully. ERPs remain widely used in human
neuroscience, and the literature now spans a broad range of paradigms and
application areas @Kappenman2021 @Donoghue2022.

As soon as these experiments become larger, the analysis problem changes.
Researchers may have many recordings, but deciding which patterns are
meaningful still takes time and expert judgment. This is why automation is
attractive: deep learning can learn structured patterns from complex signals,
but it still depends on labelled examples that are rarely available at the same
scale as the raw recordings @Roy2019.

This thesis starts from that bottleneck. Instead of waiting for large manually
annotated datasets, we use data simulation to create labelled ERP images under
controlled assumptions. Tools such as UnfoldSim.jl make that possible and shift
the central question to the point that matters in practice: can a model
trained on ERP images from data simulation still recognise the relevant
sigmoid pattern in real recordings @Schepers2025.

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

In this thesis, the central positive class is not any ERP component in general
but a specific ERP-image shape: the `sigmoid` pattern. Operationally, an image
is treated as sigmoid-like when a connected S-shaped band extends across time
while moving smoothly across the sorted trial axis. The label is therefore tied
to a relation between three elements: the time course of the ERP response, the
variable used to order the trials, and the visual continuity of the band. This
distinction matters for the later simulation. A single simulated time-by-trial
matrix can produce several different ERP images, and therefore several
different apparent pattern classes, if the trials are sorted by different event
variables.

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

== Research Gap and Positioning
The gap addressed in this thesis is the lack of a validated path from
simulation-generated ERP-image labels to reliable pattern recognition in real
ERP images. ERP-image visualisation makes single-trial structure visible
@Jung2001, EEG deep learning offers models that can learn from complex inputs
@Roy2019, and UnfoldSim provides a principled way to generate event-based
synthetic EEG-like time series @Schepers2025. What remains uncertain is whether
these pieces can be combined without the classifier learning simulator-specific
shortcuts.

The thesis therefore contributes an explicit simulation-to-real case study for
ERP-image pattern detection. It defines the simulated `sigmoid` and `no_class`
labels operationally, documents how generated time series become model-ready
ERP images, explores several ways to randomise and calibrate the simulator, and
then checks the resulting models against manually labelled real ERP images. The
main scientific question is not whether synthetic data can be classified. The
simulator can make that task easy. The question is whether the learned visual
concept survives the move from controlled simulation to real EEG data.


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

== Future Work

// ----------------------------------------------------------------------------
// Chapter 6 - End with direct answers, not a second discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Conclusion <chp:conclusion>

// Summarize the core takeaway in a few sentences and answer the research
// question directly.
