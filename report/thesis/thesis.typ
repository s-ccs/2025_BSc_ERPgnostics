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
      contrastive learning following the Simple Framework for Contrastive
      Learning of Visual Representations (SimCLR) on ERP images.
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
      from an unlabelled pool.
    ],
    [Pseudo-labelling], [
      Predict labels for unlabelled ERP images and keep only the high-confidence
      predictions as temporary training labels.
    ],
    [Trainfold images], [
      Use only ERP images from the current training folds as SSL input. The
      validation fold stays unseen.
    ],
    [Same-dataset mod-4 pool], [
      A larger unlabelled ERP-image pool from the same fixation dataset,
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

  #pagebreak()

  == Supervised Augmentation and Imbalance Handling

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

  #pagebreak()

  == SSL and Semi-SSL Comparison

  #set text(size: 8.2pt)
  #appendix-result-table(
    [*Experiment*], [*Model*], [*Key setup*], [*Outcome*],

    [Semi-SSL + pseudo-labels], [
      ResNet18, SSL fine-tune + student stage
    ], [
      SSL pool: same-dataset mod-4 pool. Same SSL stage as above, then
      confidence-based pseudo-labelling with
      threshold 0.9; about 823 pseudo-labels kept per fold.
    ], [
      BAcc 0.837.\ Macro-F1 0.849.
    ],

    [Semi-SSL], [
      ResNet18, SSL fine-tune
    ], [
      SSL pool: same-dataset mod-4 pool. Simple Framework for Contrastive
      Learning of Visual Representations (SimCLR) pretraining on the
      larger same-dataset mod-4 pool, then
      full fine-tuning on true labels.
    ], [
      BAcc 0.776.\ Macro-F1 0.788.
    ],

    [Self-supervised], [
      ResNet18, SSL fine-tune
    ], [
      SSL pool: trainfold images. Simple Framework for Contrastive Learning of
      Visual Representations (SimCLR) pretraining on the current
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

  == Behaviour on New Unlabelled ERP Candidates

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
      SSL source: trainfold images. No pseudo-labelling in the final stage.
    ], [
      Pattern rate 0.065.\ Mean confidence 0.934.
    ],

    [Screening], [
      ResNet18, SSL fine-tune
    ], [
      SSL source: same-dataset mod-4 pool. No pseudo-labelling in the final
      stage.
    ], [
      Pattern rate 0.043.\ Mean confidence 0.964.
    ],

    [Screening], [
      ResNet18, SSL fine-tune + pseudo-labelling
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
// In the text, call entries such as @eeg or @cnn to reference the shortcode.
#let abbreviations = (
  (
    key: "bacc",
    short: "BAcc",
    long: "Balanced accuracy",
  ),
  (
    key: "bci",
    short: "BCI",
    long: "Brain-computer interface",
  ),
  (
    key: "cnn",
    short: "CNN",
    long: "Convolutional neural network",
  ),
  (
    key: "eeg",
    short: "EEG",
    long: "Electroencephalography",
  ),
  (
    key: "erp",
    short: "ERP",
    long: "Event-related potential",
  ),
  (
    key: "lhs",
    short: "LHS",
    long: "Latin hypercube sampling",
  ),
  (
    key: "macro-f1",
    short: "Macro-F1",
    long: "Macro-averaged F1 score",
  ),
  (
    key: "mmd",
    short: "MMD",
    long: "Maximum mean discrepancy",
  ),
  (
    key: "mri",
    short: "MRI",
    long: "Magnetic resonance imaging",
  ),
  (
    key: "osf",
    short: "OSF",
    long: "Open Science Framework",
  ),
  (
    key: "rbf",
    short: "RBF",
    long: "Radial basis function",
  ),
  (
    key: "resnet",
    short: "ResNet",
    long: "Residual neural network",
  ),
  (
    key: "semi-ssl",
    short: "Semi-SSL",
    long: "Semi-self-supervised learning",
  ),
  (
    key: "simclr",
    short: "SimCLR",
    long: "Simple Framework for Contrastive Learning of Visual Representations",
  ),
  (
    key: "ssl",
    short: "SSL",
    long: "Self-supervised learning",
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
      affiliation: [Institute for Visualization and Interactive Systems, Stuttgart Center for Simulation Science, University of Stuttgart
      ],
    ),
    (
      title: "Supervisor",
      name: "Vladimir Mikheev",
      affiliation: [Institute for Visualization and Interactive Systems, University of Stuttgart
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
    #block(width: 124mm, height: 132mm)[
      #place(top + center, dx: -34.5mm, dy: 48mm)[
        #block(width: 83mm, height: 83mm, fill: rgb("#f5efe5"), radius: 2.5pt)[]
      ]
      #place(top + center, dx: 44mm, dy: 48mm)[
        #block(width: 72mm, height: 83mm, fill: rgb("#eaf1f8"), radius: 2.5pt)[]
      ]
      #place(top + center, dx: 1.5mm, dy: 1mm)[
        #set text(size: 8.8pt)
        #scale(64%, reflow: true)[
          #diagram(
            cell-size: 3.6mm,
            node-stroke: .6pt,
            edge-stroke: .6pt,
            node-fill: luma(98%),

            node((0, 0), align(center)[Observed\ ERP image], width: 23mm, corner-radius: 1.5pt),
            edge((0, 0), (0, 1.8), "-|>"),
            node(
              (0, 1.8),
              align(center)[Connected\ pattern\ visible?],
              shape: shapes.diamond,
              width: 20mm,
            ),

            edge((0, 1.8), (-2.8, 3.6), "-|>", [No], label-pos: 0.48),
            node((-2.8, 3.6), align(center)[No class], width: 20mm, corner-radius: 1.5pt),

            edge((0, 1.8), (0, 3.6), "-|>", [Yes], label-pos: 0.6),
            node(
              (0, 3.6),
              align(center)[Drifts across\ time?],
              shape: shapes.diamond,
              width: 20mm,
            ),

            edge((0, 3.6), (2.00, 5.8), "-|>", [Yes], label-pos: 0.55),
            node(
              (2.00, 5.8),
              align(center)[Straight\ diagonal band?],
              shape: shapes.diamond,
              width: 22mm,
            ),
            edge((2.00, 5.8), (0.85, 8.0), "-|>", [Yes], label-pos: 0.55),
            node((0.85, 8.0), align(center)[Tilted bar], width: 20mm, corner-radius: 1.5pt),
            edge((2.00, 5.8), (3.05, 8.0), "-|>", [No], label-pos: 0.55),
            node(
              (3.05, 8.0),
              align(center)[S-shaped\ curved band?],
              shape: shapes.diamond,
              width: 22mm,
            ),
            edge((3.05, 8.0), (1.75, 10.6), "-|>", [Yes], label-pos: 0.55),
            node((1.75, 10.6), align(center)[Sigmoid], width: 17mm, corner-radius: 1.5pt),
            edge((3.05, 8.0), (3.30, 10.6), "-|>", [No], label-pos: 0.55),
            node((3.30, 10.6), align(center)[No class], width: 20mm, corner-radius: 1.5pt),

            edge((0, 3.6), (-2.2, 5.8), "-|>", [No], label-pos: 0.45),
            node(
              (-2.2, 5.8),
              align(center)[Opens or\ widens?],
              shape: shapes.diamond,
              width: 20mm,
            ),
            edge((-2.2, 5.8), (-3.05, 7.7), "-|>", [Yes], label-pos: 0.55),
            node(
              (-3.05, 7.7),
              align(center)[One-sided\ opening?],
              shape: shapes.diamond,
              width: 20mm,
            ),
            edge((-3.05, 7.7), (-3.85, 10.15), "-|>", [Yes], label-pos: 0.55),
            node((-3.85, 10.15), align(center)[One-sided fan], width: 24mm, corner-radius: 1.5pt),
            edge((-3.05, 7.7), (-2.70, 10.15), "-|>", [No], label-pos: 0.55),
            node((-2.70, 10.15), align(center)[Two-sided fan], width: 24mm, corner-radius: 1.5pt),

            edge((-2.2, 5.8), (-1.35, 8.0), "-|>", [No], label-pos: 0.55),
            node(
              (-1.35, 8.0),
              align(center)[Polarity changes\ across trials?],
              shape: shapes.diamond,
              width: 23mm,
            ),
            edge((-1.35, 8.0), (-1.45, 9.6), "-|>", [Yes], label-pos: 0.55),
            node(
              (-1.45, 9.6),
              align(center)[Pinched\ middle?],
              shape: shapes.diamond,
              width: 18mm,
            ),
            edge((-1.45, 9.6), (-2.15, 11.6), "-|>", [Yes], label-pos: 0.55),
            node((-2.15, 11.6), align(center)[Hourglass bar], width: 24mm, corner-radius: 1.5pt),
            edge((-1.45, 9.6), (-1.10, 11.6), "-|>", [No], label-pos: 0.55),
            node((-1.10, 11.6), align(center)[Diverging bar], width: 24mm, corner-radius: 1.5pt),
            edge((-1.35, 8.0), (-0.45, 8.85), "-|>", [No], label-pos: 0.42),
            node((-0.45, 8.85), align(center)[No class], width: 18mm, corner-radius: 1.5pt),
          )
        ]
      ]
      #place(top + center, dx: -57mm, dy: 49.7mm)[
        #box(
          inset: (x: 1.6mm, y: 0.8mm),
          fill: rgb("#f5efe5"),
          radius: 1.5pt,
        )[
          #text(size: 7pt, weight: "semibold", fill: rgb("#6f5f43"))[Vertical or widening]
        ]
      ]
      #place(top + center, dx: 64mm, dy: 49.7mm)[
        #box(
          inset: (x: 1.6mm, y: 0.8mm),
          fill: rgb("#eaf1f8"),
          radius: 1.5pt,
        )[
          #text(size: 7pt, weight: "semibold", fill: rgb("#46627a"))[Time-drifting]
        ]
      ]
    ]
  ],
  caption: [
    Manual decision tree for ERP image patterns. TODO more explanation
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
When a person sees a bird, reads a word, moves their eyes, or presses a button, the brain response changes within fractions of a second. To study such fast processes, a measurement that follows brain activity on a comparable time scale is needed. Electroencephalography (EEG) provides this temporal view by recording electrical activity from the scalp. EEG can follow rapid event-related dynamics, but the relevant structure is embedded in a noisy trial-by-trial signal. The raw EEG signal is difficult to interpret because each single trial contains a large amount of unrelated electrical activity, and may only contain a response of interest. A common solution is to repeat the same type of event multiple times, align the EEG to that event onset, and aggregate over trials, commonly by averaging to improve the signal-to-noise ratio @Light2010, @Luck2014. The result is an event-related potential (ERP). This averaging approach is still common practice in human neuroscience and ERP research @Kappenman2021, @Donoghue2022.

Averaging is useful, but it also hides information. Two datasets can have a similar average over trials even if the individual trials behave very differently. A component may be present on some input channels but may appear less distinct or absent in others. A response may be affected by a later button press or the next fixation than to the event used for alignment. Different groups of trials, such as fast and slow responses to stimuli, can also cancel each other out when they are averaged @Jung2001, @Ouyang2017LatencyVariabilityReview.

ERP images keep the otherwise aggregated structure visible. Instead of reducing all trials to one waveform, ERP images represent the data as a two-dimensional image. Rows correspond to trials, columns correspond to time points, and colour represents signal amplitude. If the rows are sorted by a meaningful experimental variable, such as reaction time or fixation duration, a potential structure across trials can become visible. A curved band may indicate response-locked activity, a fan may indicate increasing timing variability, and a vertical split may indicate a condition difference.

This visualisation is useful to extract more information than the common aggregating approaches. In eye-tracking EEG, neighbouring fixations and saccades can often occur close together in time. The response to one event may therefore overlap with the response to the next event. This can make ERP-image patterns appear shifted, widened, or harder to interpret. A pattern in an ERP image can therefore be a clue to a cognitive effect, to event overlap, or to a preprocessing issue. This is why ERP images are useful as a first diagnostic view, often followed by more explicit overlap modelling or regression-based analysis @Dimigen2011Coregistration, @Ehinger2019Unfold.

A practical problem is scale. A single experiment can produce many ERP images, resulting from different patients, channels, conditional variables, time windows, and processing choices such as sampling rate. A researcher can only inspect a limited number of ERP images manually, which may be insufficient to cover all of the recorded data of an experiment. Human annotation decisions are hard to reproduce unless the criterias are written down and applied consistently. Manually labelling data also carries the risk of false classification, which makes it necessary to compare results across multiple researchers. Automated pattern recognition is attractive because it can turn this slow visual screening step into a consistent first pass. It does not replace interpretation, but it aids decision making.

Automated recognition enables new broad-scale applications of what researchers can do with the patterns after they are found. It makes ERP images available as a quantitative variable for tasks such as using existing ERP datasets to detect previously unlabelled patterns, biomarker exploration to aid patient diagnostics, and comparison across datasets and experiments where manual inspection would otherwise cover only a small subset of images @Pernet2011SingleTrialWhyBother, @Kappenman2021.

Automation can also provide an additional quality check. An ERP recording and its data analysis include many heterogeneous processing steps and experimental setups, such as applied filters, signal references, artefact handling, experimental variables, and time windows. These choices can preserve a visible pattern, weaken it, or create a misleading one @Clayson2021ERPMultiverse. A detector can therefore flag ERP images from a faulty processing pipeline @Cecotti2011P300CNN.

If ERP-image screening is treated as a machine-learning task, labelled training data become the main bottleneck. Real ERP images are heterogeneous due to different experimental set-ups, and are mostly not labelled with regard to our interest in finding visual patterns. Manual annotation remains the preferred reference for real ERP images, but it is expensive and therefore usually too limited to train a deep image classifier from scratch for use in a productive medical workflow @Roy2019. Simulation complements these labels by creating many labelled examples under known assumptions, while real annotated data remain necessary for validation and transfer testing.

This thesis focuses on the automated approach to find visual patterns in ERP images. The goal is to detect interpretable ERP-image morphologies of interest, not to explain the underlying brain process or origin. Deep learning is a plausible tool for this image-based task, but only if the label bottleneck and the gap between simulated and real ERP images are handled explicitly.

The project therefore also uses simulation to create labelled training data under controlled assumptions. UnfoldSim.jl is used to simulate continuous event-based EEG-like time series, which can then be rendered as ERP images @Schepers2025. One central question follows directly from this setup: can a model trained on simulated ERP images recognise the same kind of pattern in manually labelled real data?

== Research Questions and Contributions

This thesis investigates whether visual ERP-image patterns can be detected automatically with convolutional neural networks, and whether simulated ERP images can reduce the need for manually labelled real data. The work is guided by the following research questions.


#[
  #set enum(
    numbering: (..nums) => strong[RQ#(nums.pos().map(str).join(".")):],
    full: true,
    tight: false,
    indent: 0pt,
    spacing: .7em,
    body-indent: .65em,
  )

  + *Sim-to-real transfer* How accurately and efficiently can a CNN model trained on simulated ERP images detect and distinguish relevant spatio-temporal patterns in real-world ERP images?

    + *Sim-to-real sigmoid* Can a CNN trained on simulated sigmoid ERP images recognise sigmoid patterns in manually labelled real ERP images?

    + *Simulator calibration* How does simulator calibration affect sim-to-real transfer and what performance gap remains compared with training on real labelled ERP images?
  
  + *Manual labeling* How accurately and efficiently can CNNs detect visual ERP-image patterns when they are trained on manually labelled real ERP images?

    + *Preprocessing* Which training and preprocessing choices improve CNN-based classification of manually labelled ERP images?

    + *Augmentation and class imbalance* Do ERP-specific augmentations or imbalance-handling strategies improve CNN model performance?

    + *Model choice and training strategy* Which model architecture and training regime provide the best accuracy-efficiency trade-off for manually labelled real ERP-image classification?

]

The main contributions of this thesis are:

1. A manually labelled real ERP-image dataset from multiple EEG/ERP sources, covering several visual pattern.
2. A simulation pipeline for generating controlled sigmoid and no-class ERP images, together with calibration experiments for sim-to-real transfer.
3. A comparison of preprocessing choices, augmentation, imbalance handling, model choice and model training set-ups.



== Thesis Structure

is it necessary?

// ----------------------------------------------------------------------------
// Chapter 2 - Provide the conceptual basis and position the thesis briefly in
// the literature without overloading the early draft.
// ----------------------------------------------------------------------------
#pagebreak()
= Background and Related Work <chp:background>

== EEG, ERPs, and ERP Images
Electroencephalography (EEG) is a method for recording electrical brain activity from the scalp. In a typical human EEG experiment, an electrode cap is placed on the participant's head, the electrodes are connected to an amplifier, and each electrode records a voltage over a time series @Light2010. The amplitudes of the recorded potentials are on the order of microvolts. The signal at one scalp electrode is not the activity of a single neuron, but a macroscopic field signal shaped by many local brain currents. The geometry of the head, the chosen reference baseline, and volume conduction through brain tissue, skull, and scalp all have an impact on the output signal @Nunez2006. The strength of EEG is therefore temporal rather than spatial: it can follow neural activity on the scale of milliseconds. Its weakness is that each single trial mixes the event-related response of interest with ongoing brain activity, eye movements, muscle artefacts, powerline noise, and other measurement noise @Luck2014.

An event-related potential (ERP) is not recorded as a separate signal. It is estimated from the continuous EEG by using the event markers of an experiment. An event marker defines a meaningful time point, for example stimulus onset, the start of a fixation, or a participant's response. Around each event, the continuous EEG is cut into a short epoch, usually a pre-stimulus baseline period followed by several hundred milliseconds after the event, commonly up to one second. Each epoch is referred to as a trial. Standard ERP preprocessing then usually includes steps such as high- or low-pass filtering, artefact handling, and baseline correction @Luck2014, @Light2010.
Common ERP analysis aligns all epochs at time zero and averages the trial amplitudes. Signal components that occur consistently after the event remain visible in the average, whereas activity that is not consistently time-locked, including noise, is reduced by averaging @Luck2014. ERP components such as P100, N170, and P300 are then described by their polarity, latency, amplitude, and scalp distribution @Luck2014, @Kappenman2021.

The averaged ERP signal is useful, but it deliberately removes single-trial variation. Two datasets can have almost the same average waveform even if one contains a systematic latency drift across trials and the other contains a stable response with only random noise. Jung et al. make this issue explicit in their work on single-trial ERP analysis. Relevant event-related dynamics may be hidden by the average because individual trials differ in latency, amplitude, and artefact contamination @Jung2001. An ERP image keeps this single-trial information. It is a two-dimensional representation of one EEG channel. The horizontal axis is time, the vertical axis lists trials stacked on top of each other, and the colour at each cell encodes the voltage amplitude for one trial at one time point @Jung2001.

A recent survey by Mikheev et al. about ERP visualisation practice supports the terminology for ERP images. The survey shows that ERP images are known in the community, but that researchers do not consistently use the same name for this plot type. Instead, the literature and practitioners use several related terms, such as sorted ERP trials or just trials, which can make comparisons between studies and tools harder. We therefore use the term ERP image consistently throughout this thesis @Mikheev2024ArtOfBrainwaves.

The ERP images in this thesis are built from already recorded EEG/ERP data. The data are assumed not to be raw recordings. Before this pipeline starts, muscle artefacts, eye-related artefacts, line noise, and other unusable segments should already have been handled. Noise from brain activity cannot be removed and remains the biggest obstacle. The task is therefore to turn preprocessed ERP data into ERP images for pattern detection.

To create an ERP image for the purpose of this thesis, the trials from a single channel are used. The data, often in the form of a time series, is cut into event-locked trials. Everything before the event is discarded. Only the time window from event start to at most one second after is kept, because the target patterns are expected in this interval rather than in longer recordings. The trials are arranged as a trial-by-time matrix, where each row is one trial and each column is one time point. This matrix can then be further processed.

== ERP Image Patterns
The six pattern names used in this thesis form a practical vocabulary for visible structure in ERP images. They are not six separate brain components. Sorting trials in the ERP-image matrix can make different mechanisms visible. A component may shift in time, spread out with variable duration, change polarity, or vary non-linearly @Jung2001, @Delorme2004EEGLAB, @Delorme2015GrandERPImage. The six pattern labels were chosen because they cover these main cases while staying distinct enough for manual annotation. A seventh no class option was added for images that do not clearly match any of the six patterns, so that the annotation scheme can also represent the absence of a target morphology.

#let erp-pattern-examples(pattern, left, right) = figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 3mm,
    image(left, width: 100%),
    image(right, width: 100%),
  ),
  caption: [Two manually labelled real-data ERP-image examples for #pattern.],
)

A sigmoid is a smooth, curved or S-shaped diagonal band. It can appear when epochs are aligned to one event, but the relevant activity is time-locked to another event whose latency varies across trials, such as a response or a subsequent fixation @Jung2001. In fixation-related data, a sigmoid-like curve may also result from overlap with neighbouring fixations @Dimigen2021RegressionEyeTrackingEEG.

#erp-pattern-examples(
  [sigmoid],
  "figures/erp_pattern_examples/01__sigmoid__fixations_dataset__ch096__duration.svg",
  "figures/erp_pattern_examples/02__sigmoid__eye_eeg_freeviewing__oz__fixation_duration_ms.svg",
) <fig:real-sigmoid-examples>

A tilted bar is a straight diagonal band in an ERP image. A component shifts monotonically in time across the sorted trial axis at an approximately constant rate. When trials are sorted by reaction time, this morphology can reflect response-locked components such as P100 or N100, or latency-graded activity whose timing follows the response such as P300. Single-trial ERP studies show that late P300-family latencies covary with reaction time @Jung2001, @Ouyang2017LatencyVariabilityReview, @Walsh2017P3bLatencyRT.

#erp-pattern-examples(
  [tilted bar],
  "figures/erp_pattern_examples/03__tilted_bar__roamm_reading__f8__gaze_x.svg",
  "figures/erp_pattern_examples/04__tilted_bar__erp_core_n170__c6__reaction_time_ms.svg",
) <fig:real-tilted-bar-examples>

A one-sided fan looks like a band that opens to only one side. The earlier border of the visible activity stays close to the time-locking event. The later border moves farther out for rows with longer fixation durations in the sorted trial stack. In fixation-related data, such a pattern may occur when the timing of the next fixation depends on the duration of the current fixation. The response evoked by the next fixation appears later in rows with longer fixation durations @Dimigen2011Coregistration, @Dimigen2021RegressionEyeTrackingEEG.

#erp-pattern-examples(
  [one-sided fan],
  "figures/erp_pattern_examples/05__one_sided_fan__eegeyenet_saccades__e61__saccade_duration_ms.svg",
  "figures/erp_pattern_examples/06__one_sided_fan__eegeyenet_saccades__e83__saccade_duration.svg",
) <fig:real-one-sided-fan-examples>

A two-sided fan is a pattern that narrows near the middle of the sorted trial stack and opens towards both ends. It consists of early vertical bands whose amplitude and polarity vary across the sorted trials. Its possible causes can overlap with other patterns@Jung2001, @Ouyang2017LatencyVariabilityReview, @Ehinger2019Unfold. In the manual annotations, the two-sided fan was the least frequently observed pattern class.


#erp-pattern-examples(
  [two-sided fan],
  "figures/erp_pattern_examples/07__two_sided_fan__fixations_dataset__ch125__sac_amplitude.svg",
  "figures/erp_pattern_examples/08__two_sided_fan__fixations_dataset__ch058__sac_amplitude.svg",
) <fig:real-two-sided-fan-examples>

A diverging bar is a vertical band whose polarity reverses across the sorted trial stack. Its timing stays fairly stable, so the visible cue is a polarity flip rather than a latency drift. It originates from a true polarity change across different categorical experimental variables @Wang2020PhotosensitivePhantom, @CecottiRies2017SingleTrialDetection, @Teixeira2018EvokedPatterns,
@KovalenkoBusch2016PerisaccadicVision, @Kohl2019InterleavedDeconvolution.

#erp-pattern-examples(
  [diverging bar],
  "figures/erp_pattern_examples/09__diverging_bar__unfold_face_freeview__p3__saccade_amplitude.svg",
  "figures/erp_pattern_examples/10__diverging_bar__unfold_face_freeview__o1__saccade_amplitude.svg",
) <fig:real-diverging-bar-examples>

An hourglass is a pinched ERP-image pattern. The activity is strong at the lower and upper ends of the sorted trial stack but weak or almost absent in the middle. The two ends usually have opposite polarity, which separates it from a continuous diverging bar. Such a pattern can arise from non-linear covariate effects, cancelling response subtypes or subject groups, or changing overlap between stimulus-, fixation-, and response-locked activity @Jung2001, @Ouyang2017LatencyVariabilityReview, @Woldorff1993ERPOverlap, @Ehinger2019Unfold, @Dimigen2021RegressionEyeTrackingEEG, @Mikheev2024ArtOfBrainwaves.

#erp-pattern-examples(
  [hourglass],
  "figures/erp_pattern_examples/11__hourglass__kilo_word_erp__fc5__number_of_letters.svg",
  "figures/erp_pattern_examples/12__hourglass__eegeyenet_saccades__e21__saccade_duration_ms.svg",
) <fig:real-hourglass-examples>

== CNN-Based Pattern Detection in ERP Data
Convolutional neural networks (CNNs) are designed to learn local filters and combine them across multiple layers into increasingly abstract image features @LeCun2015. For ERP images, the model must detect short temporal segments, local contrast changes, neighbouring trial rows, and finally larger connected shapes. A CNN is therefore a plausible model for this image-classification task.

ERP images are not natural photographs; they are more like a heatmap of a matrix. The horizontal axis is time, the vertical axis is a sorted trial index, and the pixel values are trial amplitudes. Roy et al. review deep-learning work on EEG and emphasise that performance depends heavily on preprocessing, available labels, validation design, and reproducibility @Roy2019. For this thesis, this means that a strong CNN score is not automatically evidence of a generalised and well-optimised model. It may also indicate that the model has learned a simulator artefact, a preprocessing artefact, or a dataset-specific shortcut such as noise or resolution.

Experience from the field of biomedical imaging supports the same point: neural-network performance depends not only on the architecture, but also on the used data pipeline. nnU-Net uses a U-Net-style encoder-decoder architecture, where the encoder compresses image information into increasingly abstract features and the decoder maps these features back to spatially resolved predictions @Isensee2021nnUNet. Brain-RetinaNet, in a small MRI tumour-detection setting, uses augmentation as a central response to limited labelled data @Iqbal2026BrainRetinaNet. These examples do not transfer because the medical tasks are the same, but because they show that preprocessing, augmentation, and validation choices can determine whether a machine-learning model learns a useful signal or a dataset-specific artefact.

== Data Simulation and Sim-to-Real Transfer

=== Motivation for Simulation
Labelled real ERP-image samples are scarce, which limits direct supervised training. Simulation provides a controlled way to generate labelled ERP data under known assumptions. The goal is not to reproduce full physiological EEG recordings, but to generate time-by-trial matrices in which known event timing, component timing, covariates, and noise create interpretable ERP-image patterns @Schepers2025.

=== Simulator Parameters
Each simulation run samples a set of global and component-level parameters. The global parameters define the lognormal event-onset process, the number of trials, the sampling rate, and the epoch duration. The component parameters define P100, N170, and P300 basis functions through their widths, Hanning window centres, relative gaps, peak offsets, and amplitudes. The component timings are partly dependent because later components are placed relative to earlier components. The N170 window follows the P100 window by a sampled gap, and analogously the P300 window follows the N170 window by another sampled gap. To create no-class images, the simulated trials are assigned a random trial order.

=== ERP-Image Output
The preprocessing settings are taken from the same fixation pipeline used for the labelled images. For each selected channel, the post-fixation interval is extracted, trials are sorted by the selected event metadata, each time point is z-scored across trials, a Gaussian low-pass filter is applied with reflective borders, and the resulting trial-by-time matrix is resized to 64x64. The size of the matrix is therefore not a property of the recording itself, but the final model-input resolution produced by preprocessing. It was chosen to make all ERP images comparable, keep training fast, and retain most of the important visual information.

For the sim-to-real evaluation, the generator is used only for the binary task
sigmoid versus no class. This restriction is a feasibility choice. The labelled
fixation dataset was the available real-data target, and sigmoid was both
frequent in that dataset and stable to create in the simulator.

The simulation settings therefore use this fixation dataset as the first target
distribution. That includes a sampling rate fixed at 512 Hz, a trial count of
2508, and an epoch length of one second; the simulated matrices are then passed
through the same image preprocessing to produce 64x64 single-channel images in
matrix form. The goal is to create ERP images that resemble real fixation ERP
images in their basic dimensions and preprocessing output, rather than arbitrary
simulator examples.

This setup leads directly to RQ 1.1. A CNN can only be useful here if the
simulated sigmoid/no-class distinction corresponds to the same visual
distinction in manually labelled real ERP images. High performance on simulated
images alone would only show that the generator creates a learnable synthetic
task. It would not show that the learned decision rule captures the intended
real ERP-image morphology.

The sim-to-real experiment therefore treats simulation as a proposed substitute
for part of the manual labelling effort. The model first learns from simulated
examples whose labels are known by construction, and is then evaluated on
manually labelled real ERP images. This makes the experiment a transfer test:
if the simulated images are close enough to the real sigmoid morphology, the
model should recognise the real pattern without being trained on real labels.
If the model instead relies on simulator-specific regularities, the transfer
performance will reveal that gap. This single-dataset comparison cannot prove
general transfer, but it establishes whether the simulation is close enough to a
real target to justify broader tests.

=== Simulated Class Definition
A six-class simulation setup was considered at first, but handling all visual patterns was infeasible. In the current simulation design, each simulated ERP image contains all previously mentioned components, so a single image can contain traces of several other classes. The origin of that decision is that the visual patterns are sensitive to small parameter changes. A slight change in timing, amplitude, or noise can move, deform, or remove a pattern, or produce a shape that no longer matches its usual visual description. Therefore, the final experiments use sigmoid as the only positive ERP-image pattern. It was the most robust against parameter changes of the six simulated morphologies and also the most frequent pattern in the available labelled fixation data.

This design choice is important because the class is not encoded by a separate image generator for each pattern. Instead, the same simulated ERP activity can be made to reveal different structures through sorting. The no-class label follows the same logic. It is not an empty or all-noise image. It is an ERP image created from the same simulated activity, but with a random trial order, so that the systematic relation between trials and time is removed.

=== Parameter Search
The parameter search aims to find simulator settings that make a CNN trained on synthetic images perform as well as possible on the labelled fixation dataset. For this purpose, this thesis applies and compares broad randomisation, Latin hypercube sampling, heterogeneity scaling, Monte Carlo random search, a two-zone mixture strategy, and feature-distance scoring @Tobin2017, @McKay1979, @Bergstra2012, @Gretton2012. The next section describes these search and scoring strategies in more detail.

UnfoldSim is well suited for this purpose because it simulates continuous event-based time series for EEG and event-related signals, rather than only isolated averaged waveforms @Schepers2025.

=== Sim-to-Real Gap
The central challenge is the sim-to-real gap. Strong synthetic performance does not guarantee real-data performance, because the model may learn properties that are specific to the simulator. Domain randomisation addresses this risk by varying simulator parameters so that the real world appears as one possible variant within the window of the synthetic distribution @Tobin2017. In this thesis, the same principle applies. If parameters such as component widths, latency gaps, amplitudes, basis shapes, or noise levels are varied too aggressively, the target pattern itself becomes implausible or disappears.

=== Search and Scoring Strategies
The goal is therefore to increase variation while remaining close enough to the real sigmoid morphology. The exposed search space contains 24 parameter specifications. Because each specification has a mean and a standard deviation, the calibration problem becomes 48-dimensional. Five search strategies and one simulator-first scoring proxy were explored.

Broad random search, or domain randomisation, samples the parameter space as a random exploration baseline, with each parameter given the same weight.

Latin hypercube sampling (LHS) divides continuous parameter ranges into discrete intervals and combines them through permutations, spreading candidates more evenly across dimensions than ordinary random sampling @McKay1979.

As a naive parameter-search attempt, a heterogeneity experiment scaled the standard deviations of normally distributed parameters to 50%, 100%, and 1000% of the corresponding mean magnitudes.

Monte Carlo random search provides a simple comparison by uniformly sampling one parameter at a time, and remains defensible because random search is a strong baseline for high-dimensional hyperparameter spaces @Bergstra2012.

The two-zone mixture uses 70% of the best parameters from the LHS baseline and 30% edge-case configurations.

Feature-distance scoring is a simulator-first proxy for judging simulator settings before full model training. It compares real and simulated sigmoid images with hand-designed image features and combines feature differences with an RBF-MMD distribution distance @Gretton2012. The resulting candidates are then compared through model-first ranking by real-data balanced accuracy.

== Related Work and Positioning
The closest related work starts with single-trial ERP analysis. Grand averages remain useful summaries, but they can hide latency jitter, response subtypes, and systematic links between EEG and behaviour. ERP images and related single-trial methods address this by keeping trial-wise structure visible instead of reducing it to one waveform @Jung2001, @Pernet2011SingleTrialWhyBother, @Ouyang2017LatencyVariabilityReview. This thesis uses the same idea, but turns the visual inspection step into a classification problem.

A second line of related work studies how overlapping events should be handled in naturalistic EEG. In reading and free viewing, fixations and saccades occur close together, so the response to one event can overlap with the response to the next. Regression-based tools such as Unfold model this problem explicitly @Dimigen2011Coregistration, @Ehinger2019Unfold, @Dimigen2021RegressionEyeTrackingEEG. This matters here because a visible ERP image pattern can reflect cognitive timing, event overlap, or both.

Magnostics evaluates hand-engineered descriptors for the related task of finding patterns in an adjacency matrix @Behrisch2017Magnostics. An ERP image is also an ordered matrix whose interpretable content can appear or disappear when rows are reordered. This thesis uses a CNN instead of a fixed descriptor library, but the model still operates on a particular ordered visual representation.

Deep learning provides the classification background. CNNs have already been used successfully for EEG decoding and for P300 detection. EEG-specific models such as DeepConvNet and EEGNet show that convolutional architectures can learn useful temporal and spatial filters from EEG data @Cecotti2011P300CNN, @Schirrmeister2017DeepConvNet, @Lawhern2018EEGNet. The present task is different. The model does not predict a stimulus class or a brain-computer interface (BCI) command, but whether an ERP image contains a named visual morphology.

Label Studio provides a practical environment for that workflow of manual labelling, but the labels still depend on human judgement @LabelStudio. This makes annotation quality important; agreement statistics help to quantify it @Artstein2008InterCoderAgreement, @Hallgren2012InterRaterKappa.

Self-supervised and semi-supervised learning offer a complementary path. Instead of relying only on labelled images, an image encoder can learn from unlabelled ERP images and then use the smaller labelled set more efficiently. Pretraining based on the Simple Framework for Contrastive Learning of Visual Representations (SimCLR) learns an image encoder from augmented views @Chen2020, and confidence-based pseudo-labelling adds unlabelled examples only when the model predicts them with high confidence @Sohn2020. This idea has also been explored directly for EEG representations @Banville2021SelfSupervisedEEG.

// ----------------------------------------------------------------------------
// Chapter 3 - Describe the empirical pipeline in one place.
// For an initial draft this is easier to navigate than splitting data and
// methods into separate chapters.
// ----------------------------------------------------------------------------
#pagebreak()
= Data and Methods <chp:data-methods>
This chapter describes the real and simulated data sources, the annotation workflow, the ERP-image preprocessing pipeline, the model-training setup, and the evaluation protocol.

== Datasets, Annotation, and Task Definition
The labelled real-data pool consists of the sources in @tab:real-data-sources. For datasets that already included preprocessed EEG or ERP files, this thesis uses those files directly instead of repeating the full preprocessing from the raw recordings. When a source was available only as raw material, it was prepared according to the authors' provided code or processing recommendations before being converted into the common ERP-image format used in this thesis. All sources were inspected manually before they were used for annotation and model training. The CNN therefore trains on preprocessed ERP images instead of raw EEG recordings.

#pagebreak()

#[
  #show figure: set block(breakable: true)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.45fr, 0.55fr, 2.25fr, 1.35fr, 1.25fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x == 1 or x == 3 or x == 4 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Dataset],
        [Year],
        [Topic],
        [Classified images],
        [Source],
      ),

      [Reference Fixations],
      [n/a],
      [Reference Fixation Dataset, fixation-locked EEG and eye-tracking reference data],
      [79 patterns out of 588 total],
      [TODO citation],

      [ERP CORE N170],
      [2021],
      [Face-processing epochs for the N170 component],
      [15 patterns out of 233 total],
      [@Kappenman2021, @ERPCOREN170OSF],

      [ERP CORE N2pc],
      [2021],
      [Visual-attention epochs for the N2pc component],
      [11 patterns out of 210 total],
      [@Kappenman2021, @ERPCOREN2pcOSF],

      [Kilo-Word ERP],
      [2015],
      [Kilo-Word ERP Database, lexical-decision ERPs with word-property variables],
      [12 patterns out of 253 total],
      [@Dufau2015KiloWordERP, @KiloWordERPOSF],

      [EYE-EEG Reading],
      [2011],
      [Natural reading with synchronised eye tracking and EEG],
      [2 patterns out of 245 total],
      [@Dimigen2011Coregistration, @Dimigen2021EyeEEGTestData],

      [EYE-EEG Freeviewing],
      [2021],
      [Free viewing and visual search with synchronised eye tracking and EEG],
      [15 patterns out of 200 total],
      [@Dimigen2021EyeEEGTestData],

      [EYE-EEG Sceneviewing],
      [2021],
      [Scene viewing with Tobii eye tracking and EEG],
      [3 patterns out of 27 total],
      [@Dimigen2021EyeEEGTestData],

      [EEGEyeNet Saccades],
      [2021],
      [Saccade-locked eye-movement data with EEG],
      [77 patterns out of 447 total],
      [@Kastrati2021EEGEyeNet, @EEGEyeNetOSF, @EEGEyeNetOpenNeuro],

      [ROAMM],
      [2025],
      [Reading Observed At Mindless Moments, natural reading with EEG, eye tracking, and attention annotations],
      [26 patterns out of 365 total],
      [@ROAMM2025, @ROAMMOSF],

      [Unfold Face FV],
      [2019],
      [Unfold face free-viewing data used in overlap-modelling examples],
      [54 patterns out of 291 total],
      [@Ehinger2019Unfold, @UnfoldFaceFreeViewingOSF],

      [NOD-EEG],
      [2025],
      [Natural Object Dataset EEG, visual object recognition in naturalistic scenes],
      [0 patterns out of 10 total],
      [@Zhang2025NODEEG, @NODEEGOpenNeuro],
    ),
    caption: [Labelled real-data sources used.],
  ) <tab:real-data-sources>

]

TODO add process, first overview, then sort by model confidence, select few samples to test if suitable for thesis, select 200 more and test what sorting variables can potentially create patterns. Keep sorting variables that contained at least one pattern and label all available channels from each of them.

#pattern-decision-tree <fig:pattern-decision-tree>

== Data Simulation and Preprocessing
Before training a classifier, each EEG/ERP recording must be converted into a fixed-size ERP image. This conversion shapes the actual input to the CNN models requirements. The classifier only receives the final image matrix, so resizing, smoothing, scaling, and filtering can preserve some visual structures, weaken others, or introduce artefacts that the model may learn. This thesis therefore evaluates these preprocessing choices empirically instead of treating them as neutral background steps.

The preprocessing comparison covers several parts of the image pipeline. It compares resizing with nearest-neighbour, linear, quadratic, cubic, and Lanczos interpolation. It also compares pipelines with and without Gaussian smoothing, z-scoring before versus after resizing, value binning in the ERP matrix, input resolutions from 16x16 to 256x256, and additional visual noise-reduction filters.

Trial sorting and z-scoring remain fixed because they define a usable ERP-image input for this task. Sorting makes cross-trial ERP-image patterns visible. Per-time-point z-scoring reduces vertical amplitude-dominated bands, which often occur when many trials show a strong response at the same time, mostly early after stimulus onset. These bands are not the main morphology of interest in this thesis. Z-scoring therefore emphasises relative differences between trials.

TODO create research questions for preprocessing steps, for each.

The best model performance was achieved with a preprocessing pipeline that sorts trials, applies per-time-point z-scoring, applies Gaussian smoothing, and resizes the image matrix.

todo results

This thesis applies the same empirical selection logic to the broader training pipeline. Some steps stay fixed for comparability, some depend on dataset properties, and others require experimental comparison. This follows the general lesson from nnU-Net @Isensee2021nnUNet. In this thesis, the compared training choices include the image pipeline, class balancing, augmentation, model training, filtering, denoising, contrast adjustment, and resizing. These comparisons help to choose preprocessing and training settings that improve model performance on normalised input data without artificially inflating scores through overfitting.


== Calibration, Models, and Training
The classifier comparison uses binary CNN baselines with one, three, and ten convolutional layers, together with a pretrained ResNet18 model @He2016ResNet. ResNet is useful here because residual connections make a deeper image classifier easier to optimise, while still providing a standard computer-vision baseline for comparison. Having the best practices already implemented, rather than figuring them out from scratch, makes it easier to focus on the classification and preprocessing of the ERP images.

Data augmentation improves model robustness when labelled training data are limited. This thesis uses augmentation to increase the variation of the labelled ERP-image training set and to reduce memorisation of individual examples. Brain-RetinaNet provides a domain-distant but useful reference point for this small-data setting. Iqbal et al. apply targeted augmentation to a small labelled MRI detection dataset and report improvements across several detector backbones @Iqbal2026BrainRetinaNet. The analogy only concerns the limited-data setting, but it motivates the evaluation of class-aware ERP-image augmentation and class balancing as central training decisions.

== Evaluation Protocol
The evaluation combines accuracy, balanced accuracy, macro F1, precision, recall, and timing summaries under grouped five-fold cross-validation. These metrics are used together because a single score would not capture overall performance, class imbalance, different error types, and computational cost.




// ----------------------------------------------------------------------------
// Chapter 4 - Present observations in the same order as the pipeline.
// Keep this chapter descriptive; save interpretation for the discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Results <chp:results>

== Data Simulation and Calibration Results
TABLE to do
per model, per explored adjustment
classification on simulated data is perfect, on real data lags behind the real approach.

== Classification Performance on Real Data
TABLE
for each model and the tested settings

== Cross-Dataset Generalisation
TODO MORE DATA
work in progress


// ----------------------------------------------------------------------------
// Chapter 5 - Interpret the findings, state limitations honestly, and derive
// realistic next steps.
// ----------------------------------------------------------------------------
#pagebreak()
= Discussion <chp:discussion>

== Interpretation of the Main Findings


== Limitations
The main limitation of this thesis is the sim-to-real gap. The simulator can generate labelled ERP images in large numbers, but it can only generate the kind of variability that was built into it. Real EEG also contains subject-specific responses, non-stationary noise, artefacts, imperfect event timing, and preprocessing effects. This means that strong performance on synthetic images is not enough to show that the model has learned a real ERP-image pattern @Krol2018SEREEGA, @Schepers2025. The weak transfer result should therefore be read as a genuine limitation of the setup, not just as a tuning problem.

A related risk is shortcut learning. CNNs often exploit whichever visual regularity is easiest for the training objective, even if that regularity is not the intended concept @Geirhos2020ShortcutLearning. In this project, such shortcuts could come from simulator-specific smoothness, noise texture, colour scaling, or unusually clean pattern boundaries. Domain randomisation is meant to reduce that risk by varying the synthetic world, but it cannot remove it completely @Tobin2017. The same applies to the chosen augmentation steps: all of the tested CNN models can detect and perfectly decide whether a high- or low-resolution matrix was scaled down to the same target size.

The manual labeling is another limitation. The real-data evaluation uses only a limited number of manually labelled ERP images, and visual pattern labels are not as objective as event markers or stimulus classes. Borderline cases can reasonably be judged differently, for instance when a weak and noisy sigmoid appears. Reliability measures are normally used to quantify such disagreement when several raters label the same material @Artstein2008InterCoderAgreement, @Hallgren2012InterRaterKappa. Without that type of agreement analysis, disagreement between the classifier and the labels cannot always be interpreted as model error alone.

The results are also conditional on the chosen preprocessing pipeline. Sorting, z-scoring, smoothing, and resizing all change the image seen by the CNN. This is not a minor technical detail. ERP methods research shows that reasonable processing choices can lead to different measurements and conclusions @Clayson2021ERPMultiverse.

TODO add more, computational power is fine. Maybe the ones from midterm and proposal.

== Future Work
One useful next step is a non-CNN baseline for the same ordered-matrix task. TODO transformer model

A second extension is localisation. The current classifier assigns one label to an entire ERP image, so it cannot mark where a pattern starts, ends, or overlaps with another structure. Detection-style biomedical imaging work such as Brain-RetinaNet shows how convolutional models can move from image-level classification towards localising relevant regions @Iqbal2026BrainRetinaNet. For ERP images, such a shift would require labels for pattern extents in trial-time space, not only image-level labels.

Augmentation is very important. One option is to simulate data for a specific real dataset.

// ----------------------------------------------------------------------------
// Chapter 6 - End with direct answers, not a second discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Conclusion <chp:conclusion>

// Summarize the core takeaway in a few sentences and answer the research
// question directly.
