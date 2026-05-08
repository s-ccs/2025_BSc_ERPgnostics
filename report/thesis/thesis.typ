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
When a person sees a bird, reads a word, moves their eyes, or presses a button, the brain response changes within fractions of a second. To study such fast processes, a measurement that follows brain activity on a comparable time scale is needed. Electroencephalography (EEG) provides this temporal view by recording electrical activity from the scalp. EEG can follow rapid event-related dynamics, but the relevant structure is embedded in a noisy trial-by-trial signal.The raw EEG signal is difficult to interpret because each single trial contains a large amount of unrelated electrical activity, and only may contain a response of interest. A common solution is to repeat the same type of event multiple times, align the EEG to that event on-set, and aggregate over trials, commonly averageing to improve the signal-to-noise ratio @Light2010, @Luck2014. The result is an event-related potential (ERP).This averaging approach is still common practice in human neuroscience and ERPresearch @Kappenman2021, @Donoghue2022.

Averaging is useful, but it also hides information. Two datasets can have a similar average over trials even if the individual trials behave very differently. A component may be present on most input channels but appear blurred or absend in others. A response may be linked more closely to a later button press or the next fixation than to the event used for alignment. Different trial subtypes can also cancel each other when they are averaged @Jung2001, @Ouyang2017LatencyVariabilityReview.

ERP images keep the otherwise aggregateed structure visible. Instead of reducing all trials to one waveform, they show the data as an image. Rows are trials, columns are time points, and colour represents signal amplitude. If the rows are sorted by a meaningful experimental variable, such as reaction time or fixation duration a potential structure across trials can become visible. A curved band may indicate response-locked activity, a fan may indicate increasing timing variability, and a vertical split may indicate a condition difference.

This visualisation is useful to extract more information then over the common aggregating approaches. In eye-tracking EEG, for example, neighbouring fixations and saccades can overlap with the analysed response. A pattern in an ERP image can therefore be a clue to a cognitive effect, to event overlap, or to a preprocessing issue. This is why ERP images are useful  as a first diagnostic view, often followed by more explicit overlap modelling or regression-based analysis @Dimigen2011Coregistration, @Ehinger2019Unfold.

A practical problem is scale. A single experiment can produce many ERP images resultig of different patients, channels, conditional variables, time windows, and processing choices like samplimg rate. A researcher can only inspect a limited amount of ERP images manually, which may be insufficient to cover all of the recorded data of an expriment. Manual judgement is also hard to reproduce unless the criteria are written down and applied consistently. Manualy labelling data may also contain the risk of false classification, which makes comapring results of multiple reseachers necessary. Automated pattern recognition is attractive because it can turn this slow visual screening step into a consistent first pass. It does not replace interpretation, but it aids for desicion making.

Automated recognition enabels new broad scale applications of what researchers can do with the patterns after they are found. This makes ERP images available as a quantitative variable such as: analysing existing erp data for a retrospective study, biomarker exploration for patients to aid in diagnostic, and comparison across datasets and experiments where manual inspection would otherwise cover only a small subset of images @Pernet2011SingleTrialWhyBother, @Kappenman2021.

Automation is also useful as a quality-control signal. An ERP recording and its data analysis include many heterogeneous processing steps and experimental setups such as applied filters, aplitude references, artefact handling, experimental vraiables, and time windows. These choices can preserve a visible pattern, weaken it, or create a misleading one @Clayson2021ERPMultiverse. A detector can therefore flag ERP images form a faulty processing pieline @Cecotti2011P300CNN.

Once this screening step becomes a machine-learning problem, the bottleneck shifts. Real ERP images are heterogeneous due to different experimental set-ups, and are mostly not labelled for our interest of finding visual patterns. Manual annotation can provide a small evaluation set, but it is a weak foundation for training a deep image classifier from scratch to use in a productive medical workflow @Roy2019. Simulation offers a way to create many labelled examples under known assumptions, while real data remain necessary for testing whether the learned visual concept transfers across data sets.

This thesis focuses on the automated approach to find visual patterns in ERP images. The goal is to detect interpretable ERP-image morphology of intreset, not to explain the underlying brain process or origin. Deep learning is a plausible tool for this image-based task, but only if the label bottleneck and the gap between simulated and real ERP images are handled explicitly.

The project therefore also uses simulation to create labelled training data under controlled assumptions. UnfoldSim.jl is used for simulating continuous event-based EEG-like time series, which can then be rendered as ERP images @Schepers2025. One central question follows directly from this setup: can a model trained on simulated ERP-images recognise the same kind of pattern in manually labelled real data?

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
Electroencephalography (EEG) is a method for recording electrical brain activity from the scalp. In a typical human EEG experiment, an electrode cap is placed on the participant's head, the electrodes are connected to an amplifier, each electrode records a voltage over a time series @Light2010. The aplitudes of the recorded potentials are in the magnitude of microvolts. The signal at one scalp electrode is not the activity of a single neuron, but a macroscopic field signal shaped by many local brain currents. The geometry of the head, the chosen reference baseline, and volume conduction through brain tissue, skull, and scalp have an impact on the output signal @Nunez2006. The strength of EEG is therefore temporal rather than spatial, it can follow neural activity on the scale of milliseconds. Its weakness is that each single trial mixes the event-related response of interest with ongoing brain activity, eye movements, muscle artefacts, powerline noise, and other measurement noise @Luck2014.

An event-related potential (ERP) is not recorded as a separate signal. It is estimated from the continuous EEG by using the event markers of an experiment. An event marker defines a meaningful time point, for example stimulus onset, the start of a fixation, or a participant's response. Around each event, the continuous EEG is cut into a short epoch, usually as a pre-stimulus baseline period followed by several hundred milliseconds after the event, commonly up to one second. Each epoch is refered to as a trial. Standard ERP preprocessing then usually includes steps such as high- or low pass filtering, artefact handling, and baseline correction @Luck2014 @Light2010.
Common ERP analysis aligns all epochs at time zero and average the trial aplitudes. Signal components that occur consistently after the event remain visible in the average, whereas activity that is not consistently time-locked, including noise is reduced by averaging @Luck2014. ERP components such as P100, N170, and P300 are then described by their polarity, latency, amplitude, and scalp distribution @Luck2014, @Kappenman2021.

The averageed ERP signal is useful, but it deliberately removes single-trial variation. Two datasets can have almost the same average waveform even if one contains a systematic latency drift across trials and the other contains a stable response with only random noise. Jung et al. make this issue explicit in their work on single-trial ERP analysis. Relevant event-related dynamics may be hidden by the average because individual trials differ in latency, amplitude, and artefact contamination @Jung2001. An ERP image keeps this single-trial information. It is a two-dimensional representation of one EEG channel. The horizontal axis is time, the vertical axis are trials stacked on top of each other, and the colour at each cell encodes the voltage amplitude for one trial at one time point @Jung2001.

A recent survey by Mikheev et al. about ERP visualisation practice supports the terminology for ERP images. The survey shows that ERP images are known in the community, but that researchers do not consistently use the same name for this plot type. Instead the literature and practitioners use several related terms, which can make comparisons between studies and tools harder, such as sorted ERP trials or just trials. We therefore use the term ERP image consistently throughout this thesis @Mikheev2024ArtOfBrainwaves.

The ERP images in this thesis are built from already recorded EEG/ERP data. The data are assumed not to be raw recordings. Before this pipeline starts, muscle artefacts, eye-related artefacts, line noise, and other unusable segments should already have been handled. Noise from brain activity cannot be removed and remains the biggest obstacle. The task is therefore to turn preprocessed ERP data into ERP images for pattern detection.

To create an ERP image for the purpose of this thesis, the trials from a single channel are used. The data often is in form of a time series, is cut into event-locked trials. Everything before the event is discarded. Only the time window from event start to at most one second after is kept, because the target patterns are expected in this interval rather than in longer recordings. The trials are arranged as a trial-by-time matrix, where each row is one trial and each column is one time point. This matrix can then be fruther be processed.

== ERP Image Patterns
The six pattern names used in this thesis are a practical vocabulary for visible structure in ERP images. They are not six separate brain components. Sorting trials in the ERP image matrix can make different mechanisms visible. A component may shift in time, spread out with variable duration, change polarity, or vary non-linearly @Jung2001, @Delorme2004EEGLAB, @Delorme2015GrandERPImage. The six labels were chosen because they cover these main cases while staying distinct enough for manual annotation.

A sigmoid is a smooth, curved or S-shaped diagonal band. It can appear when epochs are aligned to one event, but the relevant activity is time-locked to another event whose latency varies across trials, such as a response or a subsequent fixation @Jung2001. In fixation-related data, a sigmoid-like curve may also result from overlap with neighbouring fixations @Dimigen2021RegressionEyeTrackingEEG.

A tilted bar is a straight diagonal band in an ERP image. A component shifts monotonically in time across the sorted trial axis at an approximately constant rate. When trials are sorted by reaction time, this morphology can reflect response-locked like P100 or N100, or latency-graded activity whose timing follows the response like P300. Single-trial ERP studies show that late P300-family latencies covary with reaction time @Jung2001, @Ouyang2017LatencyVariabilityReview, @Walsh2017P3bLatencyRT.

A one-sided fan looks like a band that opens to only one side. The earlier border of the visible activity stays close to the time-locking event. The later border moves farther out for rows with longer fixation durations in the sorted trial stack. In fixation-related data, such pattern may reflect duration-dependent overlap. The response evoked by the next fixation appears later in rows with longer fixation durations @Dimigen2011Coregistration, @Dimigen2021RegressionEyeTrackingEEG.

A two-sided fan is a pattern that narrows near the middle of the sorted trial stack and opens toward both ends. It consists of early vertical bands whose amplitude and polarity vary across the sorted trials. Such patterns can arise from latency variability, mixtures of trial types or components, or temporal overlap from neighbouring events @Jung2001, @Ouyang2017LatencyVariabilityReview, @Ehinger2019Unfold.


A diverging bar is a vertical band whose polarity reverses across the sorted trial stack. Its timing stays fairly stable, so the visible cue is a polarity flip rather than the latency drift. It originates from a true polarity change across different categorical experimental variabels @Wang2020PhotosensitivePhantom,@CecottiRies2017SingleTrialDetection, @Teixeira2018EvokedPatterns,
@KovalenkoBusch2016PerisaccadicVision, @Kohl2019InterleavedDeconvolution.

An hourglass is a pinched ERP-image pattern. The activity is strong at the lower and upper ends of the sorted trial stack but weak or almost absent in the middle. The two ends usually have opposite polarity, which separates it from a continuous diverging bar. Such a pattern can arise from non-linear covariate effects, cancelling response subtypes or subject groups, or changing overlap between stimulus-, fixation-, and response-locked activity @Jung2001, @Ouyang2017LatencyVariabilityReview, @Woldorff1993ERPOverlap, @Ehinger2019Unfold, @Dimigen2021RegressionEyeTrackingEEG, @Mikheev2024ArtOfBrainwaves.

== CNN-Based Pattern Detection in ERP Data
Convolutional neural networks (CNN) are designed to learn local filters and combine them across multiple layers into increasingly abstract image features @LeCun2015. For ERP images the model must detect short temporal segments, local contrast changes, neighbouring trial rows, and finally larger connected shapes. A CNN is therefore a plausible model for the purpose of image classification.

Magnostics evaluates their own hand-engineered descriptors for this task of finding patterns in a adjacency matrix @Behrisch2017Magnostics. An ERP image is also an ordered matrix whose interpretable content can appear or disappear when rows are reordered. This thesis uses a CNN instead of a fixed descriptor library, but the model still operates on a particular ordered visual representation.

ERP images are not natural photographs, they are more of a heatmap of a matrix. The horizontal axis is time, the vertical axis is a sorted trial index, and the pixel values are trial amplitude. Roy et al. review deep-learning work on EEG and emphasise that performance depends heavily on preprocessing, available labels, validation design, and reproducibility @Roy2019. For this thesis, this means that a strong CNN score is not automatically evidence of a generalised and well optimised model. It may also indicate that the model has learned a simulator artefact, a preprocessing artefact, or a dataset-specific shortcut such as noise or resulution.

Biomedical imaging makes the same point from a different direction. nnU-Net uses U-Net-like templates but auto configures preprocessing, network structure, training, and post-processing based dataset properties and validation results @Isensee2021nnUNet. Brain-RetinaNet, in a small MRI tumour-detection setting, that also uses augmentation as a central response to limited labelled data @Iqbal2026BrainRetinaNet. For this thesis, the transferable point is not the medical task itself, but the priority given to data representation, augmentation, and pipeline configuration.

== Data Simulation, Sim-to-Real Transfer, and Semi-Supervised Learning
The starting point for the simulation work is the label bottleneck. Simulation provides a controlled way to generate labelled ERP data. The goal is not to reproduce full physiological EEG recording, but to generate time-by-trial matrices in which known event timing, component timing, covariates, and noise create interpretable ERP-image patterns @Schepers2025. This enabels a method to deal with scarce real data samples.

Each simulation repetition samples a set of global and component-level parameters. The global parameters define the lognormal event-onset process, the number of trials, the sampling rate, and the epoch duration. The component parameters define P100, N170, and P300 basis functions through their widths, hanning window centres, relative gaps, peak offsets, and amplitudes. The component timing is partly dependent by construction. The N170 window follows the P100 window by a sampled gap, analog the P300 window follows the N170 window by another sampled gap. In order to create no class images, the simulated trials are assigned a random trial order.

The preprocessing settings are taken from the same fixation pipeline used for the labelled images. For each selected channel, the post-fixation interval is extracted, trials are sorted by the selected event metadata, each time point is z-scored across trials, a Gaussian low-pass filter is applied with reflective borders, and the resulting trial-by-time matrix is resized to 64x64. The size of the matrix is therefore not a property of the recording itself, but the final model-input resolution produced by preprocessing. It was used to make all ERP images comparable, keep training fast, and retain most of the important visual information.

The main sim-to-real experiments narrow this generator to the binary task of  sigmoid versus no class. This choice was driven by the available real data set at that time. At the beginning of the simulation work, the fixation dataset was already available and defined the real validation target. The simulation settings therefore imitate this dataset as closely as the current generator allowed. That inlcude the sampling rate is fixed to 512 Hz, the trial count to 2508, the epoch length to one second, and the simulated matrices are passed through the same image preprocessing to produce 64x64 single channel images in matrix form. The goal was to create ERP images that resemble the real fixation ERP images in their basic dimensions and preprocessing output, rather than arbitrary simulator examples.

TODO Reaseach question is it possible to simulate data and make a cnn overfit to real data? Result it does not. 

A six-class simulation setup was considered at first, but handling all visual patterns was unfeasible. In the current simulation design, each simulated ERP image contains all previously mentioned components, so a single image can contain traces of several other classes. Origin for that desicion is that the visual patterns are sensitive to small parameter changes. A slight change in timing, amplitude, or noise can move, deform, or remove a pattern, or produce a shape that no longer matches its usual visual description. Therefore, the final experiments use sigmoid as the only positive ERP-image pattern. It was the most robust against parameter changes of the six simulated morphologies and also the most frequent pattern in the available labelled fixation data.

This design choice is important because the class is not encoded by a separate image generator for each pattern. Instead, the same simulated ERP activity can be made to reveal different structures through sorting. The no class label follows the same logic. It is not an empty or all-noise image. It is an ERP image created from the same simulated activity, but with a random trial order, so that the systematic relation between trials and time is removed.

The explored simulation setups then vary the onset, component, amplitude, and noise parameter distributions using broad randomisation, Latin hypercube calibration, Monte Carlo random search, and a two-zone mixture of realistic, edge, and stress configurations @Tobin2017, @McKay1979, @Bergstra2012, @Gretton2012.

UnfoldSim is well suited to this role because it simulates continuous event-based time series for EEG and event related signals, rather than only isolated averaged waveforms @Schepers2025.

The central challenge is the sim-to-real gap. A strong synthetic performance does not guarantee real-data performance, because the model may learn properties that are specific to the simulator. Domain randomisation addresses this risk by varying simulator parameters so that the real world appears as one possible variant within the window of synthetic distribution @Tobin2017. In this thesis, the same principle applies. If parameters of component widths, latency gaps, amplitudes, basis shapes, or noise levels are varied too aggressively, the target pattern itself becomes implausible or disappears.

Increase variation and remain close enough to the real sigmoid morphology is the goal. The exposed search space contains 24 parameter specifications. Because each specification has a mean and a standard deviation, the calibration problem becomes 48-dimensional. Three sampling strategies were explored. 

Broad random search, or domain randomisation, samples the parameter space as a random exploration baseline, with each parameter given the same weight. 

Latin hypercube sampling divides continuous parameter ranges into discrete intervals and combines them through permutations, spreading candidates more evenly across dimensions than ordinary random sampling @McKay1979. 

Monte Carlo random search provides a simple comparison by uniformly sampling one parameter at a time, and remains defensible because random search is a strong baseline for high-dimensional hyperparameter spaces @Bergstra2012. 

The two-zone mixture uses 70% of the best parameters from the LHS baseline and 30% edge-case configurations.

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

Recent ERP visualisation survey work confirms that this inspection step is still
mainly treated as a plotting and software-practice problem. Mikheev et al. study
plot types, tools, naming, sorting, colour maps, and uncertainty displays, rather
than automated screening algorithms @Mikheev2024ArtOfBrainwaves. The present
thesis therefore sits between ERP visualisation practice and machine-learning
screening: it does not propose another ERP plot type, but asks whether an
existing plot type can become a reproducible classification input.

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
images and was later expanded to 400 additional unique images @LabelStudio. The
web interface made the annotation workflow fast enough to enlarge the labelled
set within the project scope. The larger set is useful because it exposes the
classifier to more pattern variants and noise structures, but it does not remove
the need for agreement checks.

Figure @fig:pattern-decision-tree shows a first draft of the manual
pattern-labelling tree. It first asks whether a visible pattern is present and
then separates local patterns from patterns that extend across time.

#pattern-decision-tree <fig:pattern-decision-tree>

This is a working aid for annotation and not yet a final rule set.

== Data Simulation and Preprocessing
Several preprocessing choices were explored, but these early runs were not
evaluated with the later reporting setup and are therefore not treated as final
model results. The exploratory comparisons include nearest-neighbour, linear,
quadratic, cubic, and Lanczos resizing; pipelines with and without Gaussian
smoothing; z-scoring before versus after resizing; value binning in the ERP
matrix; and input resolutions from `16x16` to `256x256`.

The real-data path currently sorts trials, applies per-timepoint z-scoring,
performs Gaussian low-pass filtering, and resizes the image to the model input
size.

The pipeline is therefore treated as a structured configuration problem, not as
a fixed background step. This follows the general lesson from nnU-Net: some
choices can be fixed for comparability, some can be derived from dataset
properties, and others need empirical validation @Isensee2021nnUNet. In this
thesis, resolution, smoothing, z-scoring order, channel construction, and class
balancing belong to that empirical part of the design space.

Additional explored variants include data augmentation, morphological
operations, edge detectors, denoising, contrast normalisation, gradient-based
channels, and anti-aliased resizing. These experiments were useful for
narrowing the design space, but they did not produce a retained improvement in
classification performance at that stage.
RESULT TABEL

== Calibration, Models, and Training
The classifier comparison uses binary CNN baselines with one, three, and ten
convolutional layers, together with a pretrained `ResNet18`.

Because labelled ERP images are scarce, augmentation is evaluated as part of the
training design rather than as a secondary data-cleaning step. Brain-RetinaNet is
a domain-distant but useful reference point: in a small labelled MRI detection
dataset, Iqbal et al. use targeted augmentation to address limited sample
availability and report improvements across several detector backbones
@Iqbal2026BrainRetinaNet. The analogy is limited to the data regime, but it
supports treating class-aware ERP-image augmentation and class balancing as
central training decisions.

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
measurements and conclusions @Clayson2021ERPMultiverse. Biomedical image
segmentation shows the same dependency from another angle: nnU-Net obtains
strong results with U-Net-like templates by systematically configuring the
whole preprocessing, training, and post-processing pipeline
@Isensee2021nnUNet. In the experiments, resolution, filtering, and channel
variants also changed performance. The reported metrics therefore describe one
concrete representation pipeline, not an intrinsic property of ERP images in
general.

Visualisation conventions add another layer of conditionality. Mikheev et al.
show that ERP researchers do not use fully consistent names for plot types and
that practices around sorting, polarity, colour maps, and uncertainty displays
vary across users and tools @Mikheev2024ArtOfBrainwaves. A classifier trained
on rendered ERP images therefore inherits assumptions from the rendering
toolchain as well as from the EEG preprocessing pipeline.

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
One useful next step is a non-CNN baseline for the same ordered-matrix task.
Magnostics shows that hand-engineered image descriptors can rank matrix
visualisations by visible motifs @Behrisch2017Magnostics. A small descriptor
library for ERP images would test whether sigmoid detection really needs learned
filters or whether fixed measures of curvature, continuity, gradient
concentration, and row-order structure already capture much of the task.

A second extension is localisation. The current classifier assigns one label to
an entire ERP image, so it cannot mark where a pattern starts, ends, or overlaps
with another structure. Detection-style biomedical imaging work such as
Brain-RetinaNet shows how convolutional models can move from image-level
classification toward localising relevant regions @Iqbal2026BrainRetinaNet. For
ERP images, such a shift would require labels for pattern extents in trial-time
space, not only image-level labels.

// ----------------------------------------------------------------------------
// Chapter 6 - End with direct answers, not a second discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Conclusion <chp:conclusion>

// Summarize the core takeaway in a few sentences and answer the research
// question directly.
