// A central place where libraries are imported (or macros are defined)
// which are used within all the chapters:
#import "utils/global.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes


// Fill me with the Abstract
#let abstract = [#lorem(150)]

// Fill me with acknowledgments
#let acknowledgements = none

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

  == Use of AI Tools

  I used AI-based tools as support tools during this thesis: OpenAI ChatGPT, Codex, and Anthropic Claude Code @OpenAICodex2026, @AnthropicClaudeIntro2026. These tools supported code development, debugging, and language revision. I did not use AI tools as independent scientific sources. I reviewed all AI-assisted input and output and remain responsible for the submitted thesis.

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
    key: "cpu",
    short: "CPU",
    long: "Central processing unit",
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
    key: "gpu",
    short: "GPU",
    long: "Graphics processing unit",
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
            node((-2.8, 3.6), align(center)[No-class], width: 20mm, corner-radius: 1.5pt),

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
            node((3.30, 10.6), align(center)[No-class], width: 20mm, corner-radius: 1.5pt),

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
            node((-0.45, 8.85), align(center)[No-class], width: 18mm, corner-radius: 1.5pt),
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
    Manual decision tree for ERP image patterns. It first separates images without a connected pattern from candidate patterns, then distinguishes time-drifting structures from vertical or widening structures before assigning one of the six pattern labels.
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

== Research Questions, Hypotheses, and Contributions

This thesis investigates whether visual ERP-image patterns can be detected automatically with convolutional neural networks, and whether simulated ERP images can reduce the need for manually labelled real data. The work is guided by the following research questions and hypotheses.


#[
  #set enum(
    numbering: (..nums) => strong[RQ#(nums.pos().map(str).join(".")):],
    full: true,
    tight: false,
    indent: 0pt,
    spacing: .7em,
    body-indent: .325em,
  )
  #set list(
    tight: false,
    indent: 0pt,
    spacing: .45em,
    body-indent: .325em,
  )

  + *Sim-to-real transfer* <rq:sim-to-real> How accurately and efficiently can a CNN model trained on simulated ERP images detect and distinguish relevant spatio-temporal patterns in real-world ERP images?

    + *Sim-to-real sigmoid* <rq:sim-to-real-sigmoid> To what extent can a CNN trained on simulated sigmoid ERP images recognise sigmoid patterns in manually labelled real ERP images?

      - *H1.1a: Real-data recognition* <hyp:sim-to-real-sigmoid-performance> A CNN trained on simulated sigmoid and no-class ERP images will classify manually labelled real sigmoid and no-class ERP images above chance and majority-class baseline performance.

      - *H1.1b: Sim-to-real drop* <hyp:sim-to-real-sigmoid-gap> The same CNN will perform better on held-out simulated ERP images than on manually labelled real ERP images, indicating a measurable sim-to-real gap.

    + *Simulator calibration* <rq:simulator-calibration> How does simulator calibration affect sim-to-real transfer and what performance gap remains compared with training on real labelled ERP images?

      - *H1.2a: Targeted calibration* <hyp:simulator-calibration-targeted> Simulator calibration with Latin hypercube or Monte Carlo sampling will improve sim-to-real transfer compared with broad random parameter sampling on the same real validation source.

      - *H1.2b: Source-specific gain* <hyp:simulator-calibration-source> A simulator calibrated against one real source will transfer better to that source than to other real ERP-image sources containing the same pattern, so the calibration advantage will shrink with distance from the calibration source.

  #v(.7em)

  + *Manual labelling* <rq:manual-labelling> How accurately and efficiently can CNNs detect visual ERP-image patterns when they are trained on manually labelled real ERP images?

    + *Preprocessing* <rq:real-preprocessing> Which training and preprocessing choices improve CNN-based classification of manually labelled ERP images?

      - *H2.1: Preprocessing* <hyp:real-preprocessing> ERP-image preprocessing choices, especially sorting, per-time-point z-scoring, smoothing, and resizing, will affect classification performance.

    + *Augmentation and class imbalance* <rq:real-augmentation> To what extent do ERP-specific augmentations or imbalance-handling strategies improve CNN model performance?

      - *H2.2: Augmentation and class imbalance* <hyp:real-augmentation> ERP-specific augmentation and imbalance-handling strategies will improve balanced accuracy or macro-F1, especially for underrepresented classes.

    + *Model choice and training strategy* <rq:real-model-choice> Which model architecture and training regime provide the best accuracy-efficiency trade-off for manually labelled real ERP-image classification?

      - *H2.3: Model choice and training strategy* <hyp:real-model-choice> Model architecture and training regime will produce measurable accuracy-efficiency trade-offs, so the preferred configuration will depend on both classification performance and computational cost.
]

The main contributions of this thesis are:

1. A manually labelled real ERP-image dataset from multiple EEG/ERP sources, covering several visual patterns.
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
The six pattern names used in this thesis form a practical vocabulary for visible structure in ERP images. They are not six separate brain components. Sorting trials in the ERP-image matrix can make different mechanisms visible. A component may shift in time, spread out with variable duration, change polarity, or vary non-linearly @Jung2001, @Delorme2004EEGLAB, @Delorme2015GrandERPImage. The six pattern labels were chosen because they cover these main cases while staying distinct enough for manual annotation. A seventh no-class option was added for images that do not clearly match any of the six patterns, so that the annotation scheme can also represent the absence of a target morphology.

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
sigmoid versus no-class. This restriction is a feasibility choice. The labelled
fixation dataset was the available real-data target, and sigmoid was both
frequent in that dataset and stable to create in the simulator.

The simulation settings therefore use this fixation dataset as the first target
distribution. That includes a sampling rate fixed at 512 Hz, a trial count of
2508, and an epoch length of one second; the simulated matrices are then passed
through the same image preprocessing to produce 64x64 single-channel images in
matrix form. The goal is to create ERP images that resemble real fixation ERP
images in their basic dimensions and preprocessing output, rather than arbitrary
simulator examples.

This setup leads directly to #link(<rq:sim-to-real-sigmoid>)[RQ 1.1]. The aim of simulation is to approximate real ERP images well enough that a CNN trained on simulated images can recognise the same visual pattern in real data. In the long term, such a model should not be limited to one dataset. As a first feasibility test, this thesis therefore compares the simulated sigmoid/no-class task with one manually labelled real fixation dataset.

=== Simulated Class Definition
A six-class simulation setup was considered at first, but handling all visual patterns was infeasible. In the current simulation design, each simulated ERP image contains all previously mentioned components, so a single image can contain traces of several other classes. The origin of that decision is that the visual patterns are sensitive to small parameter changes. A slight change in timing, amplitude, or noise can move, deform, or remove a pattern, or produce a shape that no longer matches its usual visual description. Therefore, the final experiments use sigmoid as the only positive ERP-image pattern. It was the most robust against parameter changes of the six simulated morphologies and also the most frequent pattern in the available labelled fixation data.

This design choice is important because the class is not encoded by a separate image generator for each pattern. Instead, the same simulated ERP activity can be made to reveal different structures through sorting. The no-class label follows the same logic. It is not an empty or all-noise image. It is an ERP image created from the same simulated activity, but with a random trial order, so that the systematic relation between trials and time is removed.

=== Parameter Search
The parameter search aims to find simulator settings that make a CNN trained on synthetic images perform as well as possible on the labelled fixation dataset. For this purpose, this thesis applies and compares broad randomisation, Latin hypercube sampling, heterogeneity scaling, Monte Carlo random search, and a two-zone mixture strategy @Tobin2017, @McKay1979, @Bergstra2012. The next section describes these search and scoring strategies in more detail.

UnfoldSim is well suited for this purpose because it simulates continuous event-based time series for EEG and event-related signals, rather than only isolated averaged waveforms @Schepers2025.

=== Sim-to-Real Gap
The central challenge is the sim-to-real gap. Strong synthetic performance does not guarantee real-data performance, because the model may learn properties that are specific to the simulator. Domain randomisation addresses this risk by varying simulator parameters so that the real world appears as one possible variant within the window of the synthetic distribution @Tobin2017. In this thesis, the same principle applies. If parameters such as component widths, latency gaps, amplitudes, basis shapes, or noise levels are varied too aggressively, the target pattern itself becomes implausible or disappears.

=== Search and Scoring Strategies
This parameter search addresses #link(<rq:simulator-calibration>)[RQ 1.2]. It tests whether better simulator calibration can improve sim-to-real transfer, and whether a remaining gap to real-label training persists.

The goal is therefore to increase variation while remaining close enough to the real sigmoid morphology. The exposed search space contains 24 parameter specifications. Because each specification has a mean and a standard deviation, the calibration problem becomes 48-dimensional. Five search strategies and one simulator-first scoring proxy were explored.

Broad random search, or domain randomisation, samples the parameter space as a random exploration baseline, with each parameter given the same weight.

Latin hypercube sampling (LHS) divides continuous parameter ranges into discrete intervals and combines them through permutations, spreading candidates more evenly across dimensions than ordinary random sampling @McKay1979.

As a naive parameter-search attempt, a heterogeneity experiment scaled the standard deviations of normally distributed parameters to 50%, 100%, and 1000% of the corresponding mean magnitudes.

Monte Carlo random search provides a simple comparison by uniformly sampling one parameter at a time, and remains defensible because random search is a strong baseline for high-dimensional hyperparameter spaces @Bergstra2012.

The two-zone mixture uses 70% of the best parameters from the LHS baseline and 30% edge-case configurations.

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


Manual labelling is the limiting step, so the annotation workflow avoids labelling every possible combination of dataset, participant, event, channel, and sort variable. It works from a coarse screening step towards more detailed labelling and keeps only sources that satisfy the preprocessing requirements mentioned earlier.

The workflow starts by letting the classifier propose ten candidate images for each dataset. These images receive manual labels to test whether the generated ERP images satisfy the preprocessing and quality requirements for this thesis. For sources that pass this check, the workflow ranks the available images by the model probability of containing a pattern and selects 200 images across the available events and sort variables from the top of this ranking. These images also receive manual labels.

The third step expands only the promising sort variables. If the 200-image screen finds at least one pattern for a dataset and sort variable, the workflow labels all available channels for that combination. The reference fixation dataset and some other sources contain only one participant, while other sources contain several participants. When several participants are available, the workflow still uses one participant per dataset so that the labelled pool covers more different sources and experiments. This also increases the chance of finding different pattern types, instead of adding many potential similar images from the same experiment.

#pattern-decision-tree <fig:pattern-decision-tree>

== Data Simulation and Preprocessing
Before training a classifier, each EEG/ERP recording must be converted into a fixed-size ERP image. This conversion shapes the actual input to the CNN models requirements. The classifier only receives the final image matrix, so resizing, smoothing, scaling, and filtering can preserve some visual structures, weaken others, or introduce artefacts that the model may learn. This thesis therefore evaluates these preprocessing choices empirically instead of treating them as neutral background steps.

The preprocessing comparison covers several parts of the image pipeline. It compares resizing with nearest-neighbour, linear, quadratic, cubic, and Lanczos interpolation. It also compares pipelines with and without Gaussian smoothing, z-scoring before versus after resizing, value binning in the ERP matrix, input resolutions from 16x16 to 256x256, and additional visual noise-reduction filters.

Trial sorting and z-scoring remain fixed because they define a usable ERP-image input for this task. Sorting does not add or alter information but only orders trials so that potential cross-trial pattern structure can become visible. Per-time-point z-scoring reduces vertical amplitude-dominated bands, which often occur when many trials show a strong response at the same time, mostly early after stimulus onset. These bands are not the main morphology of interest in this thesis. Z-scoring therefore emphasises relative differences between trials. The order of these steps is fixed to sorting, then z-scoring, then optional smoothing, then resize, because z-scoring directly after sorting removes the common vertical bands before smoothing could spread them across neighbouring time points. The same order is applied to simulated and real ERP images for comparability.

@fig:basic-preprocessing-pipeline shows the four steps of this pipeline applied to one real ERP image from the fixations dataset, channel ch096, sorted by duration. The goal of the visualisation is to make the cumulative effect of sorting, z-scoring, and Gaussian smoothing visible, so that the pattern present in the raw recording becomes easier to read step by step.

#let pipeline-panel(num, path) = align(center)[
  #image(path, width: 100%)
  #v(-0.5em)
  #text(weight: "bold", size: 9pt)[(#num)]
]

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 3mm,
    row-gutter: 4mm,
    pipeline-panel(1, "figures/pipeline_visualisations/figure_1a_raw_erp_image.svg"),
    pipeline-panel(2, "figures/pipeline_visualisations/figure_1b_sorted_by_duration.svg"),
    pipeline-panel(3, "figures/pipeline_visualisations/figure_1c_z_scored_per_time_point.svg"),
    pipeline-panel(4, "figures/pipeline_visualisations/figure_1d_gaussian_smoothed.svg"),
  ),
  caption: [Basic preprocessing pipeline applied to one real ERP image from the fixations dataset, channel ch096, sorted by duration. (1) Raw trial-by-time matrix. (2) Trials sorted in ascending duration order. (3) Per-time-point z-scoring. (4) Additional Gaussian low-pass smoothing.],
) <fig:basic-preprocessing-pipeline>

A second preprocessing variant is also used in one ablation experiment, where z-scoring is applied after downscaling and the smoothing kernel is altered. This variant is described together with its experiment in @sec:experiments-setups.

This thesis applies the same empirical selection logic to the broader training pipeline. Some steps stay fixed for comparability, some depend on dataset properties, and others require experimental comparison. This follows the general lesson from nnU-Net @Isensee2021nnUNet. In this thesis, the compared training choices include the image pipeline, class balancing, augmentation, model training, filtering, denoising, contrast adjustment, and resizing. These comparisons help to choose preprocessing and training settings that improve model performance on normalised input data without artificially inflating scores through overfitting.

== Real-to-Real Data Augmentation
Trial slicing is a data-augmentation strategy that turns one large ERP image into several smaller ones. This expands the available labelled training pool without recording new ERP data, additional labeling or changing the underlying signal. When the trial count of the parent image does not divide evenly into the chosen slice size, the leftover trials are filled up with randomly selected trials from the same recording, so every resulting slice has the same shape.

Trial slicing operates on an already sorted ERP image of shape $(n, T)$, where $n$ is the number of trials and $T$ is the number of time samples. It cuts the image horizontally along the trial axis into smaller ERP images of fixed target trial count $t$. Let

$ K = floor(n / t) $

denote the number of full slices that fit into the parent image. The trials are visited in their existing sort order and distributed across these $K$ bins in round-robin fashion, so bin $k$ for $k = 1, dots, K$ contains the trials at sorted positions

$ k, quad k + K, quad k + 2 K, quad dots, quad k + (t - 1) K. $

Any leftover trials at the end of the round-robin pass form one additional remainder bin, which is filled up to the same target trial count $t$ with randomly reused trials from the full set. Each bin becomes its own ERP image of shape $(t, T)$ that keeps the parent sort order but contains only $t$ of the original trials. From one parent ERP image with $n$ trials this produces

$ K + 1 = floor(n / t) + 1 $

sliced ERP images, where the first $K$ slices are mutually disjoint and the remainder slice share trials with the others through the fill-up step.

Trial slicing combines several practical advantages in one easy-to-reproduce procedure. The round-robin distribution takes trials evenly from across the parent ERP image, so each slice preserves the global pattern shape rather than risking capturing local clusters of neighbouring trials. A purely random selection would carry a clumping risk where some slices over-represent one part of the sorted trial axis and could visually distort the pattern. The procedure is almost entirely deterministic, with the remainder fill-up as the only random step. It is also cheap to compute, since assigning trials to slices is essentially a modulo operation on the sorted index. The fixed slice size matters because the labelled data sources differ widely in trial count per recording, while the CNN expects a uniform input shape, and because the goal is generalisation across data sources. Equalising the trial dimension keeps the model from learning trial count itself as a shortcut feature. Finally, different combinations of trials lead to different background-noise textures across slices, which adds useful variation for the classifier, while two slices that contained the same trials would produce the same noise.

Each sliced ERP image is then expanded into four augmented variants before the remaining preprocessing of z-scoring, Gaussian smoothing, and resizing is applied. The first variant is the slice as it is, with its original trial order and signal polarity. The second variant reverses the trial order, so the slice is flipped along the sort axis. The third variant inverts the signal polarity by multiplying every value in the trial-by-time matrix by minus one. The fourth variant combines both operations, with reversed trial order and inverted polarity at the same time. All four variants carry the same class label as their parent slice, because the visual pattern that defines the class is preserved.

This expansion makes the classifier more robust in two ways. The reversed trial order forces the model to recognise the pattern from either reading direction along the sort axis. The polarity inversion removes the dependency on the absolute sign of the EEG signal, which can differ between recording references and between datasets. As a side effect, quadrupling the labelled examples per slice acts as a regulariser and supports generalisation across the heterogeneous data sources used in this thesis.

@fig:trial-slicing-augmentation visualises the combined slicing and augmentation step on the same fixations recording used in @fig:basic-preprocessing-pipeline. The parent image enters this step as a sorted ERP image only, without z-scoring or Gaussian smoothing. Those two steps are applied separately to every augmented child image, so each child carries its own per-time-point z-scoring and its own smoothing pass. The visualisation slices the parent into four equal parts, which is a demonstration setting that makes the four-slice grid easy to inspect. The training pipeline uses a smaller target trial count of 200, chosen as the largest value that still fits into every labelled data source, since the source with the fewest valid trials provides the smallest common denominator.


TODO description of erp images, explain colorbar and make a unified colobar for augmentation

#figure(
  image("figures/pipeline_visualisations/figure_2_trial_slicing_augmentation.svg", width: 100%),
  caption: [Trial slicing and augmentation on one real ERP image from the fixations dataset, channel ch096, sorted by duration. Rows correspond to the four slices, columns to the four augmentation variants reference, inverse sort, inverse polarity, and inverse sort combined with inverse polarity. Each panel shows the slice after its own per-time-point z-scoring and Gaussian smoothing, so that the pattern remains visible while the augmented variants differ in orientation and polarity. The slice count of four is chosen for visualisation only, the training pipeline uses a target trial count of 200 instead.],
) <fig:trial-slicing-augmentation>

The visualisation shows that the underlying pattern is preserved across all augmented children. The four slices each contain a disjoint subset of trials of the parent recording, and the four variants per slice differ only by sort direction and signal polarity. The augmented set therefore covers a wider range of viewing conditions while keeping the class label of the parent recording.

The cross-validation folds are then defined deterministically on top of the parent recordings, not on the slices. All slices that come from the same channel-source pair always end up in the same fold across runs, so that no underlying trial leaks between the training and the validation fold of a cross-validation run. The deterministic assignment also keeps the per-fold class balance close to the overall pool balance, so that imbalance handling does not interact with fold composition.

The augmentation experiments in this thesis address #link(<rq:real-preprocessing>)[RQ 2.1], #link(<rq:real-augmentation>)[RQ 2.2], and #link(<rq:real-model-choice>)[RQ 2.3]. They cover trial-dropout views, inverse-sort and polarity views, frequency-domain perturbations, and class-aware sampling. Strategies that did not transfer well are documented together with the experiment that tested them.

== Calibration, Models, and Training
The classifier comparison uses binary CNN baselines with one, three, and ten convolutional layers, together with a pretrained ResNet18 model @He2016ResNet. ResNet is useful here because residual connections make a deeper image classifier easier to optimise, while still providing a standard computer-vision baseline for comparison. Having the best practices already implemented, rather than figuring them out from scratch, makes it easier to focus on the classification and preprocessing of the ERP images.

Data augmentation improves model robustness when labelled training data are limited. This thesis uses augmentation to increase the variation of the labelled ERP-image training set and to reduce memorisation of individual examples. Brain-RetinaNet provides a domain-distant but useful reference point for this small-data setting. Iqbal et al. apply targeted augmentation to a small labelled MRI detection dataset and report improvements across several detector backbones @Iqbal2026BrainRetinaNet. The analogy only concerns the limited-data setting, but it motivates the evaluation of class-aware ERP-image augmentation and class balancing as central training decisions.

TODO explain modsplit, its necessary different numberf of trials, input nomalsiation

== Experimental Setups <sec:experiments-setups>
The empirical work of this thesis consists of a numbered series of experiments. The Methods chapter defines each setup once. The Results chapter then references the experiments by number and reports only the observations.

=== Experiment E1 16x16 dense-network parameter search <exp:dense16>
E1 compares three sampling strategies, broad random, Latin hypercube, and Monte Carlo, against the starting parameters at 16x16 input resolution. A parameter combination is one complete assignment of all simulator settings at once. Every strategy proposes twelve such full combinations, where each combination differs from the starting parameters in many values at once. For each combination the simulator generates 1,000 images per pattern. A dense neural network is then trained on those images for a fixed budget of 30 epochs without early stopping and validated on the real fixation labels. Each combination is evaluated three times to average out random effects from the simulator draws and from the network initialisation, so that a single lucky or unlucky run does not decide the reported score. The reported BAcc and macro-F1 are the means across these three repeats. The dense network is the chosen downstream classifier because at 16x16 the input is small enough that convolutional layers are not needed. A deeper network such as ResNet18 would downsample a 16x16 input below useful spatial sizes after only a few of its convolution stages. The same setup is also run at smaller image sizes of 8x8, 4x4, and 2x2 to locate the lower bound below which further reduction removes too much of the visual pattern itself.

=== Experiment E2 Sim-to-sim training-set fit <exp:simfit>
E2 reuses the best Latin-hypercube candidate from E1 at 16x16 with low-pass smoothing and additionally evaluates the trained dense network on its own simulated training set. Three evaluation repetitions are used so the per-repeat standard deviation of BAcc on the real labelled data can be reported. The purpose of the experiment is to measure the gap between synthetic validation and real-data evaluation for the same calibrated configuration.

A separate preprocessing-pipeline ablation is conducted alongside E2 to probe how robust the calibration is to small changes in the image pipeline. In that ablation z-scoring is applied after downscaling instead of before, and the smoothing kernel is altered. Every search strategy is run again under this altered pipeline.

=== Experiment E3 Simulator-first downstream model comparison <exp:simrank>
E3 keeps the simulator fixed and grid-searches the simulator-trained classifier across three downstream models, namely a dense neural network, a random forest, and a support-vector machine with radial basis kernel, three sampling strategies (broad random, Latin hypercube, Monte Carlo), and four input resolutions (2x2, 4x4, 8x8, 16x16). The reported ranking compares the top three downstream models at 16x16 under Monte Carlo random search. The downscale serves two ends in this experiment. It shortens training time and it locates the lower resolution below which further reduction removes too much of the visual pattern itself, which the analysis aims to preserve rather than aggregate away as other methods do.

=== Experiment E4 64x64 ResNet18 parameter search <exp:resnet64>
E4 repeats the simulator parameter search with a pretrained ResNet18 classifier at 64x64 input resolution to test whether the same simulator can train the higher-capacity classifier used in the real-data experiments. Every parameter combination has all simulator parameters varied at once. The simulator generates 1,000 sigmoid and 1,000 no-class images at 64x64 resolution with low-pass smoothing applied, and the trained model is then validated on the real fixation labels. ResNet18 is pretrained, trained for a fixed budget of 8 epochs without early stopping, with batch size 64, learning rate 3e-4, and a small label smoothing of 0.02. The search budget per strategy is 12 combinations for broad random, 48 for Latin hypercube, and 48 for Monte Carlo. Each combination is evaluated three times and the starting-parameter baseline is evaluated once. This adds up to 325 ResNet18 training and evaluation runs in total. Runtime is logged per repeat for three stages, namely image simulation including the full ERP-image preprocessing pipeline (sorting trials, z-score across time points, low-pass smoothing, and resize), ResNet18 training over 8 epochs, and inference on the real fixation labels.

A separate profiling run of the same simulation pipeline is used to attribute the simulation cost between the raw ERP simulation and the subsequent image-processing steps.

=== Experiment E5 Shallow-CNN versus ResNet baseline <exp:depth>
E5 compares three shallower CNN baselines with 1, 3, and 10 convolutional layers against the pretrained ResNet18 under the same grouped five-fold cross-validation on the labelled ERP-image pool. The motivation is whether smaller models could save training time without losing accuracy. ResNet34 is included as an additional comparison point at the deeper end of the architecture range.

== Evaluation Protocol
The evaluation combines accuracy, balanced accuracy, macro F1, precision, recall, and timing summaries under grouped five-fold cross-validation. These metrics are used together because a single score would not capture overall performance, class imbalance, different error types, and computational cost.

All simulations, model training, and evaluation runs in this thesis were executed on a single PC with an AMD Ryzen 7 7800X3D 8-core CPU, 64 GB of system memory, and an NVIDIA GeForce RTX 4070 GPU with 12 GB of VRAM running Linux.




// ----------------------------------------------------------------------------
// Chapter 4 - Present observations in the same order as the pipeline.
// Keep this chapter descriptive; save interpretation for the discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Results <chp:results>

== Data Simulation and Calibration Results
This section reports the four simulation experiments E1–E4 defined in @sec:experiments-setups, plus a general observation that applies across them. Each experiment is referenced by its identifier so that the setup details are not repeated here.

=== General observations from the simulation experiments
The best classifier setting across simulation experiments uses sorting, per-time-point z-scoring, Gaussian low-pass smoothing, and a final resize. A perfect or near-perfect fit on the simulated training set was observed frequently across the evaluated setups and was therefore not pursued further as a separate result. None of the models trained at the smaller image sizes of 8x8, 4x4, and 2x2 reached the performance of the best 16x16 run.

=== Results of E1 (16x16 dense-network parameter search)
@tab:simulation-parameter-search-results presents the best run per search method and resolution. The Latin hypercube and Monte Carlo strategies both reach a top BAcc of 0.711 at 16x16, ahead of broad random search at 0.692 and the starting parameters at 0.674. At the smaller resolutions performance drops monotonically with size.

// Sources:
// - notebooks/week_15/dense_nn_lowpass_direct_search.ipynb
// - notebooks/week_15/simulation_small_sclae.ipynb
// - /home/benjamin/Dokumente/presentations/pdfs/midterm_talk_v3-1776681517355.pdf, pages 15, 22
#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (2.4fr, 0.7fr, 0.85fr, 0.95fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x > 0 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Search method],
        [Resolution],
        [BAcc],
        [Macro-F1],
      ),

      [Starting parameters],
      [16x16],
      [0.674],
      [0.601],

      [Broad random search],
      [16x16],
      [0.692],
      [0.657],

      [Monte Carlo random search],
      [16x16],
      [0.711],
      [0.636],

      [Latin hypercube search],
      [16x16],
      [0.711],
      [0.659],

      [Latin hypercube search],
      [8x8],
      [0.665],
      [0.517],

      [Latin hypercube search],
      [4x4],
      [0.638],
      [0.425],

      [Broad random search],
      [2x2],
      [0.612],
      [0.407],
    ),
    caption: [Best Dense-NN run per search method and resolution from 72 screening combinations.],
  ) <tab:simulation-parameter-search-results>
]

With smoothing deactivated for both the simulated and the real ERP images, the best 16x16 dense-network row drops from BAcc 0.711 to 0.42. Two further search strategies described in the methods chapter do not contribute a separate row to @tab:simulation-parameter-search-results. As part of the heterogeneity-scaling family, a simpler variant that only increased the standard deviations of the normally distributed parameters was discarded after visual inspection. The resulting ERP images and their visible patterns differed strongly from the real labelled images, so this variant was not investigated further.

The two-zone mixture was tested in an early prototype together with all of the other parameter-finding strategies under the altered preprocessing pipeline described as part of E2 in @exp:simfit. Every evaluated strategy, including the two-zone mixture itself, collapsed to a balanced accuracy of 0.5 or lower under that altered pipeline. The failure is therefore not specific to the two-zone mixture and confirms empirically that minor pipeline adjustments can produce large changes in classification outcomes.

=== Results of E2 (sim-to-sim training-set fit)
@tab:simulation-calibration-results shows the same Latin-hypercube-calibrated dense network from E1 evaluated on its own simulated training set and on the real labelled fixation data. The simulated training set is fit perfectly with BAcc and macro-F1 at 1.000, while the real-data BAcc is 0.711 with a per-repeat standard deviation of 0.050. The perfect score on the training set only confirms that the network overfits the data it was trained on. A perfect training score combined with a much lower score on real labels is consistent with the dense network overfitting to features of the simulated images that do not transfer to real ERP images.

#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (2.8fr, 0.9fr, 1fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x > 0 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Evaluation set],
        [BAcc],
        [Macro-F1],
      ),

      [Simulated training set],
      [1.000],
      [1.000],

      [Real labelled fixation data],
      [0.711],
      [0.659],
    ),
    caption: [Training-set fit versus real-data score for the calibrated Latin hypercube candidate from E1.],
  ) <tab:simulation-calibration-results>
]

=== Results of E3 (simulator-first downstream model comparison)
@tab:simulation-model-comparison-results shows the three top rows of the ranking from E3, all at 16x16 under Monte Carlo random search. The dense neural network is the strongest of the three downstream models, ahead of the random forest and the support-vector machine with radial basis kernel.

#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (3fr, 0.85fr, 0.95fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x > 0 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Downstream model],
        [BAcc],
        [Macro-F1],
      ),

      [Dense neural network],
      [0.684],
      [0.542],

      [Random forest],
      [0.642],
      [0.502],

      [Support-vector machine with radial basis kernel],
      [0.641],
      [0.500],
    ),
    caption: [Top three downstream models from the simulator-first ranking of E3.],
  ) <tab:simulation-model-comparison-results>
]

=== Results of E4 (64x64 ResNet18 parameter search)
@tab:simulation-resnet18-64-results shows the starting parameters and the best candidate from each search method. Broad random search reaches the highest BAcc of 0.926 at candidate 10 of 12, followed by Latin hypercube at 0.916 at candidate 21 of 48, and Monte Carlo at 0.904 at candidate 38 of 48. The starting-parameter row is at 0.886. All three strategies clearly improve over the starting parameters at this resolution.

// Sources:
// - notebooks/data_generation/outputs/strategy_64x64_resnet18/posthoc_exports/baseline_summary.csv
// - notebooks/data_generation/outputs/strategy_64x64_resnet18/posthoc_exports/method_summary.csv
// - notebooks/data_generation/outputs/strategy_64x64_resnet18/posthoc_exports/top_per_strategy.csv
#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (2.2fr, 0.85fr, 0.85fr, 1.15fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x > 0 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Search method],
        [BAcc],
        [Macro-F1],
        [Best at iteration],
      ),

      [Starting parameters],
      [0.886],
      [0.891],
      [—],

      [Broad random search],
      [0.926],
      [0.797],
      [10 of 12],

      [Latin hypercube search],
      [0.916],
      [0.782],
      [21 of 48],

      [Monte Carlo random search],
      [0.904],
      [0.852],
      [38 of 48],
    ),
    caption: [Results of E4. The Best at iteration column gives the candidate index at which the best mean BAcc occurred within the fixed search budget of the strategy.],
  ) <tab:simulation-resnet18-64-results>
]

The Best at iteration column shows that no later candidate of the same strategy reached a higher BAcc. For Latin hypercube the remaining 27 of 48 candidates therefore yield no further improvement. For broad random with 10 of 12 and Monte Carlo with 38 of 48 the best result lies close to the end of the budget, which suggests that more iterations could still have improved the result.

@tab:simulation-resnet18-64-timing reports the wall-time of E4 split into the three logged stages. Almost the entire runtime is spent generating the synthetic ERP images, while ResNet18 training and inference together add up to less than 15 minutes across all 325 repeats.

// Sources:
// - notebooks/data_generation/outputs/strategy_64x64_resnet18/posthoc_exports/all_repeats_raw.csv
#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (2.6fr, 1.05fr, 1.15fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x > 0 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Stage],
        [Mean per repeat],
        [Sum over 325 repeats],
      ),

      [Image simulation and preprocessing],
      [311.7 s],
      [28.1 h],

      [ResNet18 training (8 epochs)],
      [2.50 s],
      [13.5 min],

      [Inference on real fixation labels],
      [0.020 s],
      [6.4 s],
    ),
    caption: [Per-stage runtime of E4, aggregated over all 325 evaluated repeats.],
  ) <tab:simulation-resnet18-64-timing>
]

The separate profiling run mentioned for E4 in @sec:experiments-setups shows that about 84 percent of the per-image cost comes from the raw ERP simulation itself, while only about 16 percent is spent on the subsequent image-processing steps.

== Classification Performance on Real Data
The supervised real-data experiments use the manually labelled ERP-image pool. The class distribution is imbalanced, so balanced accuracy and macro-F1 remain the main comparison metrics.

// Sources:
// - notebooks/week_21/outputs/week21_labeling_summary/summary.json
// - notebooks/week_23/outputs/augmentation_inverse_sort_polarity/run_config.json
#[
  #show figure: set block(breakable: true)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.7fr, 0.75fr, 3.4fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => if y == 0 { center + horizon } else if x == 1 { center + top } else { left + top },
      table.header(
        [Quantity],
        [Value],
        [Meaning],
      ),

      [Classified annotations],
      [2,879],
      [Manually classified ERP images],

      [Pattern labels],
      [294],
      [Positive binary class],

      [Datasets with patterns],
      [10],
      [Sources with at least one positive pattern label found],
    ),
    caption: [Additional metrics about the manual label pool.],
  ) <tab:manual-label-pool-results>
]


=== Results of E5 (shallow-CNN versus ResNet baseline)
The per-fold training times of the shallow CNN baselines stay within a few seconds of the pretrained ResNet18, a difference that is negligible for the project budget. The accuracy gap is much larger. The 1- and 3-layer CNNs often collapse to a single predicted class, the 10-layer CNN improves marginally, and the pretrained ResNet18 reaches the overall best performance on the same data. ResNet34 matches the ResNet18 accuracy with increased training time, so the deeper variant adds cost without accuracy and ResNet18 remains the chosen model.

// Sources:
// - notebooks/week_21/outputs/resnet18_labeled_erp_cv/metrics_summary.csv
// - notebooks/week_21/outputs/resnet34_labeled_erp_cv_128_resize/metrics_summary.csv
// - notebooks/week_23/outputs/augmentation_inverse_sort_polarity/metrics_summary.csv
// - notebooks/week_18/preprocessing_test.ipynb
// - notebooks/week_18/outputs/data_augmentation_tests_ranked_summary.csv
// - notebooks/week_18/outputs/semi_supervised_learing2_summary.csv
// - /home/benjamin/Dokumente/presentations/pdfs/week_22.pdf, page 5
// - /home/benjamin/Dokumente/presentations/pdfs/week_23.pdf, page 2
#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.35fr, 1.25fr, 1.35fr, 2.45fr, 0.65fr, 0.75fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x == 4 or x == 5 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Experiment],
        [Model],
        [Training data],
        [Changed variables],
        [BAcc],
        [Macro-F1],
      ),

      [Inverse sort and polarity augmentation],
      [ResNet18 pretrained],
      [12 labelled data sources with augmentation],
      [Trial sort direction, signal polarity, binary pattern task],
      [0.918],
      [0.917],

      [Preprocessing sweep],
      [ResNet18 pretrained],
      [fixations dataset only],
      [Gaussian smoothing, Dilation filter, high 100 smoothing],
      [0.907],
      [0.903],

      [Trial-dropout augmentation],
      [ResNet18 random initialisation],
      [fixations dataset only],
      [Trial dropout, threshold tuning, class-aware augmentation],
      [0.902],
      [0.880],

      [Labelled ResNet18 baseline],
      [ResNet18 pretrained],
      [12 labelled data sources],
      [No inverse sort or polarity augmentation, binary pattern task],
      [0.866],
      [0.866],

      [Labelled ResNet34 comparison],
      [ResNet34 pretrained],
      [12 labelled data sources],
      [Backbone depth, resize to 128x128, binary pattern task],
      [0.854],
      [0.854],

      [Pseudo-label semi-supervised training],
      [ResNet18 SSL fine-tune and student stage],
      [fixations dataset with same-dataset trial-sliced SSL pool],
      [SSL fine-tuning, confidence threshold 0.9 pseudo-labels],
      [0.837],
      [0.849],
    ),
    caption: [Real-data classification results by training data, model, and changed training variables.],
  ) <tab:real-data-classification-results>
]


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

TODO simulation speed up, currently its not feasable. better performance on multitheading

todo simulation find parameters for each patterns rather than overall. in this case sigmoid was tested. all patterns may take up hole week of runtime. maybe data set per input dimensions, as cnn model can detect very accurately if a high resolution image was scaled down or a low resolution image was sclaed down.

A second extension is localisation. The current classifier assigns one label to an entire ERP image, so it cannot mark where a pattern starts, ends, or overlaps with another structure. Detection-style biomedical imaging work such as Brain-RetinaNet shows how convolutional models can move from image-level classification towards localising relevant regions @Iqbal2026BrainRetinaNet. For ERP images, such a shift would require labels for pattern extents in trial-time space, not only image-level labels.

Augmentation is very important. One option is to simulate data for a specific real dataset.

// ----------------------------------------------------------------------------
// Chapter 6 - End with direct answers, not a second discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Conclusion <chp:conclusion>

// Summarize the core takeaway in a few sentences and answer the research
// question directly.
