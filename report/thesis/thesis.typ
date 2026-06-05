// A central place where libraries are imported (or macros are defined)
// which are used within all the chapters:
#import "utils/global.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge, shapes


// Fill me with the Abstract
#let abstract = [
  Electroencephalography (EEG) measures the electrical activity of neurons at the scalp with millisecond resolution @Light2010. Each recording contains unrelated activity and noise, so researchers average many repeated trials into one event-related potential (ERP) wave form to raise the signal-to-noise ratio. This averaging obscures the trial-to-trial information. An ERP image keeps the single-trial structure, by stacking sorted trials into a two-dimensional image in which recurring shapes may point to cognitive effects @Jung2001. Inspecting these images by hand does not scale well to studies with many subjects, channels, and sorting variables, and the labelled data required to automate it are scarce.
  This thesis tests whether a convolutional neural network (CNN) can detect such patterns, and whether simulated ERP images can reduce the need for manually labelled real data. 
  
  A CNN trained only on simulated images recognises one pattern type in a template real dataset but transfers less reliably to other recordings, which leaves a measurable gap between simulated and real data. Trained instead on a pool of manually labelled real images, with a pattern-preserving augmentation, a pretrained CNN reaches a balanced accuracy of 0.92 at a 64×64 input size. Convolutional networks can therefore detect these patterns reliably and at low computational cost, leaving the gap between simulated and real data and the runtime of the simulation as the main open problems.

  #v(2em)
  = Kurzfassung

Elektroenzephalografie (EEG) misst die elektrische Aktivität von Neuronen an
der Kopfhaut mit einer zeitlichen Auflösung im Millisekundenbereich @Light2010.
Jede Aufzeichnung enthält nicht zugehörige Aktivität und Rauschen, weshalb
Forschende viele wiederholte Durchgänge (trials) zu einem ereigniskorrelierten
Potenzial (ERP) mitteln, um das Signal-Rausch-Verhältnis zu erhöhen. Diese
Mittelung verdeckt die Unterschiede zwischen den einzelnen Durchgängen. Ein ERP
image bewahrt die Einzeltrial-Struktur, indem es sortierte trials zu einem
zweidimensionalen Bild stapelt, in dem wiederkehrende Formen auf kognitive
Effekte hindeuten können @Jung2001. Die manuelle Inspektion dieser Bilder
skaliert nicht gut auf Studien mit vielen Probanden, Kanälen und
Sortiervariablen, und die für die Automatisierung benötigten gelabelten Daten
sind rar. Diese Arbeit untersucht, ob ein Convolutional Neural Network (CNN)
solche Muster erkennen kann und ob simulierte ERP images den Bedarf an manuell
gelabelten realen Daten verringern können.

Ein CNN, das nur auf simulierten Bildern trainiert wurde, erkennt einen
Mustertyp in einem als Vorlage dienenden realen Datensatz, überträgt sich aber
weniger zuverlässig auf andere Datensätze, wodurch eine messbare Lücke
zwischen simulierten und realen Daten bestehen bleibt. Wird ein vortrainiertes
CNN stattdessen auf einem Pool manuell gelabelter realer Bilder mit einer
musterbewahrenden Augmentierung trainiert, erreicht es eine balanced accuracy
von 0,92 bei einer Eingabegröße von 64×64. Convolutional-Netze können diese
Muster somit zuverlässig und mit geringem Rechenaufwand erkennen. Damit bleiben
die Lücke zwischen simulierten und realen Daten und die Laufzeit der Simulation
die wichtigsten offenen Probleme.
]

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

  == Use of AI Tools / Nutzung von KI-Tools

  I used AI-based tools as support tools during this thesis, namely: OpenAI ChatGPT, Codex, and Anthropic Claude Code @OpenAICodex2026, @AnthropicClaudeIntro2026. These tools supported code development, debugging, phrasing, grammar and language revision. I did not use AI tools as independent scientific sources. I reviewed all AI-assisted input and output and remain responsible for the submitted thesis.

  == Source code repository

 The source code used for this thesis is available in the GitHub repository #link("https://github.com/s-ccs/2025_BSc_ERPgnostics")[`https://github.com/s-ccs/2025_BSc_ERPgnostics`].
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
  key: "h",
  short: "H",
  long: "Hypothesis",
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
  key: "pc",
  short: "PC",
  long: "Personal computer",
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
  key: "rq",
  short: "RQ",
  long: "Research question",
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
  title: "Automated Pattern Detection in ERP Images Using Convolutional Neural Networks (CNN)",
  degree: "Bachelor of Science",
  faculty: "Institute for Visualization and Interactive Systems (VIS)",
  department: "Computational Cognitive Science",
  major: "Data Science",
  supervisors: (
    (
      title: "Examiner",
      name: "Jun.-Prof. Dr. Benedikt Ehinger",
      affiliation: [Institute for Visualization and Interactive Systems, University of Stuttgart, Stuttgart, Germany,

      Stuttgart Center for Simulation Science, University of Stuttgart, Stuttgart, Germany 
      ],
    ),
    (
      title: "Supervisor",
      name: "Vladimir Mikheev",
      affiliation: [Institute for Visualization and Interactive Systems, University of Stuttgart, Stuttgart, Germany
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
When a person sees a bird, reads a word, moves their eyes, or presses a button, the neural activity changes within fractions of a second. To effectively study such fast processes, a measurement that is just as fast is needed. Electroencephalography (EEG) suits this task well because it records electrical activity from the scalp with millisecond-level temporal resolution. The EEG signal is difficult to interpret because the response of interest is embedded in a large amount of unrelated electrical activity; we refer to this as noise. A common solution for improving the signal-to-noise ratio is to repeat the same type of event many times, align the EEG to each event onset, and aggregate over trials, mostly by averaging @Light2010, @Luck2014. The result is an event-related potential (ERP). This averaging approach remains common practice in neuroscience and ERP research @Kappenman2021, @Donoghue2022, it has the drawback of hiding potentially valuable information @Jung2001.

What averaging obscures is the trial-to-trial information. A single trial may reflect a later button press or the next fixation rather than the event used for alignment, and fast and slow responses can cancel each other out when averaged @Jung2001, @Ouyang2017LatencyVariabilityReview. Two datasets can therefore share a similar average even when their single trials look very different.

Instead of reducing all trials to one waveform, ERP images keep the otherwise aggregated structure visible by representing the data as a two-dimensional image. Here, rows correspond to trials, columns correspond to time points, and colour represents signal amplitude. If the rows are sorted by a meaningful experimental variable, such as reaction time or fixation duration, a potential structure can become visible across trials. For example, a curved band may indicate response-locked activity, a fan may indicate increasing timing variability, and a vertical split may indicate a condition difference.

This visualisation is useful to extract more information than the common aggregating approaches. In EEG experiments with eye-tracking, neighbouring fixations and saccades can often occur close together in time. Therefore, the response to one event may overlap with the response to the next event. This overlap can make ERP image patterns appear shifted, widened, or just harder to interpret. Thus, a pattern in an ERP image can be a clue to a cognitive effect, to event overlap, or to a preprocessing issue. As a result, ERP images are useful as a first diagnostic view, often followed by more explicit overlap modelling or regression-based analysis @Dimigen2011Coregistration, @Ehinger2019Unfold.

From a single experiment, many ERP images can be derived, across different subjects, channels, conditional variables, time windows, and processing choices such as sampling rate. This makes scale a practical problem. A researcher can only inspect a limited number of ERP images manually, which may be insufficient to cover all of the recorded data of an experiment. Human annotation decisions are hard to reproduce unless the criteria are written down and applied consistently. Manually labelling data also carries the risk of false classification, which makes it necessary to compare results across multiple annotators. Automated pattern recognition is attractive because it turns this slow visual screening step into a consistent first pass. It does not replace interpretation, but it aids decision-making.

Therefore automated recognition makes these patterns accessible for large-scale analysis. This enables their use in existing ERP datasets, biomarker exploration to support patient diagnostics, and comparisons across experiments where manual inspection would otherwise be infeasible
@Pernet2011SingleTrialWhyBother, @Kappenman2021.

This automation can also provide an additional quality check. An ERP recording and its data analysis include many heterogeneous processing steps and experimental setups, such as applied filters, signal references, artefact handling, experimental variables, and time windows. These choices can preserve a visible pattern, weaken it, or create a misleading one @Clayson2021ERPMultiverse. Therefore, a detector could flag ERP images from a faulty processing pipeline @Cecotti2011P300CNN.

If the ERP image screening task is addressed with a machine-learning approach, labelled training data become the main bottleneck. ERP images sourced from recordings are heterogeneous due to differences in experimental setups, and are mostly not labelled with regard to our interest in finding visual patterns. Manual annotation remains the preferred option for real ERP images, but is expensive. With labels this scarce, this thesis does not aim to deliver a detector for real-world analysis workflows @Roy2019. It first investigates whether CNNs can detect the patterns at all.

This thesis focuses on the automated approach to find visual patterns in ERP images. The goal is to detect interpretable ERP image morphologies of interest, instead of explaining the underlying neural process or origin. Deep learning is a natural first choice for this image-based task, but only if the label bottleneck and the gap between simulated and real ERP images are handled explicitly.

Hence, we used simulation to create labelled training data under controlled assumptions. UnfoldSim.jl is used to simulate continuous event-based EEG-like time series, which can then be rendered as ERP images @Schepers2025. This setup motivates the research questions and hypotheses set out in the next section.

== Research Questions, Hypotheses, and Contributions

We investigate whether convolutional neural networks can detect visual ERP image patterns automatically, and whether simulated ERP images can reduce the need for manually labelled real data. The work is guided by the following research questions (RQ) and hypotheses (H) @Willis2023ResearchQuestionHypothesis.


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

  + *Sim-to-real transfer* <rq:sim-to-real> How accurately and efficiently can a CNN model trained on simulated ERP images detect and distinguish relevant spatio-temporal patterns in ERP images from real data?

    + *Sim-to-real sigmoid* <rq:sim-to-real-sigmoid> To what extent can a CNN trained on simulated sigmoid ERP images recognise sigmoid patterns in manually labelled real ERP images?

      - *H1.1a: Real-data recognition* <hyp:sim-to-real-sigmoid-performance> A CNN trained on simulated sigmoid and _no class_ ERP images will classify manually labelled real sigmoid and _no class_ ERP images above chance and majority-class baseline performance.

      - *H1.1b: Cross-source transfer* <hyp:sim-to-real-sigmoid-gap> A CNN trained on simulated ERP images will perform better on held-out simulated ERP images than on manually labelled real ERP images, indicating a sim-to-real gap.

    + *Simulator parameter calibration* <rq:simulator-calibration> How does simulator parameter calibration affect sim-to-real transfer and what performance gap remains compared with training on real labelled ERP images?

      - *H1.2a: Targeted simulator parameter calibration* <hyp:simulator-calibration-targeted> Simulator parameter calibration with Latin hypercube or Monte Carlo sampling will improve the sim-to-real transfer compared with broad random parameter sampling on different real validation sources.
        #v(.7em)

  + *Manual labelling* <rq:manual-labelling> How accurately and efficiently can CNNs detect visual ERP image patterns when they are trained on manually labelled real ERP images?

    + *Preprocessing* <rq:real-preprocessing> Which training and preprocessing choices improve CNN-based classification of manually labelled ERP images?

      - *H2.1: Preprocessing* <hyp:real-preprocessing> The selected preprocessing steps and their order will affect classification performance.

    + *Augmentation and class imbalance* <rq:real-augmentation> To what extent do ERP-specific augmentations or imbalance-handling strategies improve CNN model performance?

      - *H2.2: Augmentation and class imbalance* <hyp:real-augmentation> ERP-specific augmentation and imbalance-handling strategies will improve overall model performance.

    + *Model choice and image resolution* <rq:real-model-choice> Which model architecture and ERP image resolution provide the best accuracy-efficiency trade-off for manually labelled real ERP image classification?

      - *H2.3: Model choice and image resolution* <hyp:real-model-choice> Model architecture and ERP image resolution will produce measurable accuracy-efficiency trade-offs, so the preferred configuration will depend on both classification performance and computational cost.
]

Our main contributions are:

1. A manually labelled real ERP image dataset from multiple EEG/ERP sources, covering the pattern of interest.
2. A simulation pipeline for generating controlled sigmoid and _no class_ ERP images, together with calibration experiments for sim-to-real transfer.
3. A comparison of preprocessing choices, augmentation, imbalance handling, model choice and model training setups.

// Chapter 2 - Provide the conceptual basis and position the thesis briefly in
// the literature without overloading the early draft.
// ----------------------------------------------------------------------------
#pagebreak()
= Background and Related Work <chp:background>

== EEG, ERPs, and ERP Images
Electroencephalography (EEG) is a method for recording electrical brain activity from the scalp. In a typical human EEG experiment, an electrode cap is placed on the participant's head, the electrodes are connected to an amplifier, and each electrode records a voltage over a time series @Light2010. The amplitudes of the recorded potentials are on the order of microvolts. The signal recorded at a scalp electrode is not the activity of a single neuron, but rather a macroscopic field signal shaped by many local brain currents. Several factors affect the recorded EEG signal, including head geometry, volume conduction through brain tissue, and the reference potential that defines the electrical baseline for voltage measurements @Nunez2006. Hence, the strength of EEG lies in its temporal rather than spatial resolution: it can follow neural activity on the scale of milliseconds. However, its weakness is that each individual trial mixes the event-related response of interest with ongoing brain activity, eye movements, muscle artefacts, powerline noise, and other sources of measurement noise @Luck2014.

ERPs are not recorded as separate signals, but are derived from the continuous EEG by using event markers set throughout an experiment. Event markers define meaningful time points, for example stimulus onsets, the start of fixations, or participant's responses. Around each event, the continuous EEG is cut into a short epoch, usually a pre-stimulus baseline period followed by several hundred milliseconds after the event, commonly up to one second. Each epoch is referred to as a trial. Standard EEG/ERP preprocessing usually includes steps such as frequency filtering, artefact handling, and baseline correction @Luck2014, @Light2010. 
ERP Researchers then commonly align all epochs at time zero and average the trial amplitudes. This results in consistently time-locked activity, such as noise being reduced, while consistently occurring signal components remain visible in the average@Luck2014.
ERP components, such as P100, N170, and P300, are then described by their polarity, latency, amplitude, and scalp distribution @Luck2014, @Kappenman2021.

However, the drawback of this averaging method is that it deliberately removes single-trial variation. Thus, the average waveform of two datasets can appear almost identical, although one of them contains a systematic latency drift across trials, while the other contains a stable response with random noise. Jung et al. make this issue explicit in their work on single-trial ERP analysis. According to their research, relevant event-related dynamics may be hidden by the average due to the differences of individual trials in latency, amplitude, and artefact contamination @Jung2001. However, ERP images representations of the EEG channels keep this single-trial information. In this two-dimensional representation, the horizontal axis is time, the vertical axis lists trials stacked on top of each other, and the colour at each cell encodes the voltage amplitude for one trial at a time point @Jung2001.

A recent survey by Mikheev et al. attempts to standardize the notation for ERP visualisations. The survey shows that ERP images are recognised by practitioners, but that researchers do not consistently use the same name for this plot type. Instead, the literature and practitioners use several related terms, such as sorted ERP trials or just trials, which can make comparisons between studies and tools harder. We therefore use the term ERP image consistently throughout this thesis @Mikheev2024ArtOfBrainwaves.

The ERP images in this thesis are built from already recorded EEG/ERP data. The data are assumed not to be raw recordings. 
Before these data are used in this thesis, muscle artefacts, eye-related artefacts, line noise, and other unusable segments should already have been handled. Ongoing brain activity unrelated to the event of interest remains a major source of variability. The task is therefore to convert preprocessed ERP data into ERP images suitable for pattern detection.

We created the ERP images for this thesis from single channels. We cut the signal, often in the form of a time series, into event-locked trials. To do so, we discarded the signal before the event-onset. We only kept the time windows from the event onsets to at most one second after, because we expect the target patterns to be in these intervals. The trials are arranged as trial-by-time matrices. These can then be further processed and used as the models input.

== ERP Image Patterns
Sorting trials in the ERP image matrix can make underlying patterns visible. An
ERP component may shift in time, spread out with variable duration, change
polarity, or vary non-linearly @Jung2001, @Delorme2004EEGLAB,
@Delorme2015GrandERPImage. The six pattern names used in this thesis form a
practical vocabulary for this visible structure and stay distinct enough for
manual annotation. A seventh _no class_ option covers images that match none of
the six, so the scheme can also represent the absence of a target pattern.

The thesis uses the terms _class_ instance and _no class_ instance with fixed meanings. A _class_ instance is an ERP image whose labelled with one of the patterns below, regardless of the specific pattern. A _no class_ instance is an ERP image where non of the six patterns are visible. Therefore, the binary classification task collapses any of the six patterns into the _class_ side and keeps _no class_ as the other side.

#let erp-pattern-examples(pattern, left, right, note: none) = figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 3mm,
    image(left, width: 100%),
    image(right, width: 100%),
  ),
  caption: [
    Two manually labelled real-data ERP image examples for #pattern.#if note != none { [ #note] }
  ],
)

A *sigmoid* is a smooth, curved or S-shaped diagonal band. It may appear when epochs are aligned to one event, but the relevant activity is time-locked to another event whose latency varies across trials, such as a response or a subsequent fixation @Jung2001. In fixation-related data, a sigmoid-like curve may also result from overlap with neighbouring fixations @Dimigen2021RegressionEyeTrackingEEG. Examples of a sigmoid can be seen in @fig:real-sigmoid-examples.

For the following ERP image examples, the title reports the pattern, data source, channel, and sorting variable. Rows represent sorted trials, columns represent time-sample indices, and colour encodes trial amplitude.

#erp-pattern-examples(
  [sigmoid],
  "figures/erp_pattern_examples/01__sigmoid__fixations_dataset__ch096__duration.svg",
  "figures/erp_pattern_examples/02__sigmoid__eye_eeg_freeviewing__oz__fixation_duration_ms.svg",
  note: [],
) <fig:real-sigmoid-examples>

A *tilted bar* is a straight diagonal band in an ERP image. A component shifts monotonically in time across the sorted trial axis at an approximately constant rate. When trials are sorted by reaction time, this morphology can reflect early sensory components such as P100 or N100, or latency-graded activity whose timing follows the response such as P300. Single-trial ERP studies show that late P300-family latencies covary with reaction time @Jung2001, @Ouyang2017LatencyVariabilityReview, @Walsh2017P3bLatencyRT. Examples of a tilted bar can be seen in @fig:real-tilted-bar-examples.

#erp-pattern-examples(
  [tilted bar],
  "figures/erp_pattern_examples/03__tilted_bar__roamm_reading__f8__gaze_x.svg",
  "figures/erp_pattern_examples/04__tilted_bar__erp_core_n170__c6__reaction_time_ms.svg",
) <fig:real-tilted-bar-examples>

A *one-sided fan* looks like a band that opens to only one side. The earlier border of the visible activity stays close to the time-locking event. The later border moves farther out for rows with longer fixation durations in the sorted trial stack. In fixation-related data, such a pattern may occur when the timing of the next fixation depends on the duration of the current fixation @Dimigen2011Coregistration, @Dimigen2021RegressionEyeTrackingEEG. Examples of a one-sided fan can be seen in @fig:real-one-sided-fan-examples.

#erp-pattern-examples(
  [one-sided fan],
  "figures/erp_pattern_examples/05__one_sided_fan__eegeyenet_saccades__e61__saccade_duration_ms.svg",
  "figures/erp_pattern_examples/06__one_sided_fan__eegeyenet_saccades__e83__saccade_duration.svg",
) <fig:real-one-sided-fan-examples>

A *two-sided fan* is a pattern that narrows near the middle of the sorted trial stack and opens towards both ends. It consists of early vertical bands whose amplitude and polarity vary across the sorted trials. Its possible causes can overlap with other patterns @Jung2001, @Ouyang2017LatencyVariabilityReview, @Ehinger2019Unfold. In the manual annotations, the two-sided fan was the least frequently observed pattern class. Examples of a two-sided fan can be seen in @fig:real-two-sided-fan-examples.


#erp-pattern-examples(
  [two-sided fan],
  "figures/erp_pattern_examples/07__two_sided_fan__fixations_dataset__ch125__sac_amplitude.svg",
  "figures/erp_pattern_examples/08__two_sided_fan__fixations_dataset__ch058__sac_amplitude.svg",
) <fig:real-two-sided-fan-examples>

A *diverging bar* is a vertical band whose polarity reverses across the sorted trial stack. Its timing stays fairly stable, so the visible cue is a polarity flip rather than a latency drift. It may originate from a polarity change across different categorical experimental variables @Wang2020PhotosensitivePhantom, @CecottiRies2017SingleTrialDetection, @Teixeira2018EvokedPatterns,
@KovalenkoBusch2016PerisaccadicVision, @Kohl2019InterleavedDeconvolution. Examples of a diverging bar can be seen in @fig:real-diverging-bar-examples.

#erp-pattern-examples(
  [diverging bar],
  "figures/erp_pattern_examples/09__diverging_bar__unfold_face_freeview__p3__saccade_amplitude.svg",
  "figures/erp_pattern_examples/10__diverging_bar__unfold_face_freeview__o1__saccade_amplitude.svg",
) <fig:real-diverging-bar-examples>

An *hourglass* is a pinched ERP image pattern. The activity is strong at the lower and upper ends of the sorted trial stack but weak or almost absent in the middle. The two ends usually have opposite polarity, which separates it from a continuous diverging bar. Such a pattern can arise from non-linear covariate effects, cancelling response subtypes or subject groups, or changing overlap between stimulus-, fixation-, and response-locked activity @Jung2001, @Ouyang2017LatencyVariabilityReview, @Woldorff1993ERPOverlap, @Ehinger2019Unfold, @Dimigen2021RegressionEyeTrackingEEG, @Mikheev2024ArtOfBrainwaves. Examples of an hourglass can be seen in @fig:real-hourglass-examples.

#erp-pattern-examples(
  [hourglass],
  "figures/erp_pattern_examples/11__hourglass__kilo_word_erp__fc5__number_of_letters.svg",
  "figures/erp_pattern_examples/12__hourglass__eegeyenet_saccades__e21__saccade_duration_ms.svg",
) <fig:real-hourglass-examples>

== CNN-Based Pattern Detection in ERP Data
Convolutional neural networks (CNNs) are designed to learn filters and combine them across multiple layers into increasingly abstract image features @LeCun2015. For ERP images, the model must detect short temporal segments, local contrast changes, neighbouring trial rows, and larger connected shapes. Therefore a CNN is a suitable model for this image-classification task.

ERP images are not natural photographs; they are more like a heatmap of a matrix @Mikheev2024ArtOfBrainwaves. Roy et al. review deep-learning work on EEG and emphasise that performance depends heavily on preprocessing, available labels, validation design, and reproducibility @Roy2019. For this thesis, this means that a strong CNN score is not automatically evidence of a generalised and well-optimised model. It may also indicate that the model has learned a simulator artefact, a preprocessing artefact, or a dataset-specific shortcut such as noise or resolution.


//nnU-Net uses a U-Net-style encoder-decoder architecture, where the encoder compresses image information into increasingly abstract features and the decoder maps these features back to spatially resolved predictions @Isensee2021nnUNet

Experience from the field of biomedical imaging supports the same point: neural-network performance depends not only on the architecture, but also on the used data pipeline. Brain-RetinaNet, in a small MRI tumour-detection setting, uses augmentation as a primary strategy to deal with limited labelled data @Iqbal2026BrainRetinaNet. They show that preprocessing, augmentation, and validation choices can determine whether a machine-learning model learns a useful signal or a dataset-specific artefact.

== Data Simulation and Sim-to-Real Transfer

For the sim-to-real evaluation, the simulator is set up for the binary task of simulating sigmoid and _no class_ ERP images. Sigmoid images contain the target sigmoid pattern, while _no class_ images contain none of the searched ERP image patterns. This restriction was introduced as a pragmatic measure to ensure feasibility. The fixation dataset was the only available real-dataset at that time in earlier stages of the thesis #link(<tab:real-data-sources>)[@tab:real-data-sources]. Sigmoid patterns were both frequent in that dataset and stable to create in the simulator.


For the binary task of generating sigmoid and _no class_ ERP images, we implemented a simulator based on UnfoldSim.jl, a free and open-source Julia package for simulating EEG data with a particular focus on event-related potentials @Schepers2025. This framework is well suited to the present application because it generates continuous event-based EEG time series rather than limiting simulations to isolated or averaged waveforms.
The simulator used the fixation dataset as a template for parameter configurations, which has a sampling rate of 512 Hz, 2508 trials, and a trial epoch length of one second. The goal is to simulate ERP images that resemble real fixation ERP images in their basic dimensions.

=== Motivation for Simulation
Labelled real ERP image samples are scarce, which limits supervised training. Simulation provides a controlled way to generate labelled ERP data under known assumptions. The goal is not to reproduce full physiological EEG recordings, but to generate trial-by-time matrices in which known event timing, ERP component timing, and noise create interpretable ERP image patterns @Schepers2025.

=== ERP Image Output
The default conversion from EEG data to an ERP image starts with one selected EEG channel. From this channel, only the post-fixation interval is kept, trials are sorted by a selected event metadata, each time point is z-scored across trials, a Gaussian smoothing filter is applied across both axes with reflective borders, and the resulting trial-by-time matrix is resized to 64x64. In this thesis, we use this sequence as the default pipeline for converting ERP data into ERP images. Thus, the final size of the matrix is not dependent on the recording itself, but the final model-input resolution. It was chosen to make all ERP images comparable, keep training fast, and retain most of the important visual information.

=== Simulated Class Definition
At first, we considered a six-class simulation setup, but we figured that handling all visual patterns was infeasible. In the current simulation design, each simulated ERP image contains all previously mentioned ERP components, so a single image can contain traces of several other pattern classes. We made that decision because the visual patterns are sensitive to small parameter changes. A slight change in timing, amplitude, or noise can move, deform, remove a pattern, or produce a shape that no longer matches its visual description. Therefore, the final experiments use sigmoid as the only positive ERP image pattern. This pattern was the most robust against parameter changes of the six simulated patterns and also the most frequently observed pattern in the available fixation dataset.

The simulator does not use an additional run for each pattern. Instead, the same simulated ERP activity can reveal different structures depending on the trial sorting. The _no class_ label is created from the same simulated activity with a random trial order, which removes the systematic relation between trials.Hence, it is not an empty or all-noise image.

=== Sim-to-Real Gap
Strong performance on synthetic data does not guarantee real-data performance, because the model may learn properties that are specific to the simulator. This risk can be reduced by varying the simulator parameters widely, making real ERP images more likely to fall within the range of simulated examples @Tobin2017. In this thesis, the same principle applies: If parameters such as component widths, latency gaps, amplitudes, basis shapes, or noise levels are varied too aggressively, the target pattern itself becomes implausible or disappears.

=== Simulator Parameters
Each simulation run samples a set of global and component-level parameters. The _global parameters_ are the lognormal event-onset process, the number of trials, the sampling rate, and the epoch duration. The _component parameters_ define P100, N170, and P300 basis functions through their widths, Hanning window centres, relative gaps, peak offsets, and amplitudes. The component timings are partially dependent because later components are placed relative to earlier components. The N170 window follows the P100 window by a sampled gap, and analogously the P300 window follows the N170 window by another sampled gap. To create _no class_ images, the simulated trials were assigned a random trial order.

=== Parameter Search
The parameter search aims to identify simulator parameters that maximize the performance of a CNN trained on simulated images and validated on the real fixation dataset. We addresses #link(<rq:simulator-calibration>)[RQ 1.2] with this design, by testing whether better simulator calibration can improve sim-to-real transfer and whether a remaining gap to real-label training persists.



Therefore the goal for the simulation-based experiments is to increase variation while remaining close enough to the real sigmoid morphology. The exposed search space contains 24 parameter specifications. The calibration problem becomes 48-dimensional because each specification has a mean and a standard deviation. Here, the following five search strategies were explored:

*Broad random search.* Also called domain randomisation. This search strategy samples the parameter space as a random exploration baseline, with each parameter given the same weight @Tobin2017.

*Latin hypercube sampling (LHS).* This search strategy divides continuous parameter ranges into discrete intervals and combines them through permutations, spreading candidates more evenly across dimensions than with ordinary sampling @McKay1979.

*Heterogeneity scaling.* This search strategy was a naive parameter-search approach that scaled the standard deviations of normally distributed parameters to 50%, 100%, and 1000% of the corresponding mean magnitudes.

*Monte Carlo random search.* This search strategy provides a simple comparison by uniformly sampling one parameter at a time, and remains a viable option because random search is a strong baseline for high-dimensional hyperparameter spaces @Bergstra2012.

*Two-zone mixture.* This search strategy uses 70% of the best parameters from the LHS baseline and 30% edge-case configurations @Singh2013BalancedSequentialDesign.

== Related Work and Positioning
The closest related work starts with single-trial ERP analyses. Grand averages remain useful summaries, but they can hide latency jitter, response subtypes, and systematic links between EEG and behaviour. ERP images and related single-trial methods address this by keeping trial-wise structure visible @Jung2001, @Pernet2011SingleTrialWhyBother, @Ouyang2017LatencyVariabilityReview.

CNNs have already been used successfully for EEG decoding and for P300 detection. EEG-specific models such as DeepConvNet and EEGNet show that convolutional architectures can learn useful temporal and spatial filters from EEG data @Cecotti2011P300CNN, @Schirrmeister2017DeepConvNet, @Lawhern2018EEGNet. However, the present task is different. The model does not predict a stimulus class or a brain-computer interface (BCI) command, but whether an ERP image contains a named visual morphology.

A self-adapting framework for medical image segmentation, like nnU-Net, has its main contribution outside the network architecture @Isensee2021nnUNet. The U-Net Model itself stays nearly unchanged, while preprocessing, training, and augmentation, are defined as adjustable for the whole training pipeline. This thesis adopts the same principle for ERP image classification, treating preprocessing, augmentation, and training choices as the main targets of systematic comparison.

Pharmacological EEG/ERP research already analyses drug-effect recordings with established EEG and ERP methods. Companies such as PI PharmaImage record EEG and ERP data across drug trials and maintain reference databases of more than 4000 subjects @PIPharmaImage. Researchers then evaluate these recordings for drug effects and safety signals. An automated pattern detector for ERP images can extract more insights from this scale of data and open up lines of investigation that manual review cannot cover.

Tukey and Tukey first proposed computing summary statistics over a large set of scatterplots, so that an analyst can rank or filter views before inspecting them @TukeyTukey1985. Wilkinson et al. later formalised this idea as scagnostics, a fixed set of graph-theoretic measures such as outlying, clumpy, stringy, or monotonic @Wilkinson2005Scagnostics. Matrices with an assigned score in this space can therefore guide the selection of scatterplots for closer inspection.

Behrisch et al. extend this diagnostics idea from scatterplots to adjacency matrices and call it Magnostics @Behrisch2017Magnostics. They use handcrafted image-based descriptors to score how strongly a given matrix shows certain visual patterns, so that interesting matrices can be retrieved from a large collections. ERP images are also ordered matrices whose interpretable content depends on the row ordering, which makes the same exploratory framing useful. The output scores from a trained CNN model can then highlight the ERP images that may contain a pattern of interest across many subjects, channels, and trial-sorting variables, narrowing a broad search space to views worth a closer look.

This CNN feeds into ERPgnostics, a Julia package that applies the diagnostics idea directly to ERP image collections @ERPgnosticsDocs. The class probabilities act as the pattern measures and surface the ERP images that may contain a pattern of interest from unseen ERP datasets. Therefore, the contribution of this thesis is the classifier itself, while ERPgnostics provides the surrounding exploration framework.

// ----------------------------------------------------------------------------
// Chapter 3 - Describe the empirical pipeline in one place.
// For an initial draft this is easier to navigate than splitting data and
// methods into separate chapters.
// ----------------------------------------------------------------------------
#pagebreak()
= Data and Methods <chp:data-methods>
This chapter describes the simulated and real data sources, the annotation workflow, the ERP image preprocessing pipeline, the model-training, and evaluation setup.

== Datasets, Annotation, and Task Definition
To train and evaluate the classifiers, we use real ERP images from several ERP datasets. @tab:real-data-sources lists the sources of this labelled pool. For datasets that already included preprocessed EEG or ERP files, this thesis uses those files directly. When a source was available only as raw recording, it was prepared according to the code provided by the original authors. All sources were inspected manually before they were used for annotation and model training. The CNN therefore trains on preprocessed ERP images instead of raw EEG recordings.

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
      [2022],
      [Reference Fixation Dataset, fixation-locked EEG and eye-tracking reference data],
      [79 patterns out of 588 total],
      [@Gert2022WildLab],

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

Hand-labelling every dataset, participant, channel, and sort variable was not feasible for this thesis. The annotation workflow therefore used three passes. Each step narrowed the selection further.

In the first pass, we labelled ten images per dataset to check whether the conversion pipeline from EEG to ERP produced ERP images that we could use for this thesis. Sources that failed this check were dropped.

In the second pass, a CNN trained on the existing labels from the reference fixations dataset scored the remaining images. We then labelled the 200 highest-ranked images per source, spread across the available sort variables. This directed the labelling effort toward images most likely to contain a pattern.

In the third pass, we kept only those datasets and sort variable combinations in which we had found at least one pattern during the previous two passes. We then labelled all available channels for each of them. Participant numbers varied across sources, ranging from single-participant sources to sources with multiple participants.
We used one participant per source so the labelled pool would cover more different experiments and introduce more heterogeneity in pattern shapes.

#pattern-decision-tree <fig:pattern-decision-tree>

== Preprocessing
This section describes the preprocessing pipeline that turns each EEG/ERP recording into the ERP image the CNN receives as input. A single EEG channel is selected, the post-event interval is extracted, trials are sorted by the chosen event metadata, each time point is z-scored across trials, a Gaussian smoothing filter is applied across both axes, and the resulting trial-by-time matrix is resized for model input.

Sorting and z-scoring were kept constant across all experiments because they defined the ERP image input used for this task. After sorting, z-scoring was applied per time point to remove vertical bands caused by strong responses shared across many trials. Because these bands were not the main morphology of interest, they were removed before smoothing could spread them to neighbouring time points. The same preprocessing order was used for simulated and real ERP images to ensure comparability.

@fig:basic-preprocessing-pipeline shows the four steps applied to one real ERP image. The visualisation makes the cumulative effect of the pipeline visible, so that the pattern becomes easier to detect step by step.

All ERP image figures in this thesis use the same colour mapping. A diverging red-blue colormap is anchored at zero, so positive amplitudes appear red and negative amplitudes appear blue. The colour range is clipped to the 1st and 99th percentile of the image values, which keeps the scale robust to outliers and preserves contrast in the image. The colorbar shows ticks at the 1st percentile, zero, and the 99th percentile.

The experiments in @sec:experiments-setups add, remove, or vary individual steps of this pipeline to test how the change affects the ERP image visually and the model performance. They compare resizing with different interpolation functions, pipelines with and without Gaussian smoothing, z-scoring before versus after resizing, value binning in the ERP matrix, input resolutions from 16x16 to 256x256, and additional visual noise-reduction filters.

#let pipeline-panel(num, path) = align(center)[
  #image(path, width: 100%)
  #v(-2em)
  #text(weight: "bold", size: 9pt)[(#num)]
]

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 3mm,
    row-gutter: 1mm,
    pipeline-panel(1, "figures/pipeline_visualisations/figure_1a_raw_erp_image.svg"),
    pipeline-panel(2, "figures/pipeline_visualisations/figure_1b_sorted_by_duration.svg"),
    pipeline-panel(3, "figures/pipeline_visualisations/figure_1c_z_scored_per_time_point.svg"),
    pipeline-panel(4, "figures/pipeline_visualisations/figure_1d_gaussian_smoothed.svg"),
  ),
  caption: [Basic preprocessing pipeline applied to one real ERP image from the fixations dataset, channel ch096, sorted by duration. (1) Raw trial-by-time matrix. (2) Trials sorted in ascending duration order. (3) Per-time-point z-scoring. (4) Gaussian smoothing.],
) <fig:basic-preprocessing-pipeline>

== Real-to-Real Data Augmentation <sec:real-to-real-augmentation>
This section introduces trial slicing, a data augmentation strategy developed for this thesis. The labelled real-data pool is limited and the trial counts vary widely across sources, so trial slicing turns one parent ERP image into several smaller ones of fixed shape. This expands the labelled pool without introducing new data or additional labels, and it equalises the trial dimension across sources.

Data augmentation also improves model robustness. We used augmentation to increase the variation of the labelled ERP image training set and to reduce memorisation of individual examples. _Brain-RetinaNet_ provides a domain-distant but useful reference point for this small-data setting. Iqbal et al. applied targeted augmentation to a small labelled MRI detection dataset and report improvements across several detector backbones @Iqbal2026BrainRetinaNet. The analogy only concerns the limited-data setting, but it motivates the evaluation of class-aware ERP image augmentation and class balancing as central training decisions.

Trial slicing operates on an already sorted ERP image of shape $(N, T)$, where $N$ is the number of trials and $T$ is the number of time samples. It cuts the image horizontally along the trial axis into smaller ERP images of fixed target trial count $n$. Let

$ K = floor(N / n) $

denote the number of full slices that fit into the parent image, and let $r = N - K n$ be the number of leftover trials. The trials are visited in their existing sort order and distributed one at a time across the $K$ full bins and one additional remainder bin. The remainder bin receives one trial per round for the first $r$ rounds, after which the full bins continue alone until each holds exactly $n$ trials. The remainder bin is then filled up to size $n$ with random unique trials from the same parent matrix. When each bin is rendered into an ERP image, its trials are re-sorted by the chosen sort variable, to insure integrity. From one parent ERP image with $N$ trials this produces

$ K + 1 = floor(N / n) + 1 $

sliced ERP images, where the first $K$ slices are mutually disjoint and the remainder slice shares fill-up trials with the others.

Trial slicing combines several practical advantages in one easy-to-reproduce procedure. The round-robin distribution takes trials evenly from across the parent ERP image, so each slice preserves the global pattern shape rather than risking capturing local clusters of neighbouring trials. A purely random selection would carry a clumping risk where neighbouring trials over-represent one part of the sorted trial axis and could visually distort the pattern. The procedure is almost entirely deterministic, with the remainder fill-up as the only random step. It is also cheap to compute, since assigning trials to slices is a modulo operation on the sorted trial index. 

The fixed slice size matters because the real data sources differ widely in trial count, while the CNN expects a uniform input shape, and because the goal is generalisation across data sources. Equalising the trial dimension keeps the model from learning trial count itself as a shortcut feature. Finally, different combinations of trials lead to different background-noise textures across slices, which adds useful variation for the classifier, since two slices that contained the same trials would produce the same noise.

Each sliced ERP image is then expanded into four augmented variants before the remaining preprocessing of z-scoring, Gaussian smoothing, and resizing is applied. The first variant is the slice as it is, with its original trial order and signal polarity. The second variant reverses the trial order, so the slice is flipped along the sort axis. The third variant inverts the signal polarity by multiplying every value in the trial-by-time matrix by minus one. The fourth variant combines both operations, with reversed trial order and inverted polarity. All four variants carry the same class label as their parent slice, because the visual pattern that defines the _class_ is preserved.

This expansion makes the classifier more robust in two ways. The reversed trial order forces the CNN model to recognise the pattern from either reading direction along the sort axis. The polarity inversion removes the dependency on the absolute sign of the EEG signal, which can differ between recording references and between datasets. As a side effect, quadrupling the labelled examples also increase the amout of images.

The slicing step is applied asymmetrically to _class_ and _no class_ instances in order to counter the strong imbalance in the labelled pool, where _no class_ clearly dominates over _class_. For a _class_ instance, all $K + 1$ slices are kept and each receives the four augmentation variants, yielding $4 (K + 1)$ training images per parent recording. For a _no class_ instance, only one slice per parent recording is kept, with the four augmentation variants, so that one _no class_ parent contributes only 4 training images. This rule lifts the _class_ training count by a factor of $K + 1$ while leaving the _no class_ count unchanged, which brings the two training counts closer to balance and reduces the pull of the majority _no class_.
We chose $n = 200$ as the target trial count, as a small common denominator across all labelled data sources. A second reason is class balance. Together with the asymmetric slicing, this lifts the minority class share enough to bring the augmented pool close to an even split, as @tab:augmentation-trial-balance and @tab:manual-label-pool-results show.

#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.7fr, 1.25fr, 0.75fr, 1.05fr, 1.05fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x == 0 {
          left + top
        } else {
          center + top
        }
      },
      table.header(
        [Dataset],
        [Trial count and sampling rate],
        [Slices per parent],
        [Class share before aug.],
        [Class share after aug.],
      ),

      [Reference Fixations],
      [2508 trials × 512 Hz],
      [13],
      [13.4 %],
      [66.9 %],

      [ERP CORE N170],
      [1147 trials × 256 Hz],
      [6],
      [6.4 %],
      [29.2 %],

      [ERP CORE N2pc],
      [1128 trials × 256 Hz],
      [6],
      [5.2 %],
      [24.9 %],

      [Kilo-Word ERP],
      [960 trials × 250 Hz],
      [5],
      [4.7 %],
      [19.9 %],

      [EYE-EEG Reading],
      [297 trials × 512 Hz],
      [2],
      [0.8 %],
      [1.6 %],

      [EYE-EEG Freeviewing],
      [435 trials × 500 Hz],
      [3],
      [7.5 %],
      [19.6 %],

      [EYE-EEG Sceneviewing],
      [644 trials × 500 Hz],
      [4],
      [11.1 %],
      [33.3 %],

      [EEGEyeNet Saccades],
      [346 trials × 500 Hz],
      [2],
      [17.2 %],
      [29.4 %],

      [ROAMM],
      [997 trials × 256 Hz],
      [5],
      [7.1 %],
      [27.7 %],

      [Unfold Face FV],
      [1800 trials × 500 Hz],
      [10],
      [18.6 %],
      [69.5 %],
    ),
    caption: [Per-source parent ERP image shape, slice count produced by the trial slicing step with $n = 200$, and class share of the labelled pool before and after augmentation.],
  ) <tab:augmentation-trial-balance>
]

// Sources:
// - notebooks/week_21/labelstudio_annotations_all.csv (manual annotation counts per erp_class)
// - notebooks/week_23/outputs/augmentation_inverse_sort_polarity/augmented_label_summary.csv (final augmented counts incl. 4 variants)
// - notebooks/week_23/outputs/augmentation_inverse_sort_polarity/run_config.json (n_augmented_images 18596, n_base_mod_images 4649, target_trials 200)
#[
  #show figure: set block(breakable: true)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    align(center, table(
      columns: (36mm, 28mm, 32mm),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => if y == 0 { center + horizon } else if x == 0 { left + top } else { center + top },
      table.header(
        [Label],
        [Manually found],
        [Augmented images],
      ),

      [sigmoid], [116], [4,268],
      [diverging bar], [53], [1,632],
      [one-sided fan], [49], [392],
      [tilted bar], [37], [1,228],
      [hourglass], [33], [548],
      [two-sided fan], [6], [228],
      [All six patterns], [294], [8,296],
      [no class], [2,575], [10,300],
      [Total], [2,869], [18,596],
    )),
    caption: [Manual label pool by pattern. The Manually found column counts the manual labels for each pattern. The Augmented images column counts the final training images after trial slicing with $n = 200$ and the reference, inverse-sort, inverse-polarity, and combined variants per slice.],
  ) <tab:manual-label-pool-results>
]

@fig:trial-slicing-augmentation visualises the combined slicing and augmentation step on the same fixations recording used in @fig:basic-preprocessing-pipeline. The parent image enters this step as a sorted ERP image only, without z-scoring or Gaussian smoothing. Those two steps are applied separately to every augmented child image, so each child carries its own per-time-point z-scoring and its own smoothing. The visualisation slices the parent into four equal parts, which is for demonstration purpose. To save space, the individual colorbars are omitted and one shared colorbar is used for all panels in this plot. 


#figure(
  image("figures/pipeline_visualisations/figure_2_trial_slicing_augmentation.svg", width: 100%),
  caption: [Trial slicing and augmentation on one real ERP image from the fixations dataset, channel ch096, sorted by duration. Rows correspond to the four slices, columns to the four augmentation variants reference, inverse sort, inverse polarity, and inverse sort combined with inverse polarity. Each panel shows the slice after its own per-time-point z-scoring and Gaussian smoothing.],
) <fig:trial-slicing-augmentation>

The visualisation shows that the underlying pattern is preserved across all augmented children. The four slices each contain a disjoint subset of trials of the parent recording.

== Model Training Setup
This section describes the classifier choices, the training setup, and the cross-validation fold design used to fit and evaluate the models on the labelled ERP image pool.

The classifier addresses a binary task, distinguishing _class_ from _no class_. We use this binary setup for feasibility, because the pattern classes appear at different frequencies in the labelled pool, and to keep the problem simpler. For model comparison we used CNNs with one, three, and ten convolutional layers, together with a pretrained ResNet18 model @He2016ResNet. ResNet is useful here because its residual connections, which are skip paths that let a layer pass its input directly to a later layer in addition to the transformed signal, make a deeper image classifier easier to optimise, while still providing a standard computer-vision baseline for comparison. Having the best practices already implemented, rather than figuring them out from scratch, makes it easier to focus on the classification and preprocessing of the ERP images.

We used 5-fold cross-validation. The augmented images are distributed across the five folds as evenly as possible, balanced by total count, class label, and augmentation variant. The four augmentation variants from inverse sort order and polarity that share a parent image always land in different folds, so the model is never validated on a similar variant of an image it was trained on.

== Experimental Setups <sec:experiments-setups>
The empirical work consists of a numbered series of experiments.
A few conventions are shared across the experiments unless an experiment overrides them. Each parameter configuration for simulation is evaluated three times with different random seeds, and the reported BAcc and macro-F1 are the means across these three repeats. This averages out random effects from the simulator draws. Simulator parameter searches compare three sampling strategies, namely broad random, Latin hypercube, and Monte Carlo, against a hand-crafted baseline configuration that we also call the starting parameters. The simulator generates 1,000 images per pattern per parameter combination. ResNet18 was the most used classifier in this thesis, trained for 8 epochs without early stopping at batch size 64, learning rate 3e-4, and label smoothing 0.02. These settings are default recommendations rather than the result of a hyperparameter search. We mostly use the model BAcc as the measure of how well an experiment variable performs and optimise towards it, whereas earlier attempts optimised towards macro-F1.

=== Experiment E1 16x16 dense-network parameter search <exp:dense16>
E1 runs the simulator parameter search at 16x16 input ERP image resolution, with the same setup repeated at 8x8, 4x4, and 2x2 to locate a lower end of an optimal resolution, and once more at 16x16 without Gaussian smoothing to measure how much the smoothed pipeline matters. A dense neural network replaces ResNet18 because a 16x16 input would lose its spatial structure after only a few convolution stages. Each strategy proposes twelve parameter combinations, trained for 30 epochs and validated on the fixation dataset. No augmentation was used.

=== Experiment E2 Sim-to-sim training-set fit <exp:simfit>
E2 reuses the best Latin-hypercube candidate from E1 at 16x16 with Gaussian smoothing and additionally evaluates the trained dense network on its own simulated training set. The purpose of the experiment is to measure the gap between synthetic validation and real-data evaluation for the same calibrated configuration. E2 also runs without augmentation, so the sim-to-sim fit reflects the raw calibrated images. 

A separate preprocessing-pipeline variant is run alongside E2 to probe how robust the calibration is to small changes in the image pipeline. In that variant z-scoring is applied after downscaling instead of before, and the smoothing kernel is altered for the simulated ERP images only.

=== Experiment E3 Simulator-first downstream model comparison <exp:simrank>
E3 is the only experiment that holds the simulator fixed and varies the downstream classifier instead. It grid-searches three downstream models, namely a dense neural network, a random forest, and a support-vector machine, across the three sampling strategies and four input resolutions of 2x2, 4x4, 8x8, and 16x16. The goal is simply to see how other machine-learning models perform on the simulator-trained output and resolution. This comparison also uses no augmentation.

=== Experiment E4 64x64 ResNet18 parameter search <exp:resnet64>
E4 repeats the simulator parameter search with ResNet18 at 64x64 input resolution to test whether the same simulator can train a higher-capacity classifier used in the real-data experiments. The search budget per strategy is 12 parameter combinations for broad random, 48 for Latin hypercube, and 48 for Monte Carlo, for a total of 325 ResNet18 training and evaluation runs. Runtime is logged per repeat for image simulation, training, and inference. The simulated training images enter the network without augmentation.

=== Experiment E5 Cross-source sim-to-real transfer <exp:cross-source>
E5 tests whether a ResNet18 trained on the best simulator parameters from E4 generalises beyond the fixations dataset. For each of the three top candidates from the search strategies, a ResNet18 is trained on simulated data passed through the trial slicing and inverts augmentation pipeline. Using the same augmentation as the real-to-real experiments lets us see whether it acts on simulated images as it does on real ones. A fourth ResNet18 model trained only on the labelled fixation dataset serves as baseline.

=== Experiment E6 Shallow-CNN versus ResNet baseline <exp:depth>
E6 compares three shallower CNN baselines with 1, 3, and 10 convolutional layers against the pretrained ResNet18 under the same five-fold cross-validation on the labelled ERP image pool. The motivation is whether smaller models could save training time without losing accuracy. ResNet34 is included as an additional comparison point. E6 trains with the full trial-slicing augmentation, including the inverse-sort and polarity variants from @sec:real-to-real-augmentation.

=== Experiment E7 Image pipeline and processing-step choices <exp:preprocessing-sweep>
E7 varies single steps of the ERP image pipeline, validated on the reference fixations dataset only, and measures how each change affects both the visible image and the classification score. The qualitative part compares the resize interpolation function across nearest, linear, quadratic, cubic, and Lanczos sampling, the pipeline with and without Gaussian smoothing, per-time-point z-scoring before versus after the resize, and amplitude binning from 2 up to 32 discrete levels against the continuous colorbar. The quantitative part trains the 1-, 3-, and 10-layer CNNs and the pretrained and random initialised ResNet18 from @exp:depth at 64x64 under five-fold cross-validation and compares the reference pipeline with three added steps, namely per-image clipping at the 1st and 99th amplitude percentile, per-image scaling to a fixed range from -1 to 1, and class balancing to a 50/50 split by dropping _no class_ instances. E7 builds its training images with an early prototype of that augmentation, the sorted modulo 4 trial split, which slices the trials of the reference fixations recording into four interleaved subsets, without the inverse-sort and polarity variants.

=== Experiment E8 Noise reduction and morphological filtering <exp:filters>
E8 adds an image filter to the pipeline, validated on the reference fixations dataset only, and asks two questions. 
First, it screens 40 image filters, most of them from the JuliaImages image-processing packages @JuliaImages, to find which noise-reduction or morphological filter helps most to improve model performance, covering morphological operations and edge filters. In this screen the filtered image forms a second input channel next to the standard ERP image channel, with the filter applied after z-scoring and before the Gaussian low-pass, so the model sees the unfiltered and the filtered image together.

Second, it tests how the eight best performing morphological filters combine with Gaussian smoothing, comparing a Gaussian-only reference, Gaussian followed by the filter, the filter followed by Gaussian, and the filter without Gaussian. Both a pretrained and a randomly initialised ResNet18 are trained. Like E7, E8 trains on this early sorted-modulo 4 split prototype augmentation.

=== Experiment E9 Image and ERP-specific augmentation with imbalance handling <exp:augmentation>
E9 studies augmentation and class imbalance on the reference fixations dataset only. It first checks whether generic image augmentations such as rotations or croping keeps the ERP image label intact, then compares ERP-specific augmentations, namely trial dropout, pink-noise addition, time jitter, and a combination of these. Each tested variant is added on top of the same modulo 4 split.

As a separate branch, E9 tests three imbalance strategies without augmentation, each with a tuned decision threshold. Class-weighted cross-entropy raises the loss weight of the minority class, so misclassifying a pattern costs more than misclassifying a _no class_ image. Focal loss adds a factor to the cross-entropy that shrinks the contribution of examples the model already predicts with high confidence, which are the _no class_ images, so the remaining training signal comes from the rarer pattern cases. Balanced batches resample the data when each mini-batch is formed and draw the minority class more often, with repetition, so the model sees roughly as many _class_ as _no class_ images per update instead of mostly _no class_ images. The classifier is a randomly initialised ResNet18 so the comparison reflects the augmentation rather than ImageNet pretraining.

=== Experiment E10 Input resolution and model capacity <exp:resolution>
E10 extends the capacity comparison of @exp:depth across the input resolution on the reference fixations dataset only. It trains the 1-, 3-, and 10-layer CNNs and the pretrained ResNet18 at resolutions from 16x16 up to 256x256 under five-fold cross-validation and logs the per-fold training time. The aim is to find the smallest resolution at which the higher-capacity models still separate the two classes and to expose the accuracy and runtime trade-off. The shallow CNNs run at every resolution, while ResNet18 and the 10-layer CNN start at 64x64 since of a minimum input size required. E10 also uses the early modulo 4 split augmentation.

== Model evaluation
To keep the comparison manageable, the evaluation uses two classification metrics, balanced accuracy and macro-F1, together with training and inference time. Balanced accuracy is robust to class imbalance, and macro-F1 weighs precision and recall equally across the two classes.

We used Julia for all simulation, preprocessing, model training, and evaluation. Julia fits this role because it targets numerical and scientific computing and compiles programs to efficient machine code. @Bezanson2017Julia. The ResNet-based models run in _Flux.jl_, the Julia machine-learning library used for the training code @Innes2018Flux. _Metalhead.jl_ provides the ResNet model family and the ImageNet-pretrained weights @MetalheadDocs2026.

All simulations, model training, and evaluation in this thesis ran on a single PC with an AMD Ryzen 7 7800X3D 8-core CPU, 64 GB of system memory, and an NVIDIA GeForce RTX 4070 GPU with 12 GB of VRAM running Linux.


// ----------------------------------------------------------------------------
// Chapter 4 - Present observations in the same order as the pipeline.
// Keep this chapter descriptive; save interpretation for the discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Results <chp:results>
Results are split into two parts. The first part reports the simulator-side experiments E1 to E5, which calibrate the simulator and probe how far a simulator-trained model carries over to real ERP images. The second part reports E6 to E10 on the real-data pool. The experiment numbers reflect the order of presentation, not the order in which we ran the experiments. E7 to E10 were early attempts that vary the preprocessing, filtering, augmentation, and input resolution on the reference fixations dataset only. E6 came last and is the final real-data experiment, where it compares classifier capacities on the full labelled pool.

== Data Simulation and Calibration Results
The best ERP image processing pipeline setting across simulation experiments uses sorting, per-time-point z-scoring, Gaussian smoothing, and a final resize. A (near)perfect fit on the simulated training set was observed frequently across the evaluated setups. 

=== Results of E1 (16x16 dense-network parameter search)
@tab:simulation-parameter-search-results presents the best run per search method and resolution, validated against the Reference Fixations dataset. With smoothing deactivated for both the simulated and the real ERP images, the best 16x16 dense-network row drops from BAcc 0.711 to 0.42.

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
    caption: [Best Dense-NN run per search method and resolution. At 16x16 all four configurations are listed. For the smaller resolutions of 8x8, 4x4, and 2x2 only the single best run across search methods is shown],
  ) <tab:simulation-parameter-search-results>
]

Two further search strategies described in the methods chapter do not contribute a separate row to @tab:simulation-parameter-search-results. As part of the heterogeneity-scaling family, a simpler variant that only increased the standard deviations of the normally distributed parameters was discarded after visual inspection. The resulting ERP images and their visible patterns differed strongly from the real labelled images, so this variant was not investigated further.

The two-zone mixture was tested in an early prototype together with all of the other parameter-finding strategies under the altered preprocessing pipeline described as part of E2 in @exp:simfit. Every evaluated strategy, including the two-zone mixture itself, collapsed to a balanced accuracy of 0.5 or lower under that altered pipeline. The failure is therefore not specific to the two-zone mixture and confirms empirically that pipeline adjustments can produce large changes in classification outcomes.

=== Results of E2 (sim-to-sim training-set fit)
The Latin hypercube row from E1 holds the best simulator parameters found in that search, and these parameters are reused here to generate the ERP images on which the dense network is trained. @tab:simulation-calibration-results presents the NN performance on additional simulated training set and on the reference fixation dataset.

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
@tab:simulation-model-comparison-results results are all at 16x16 under Monte Carlo random search. The dense neural network is the strongest of the three downstream models, ahead of the random forest and the support-vector machine with radial basis kernel. The absolute scores in this table are lower than in the other simulation experiments because E3 was run as an early experiment, before the simulation setup was extended to its final form.

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

      [Support-vector machine],
      [0.641],
      [0.500],
    ),
    caption: [Top three downstream models from the simulator-first ranking of E3.],
  ) <tab:simulation-model-comparison-results>
]

=== Results of E4 (64x64 ResNet18 parameter search)
@tab:simulation-resnet18-64-results shows the hand-crafted starting parameters and the best candidate from each search method. All three strategies clearly improve over the starting parameters at this resolution.

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

The Best at iteration column shows that no later candidate of the same strategy reached a higher BAcc. For instance for Latin hypercube the remaining 27 of 48 candidates therefore yielded no further improvement.

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

Earlier simulation experiments showed that about 84 percent of the per-image time cost comes from the ERP simulation itself, while only about 16 percent is spent on the subsequent image-processing steps. Among the image-processing steps, the low-pass filter takes almost all of the time at around 96 percent, while sorting, z-scoring, and resizing together make up the small remainder.

=== Results of E5 (cross-source sim-to-real transfer)

// Sources
// notebooks/data_generation/outputs/cross_source_sim_to_real_sigmoid/metrics_summary.csv
// notebooks/data_generation/outputs/cross_source_sim_to_real_sigmoid/h1_1b_gap_summary.csv
// notebooks/data_generation/outputs/cross_source_sim_to_real_sigmoid/metrics_per_run.csv
@tab:cross-source-sim-to-real-performance reports the balanced accuracy of the four E5 models across the simulated holdout split and the five labelled real sources. Each numeric cell is the mean over non-collapsed repeats.

#[
  #show figure: set block(breakable: true)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (2.3fr, 1.1fr, 1.1fr, 1.1fr, 1.1fr),
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
        [Validation source],
        [Real fixations baseline],
        [Sim broad random],
        [Sim Latin hypercube],
        [Sim Monte Carlo],
      ),

      [Simulated holdout],
      [-],
      [0.756],
      [0.714],
      [collapsed],

      [Reference Fixations],
      [0.993],
      [0.602],
      [0.577],
      [collapsed],

      [ROAMM],
      [0.715],
      [0.606],
      [0.569],
      [collapsed],

      [ERP CORE N170],
      [0.618],
      [0.580],
      [0.504],
      [collapsed],

      [ERP CORE N2pc],
      [0.510],
      [0.555],
      [0.564],
      [collapsed],

      [EYE-EEG Freeviewing],
      [0.556],
      [0.459],
      [0.462],
      [collapsed],
    ),
    caption: [Mean balanced accuracy of the four ResNet18 models from E5 across the simulated holdout split and five labelled real sources that contain sigmoid patterns. Each numeric cell is the mean over non-collapsed repeats. The collapse marks a repeat as such when the less frequent predicted class accounts for less than 5 percent of predictions. The real fixations baseline is not evaluated on the simulated holdout split.],
  ) <tab:cross-source-sim-to-real-performance>
]

== Classification Performance on Real Data
This section reports how the classifiers perform on the manually labelled real data.

=== Results of E6 (shallow-CNN versus ResNet baseline)
The per-fold training times of the shallow CNN baselines stay within a few seconds of the ResNet18, a difference that is negligible for the project budget. The accuracy gap is much larger. The 1- and 3-layer CNNs often collapse to a single predicted class, the 10-layer CNN improves marginally, and the pretrained ResNet18 reaches the overall best performance on the same data. ResNet34 matches the ResNet18 accuracy with increased training time, so the deeper variant adds cost without accuracy so the ResNet18 remains the chosen model. @tab:real-data-classification-results reports the full-pool results.

// Sources:
// - notebooks/week_21/outputs/resnet18_labeled_erp_cv/metrics_summary.csv
// - notebooks/week_21/outputs/resnet34_labeled_erp_cv_128_resize/metrics_summary.csv
// - notebooks/week_23/outputs/augmentation_inverse_sort_polarity/metrics_summary.csv
#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.3fr, 3.7fr, 0.7fr, 0.8fr),
      inset: (x: 4pt, y: 3pt),
      stroke: (x: none, y: 0.45pt + luma(190)),
      fill: (_, y) => if y == 0 { rgb("#f3f5f7") },
      align: (x, y) => {
        if y == 0 {
          center + horizon
        } else if x == 2 or x == 3 {
          center + top
        } else {
          left + top
        }
      },
      table.header(
        [Model],
        [Changed variables],
        [BAcc],
        [Macro-F1],
      ),

      [ResNet18 pretrained],
      [Trial slicing and inverse sort and polarity augmentation],
      [0.918],
      [0.917],

      [ResNet18 pretrained],
      [Only trial slicing, no inverse sort or polarity augmentation],
      [0.866],
      [0.866],

      [ResNet34 pretrained],
      [increased model size and ERP image size to 128x128. Only trial slicing, no inverse sort or polarity augmentation],
      [0.854],
      [0.854],
    ),
    caption: [Classification results from E6 on the full labelled pool, comparing the ResNet18, the deeper ResNet34, and the inverse-sort and polarity augmentation. All runs use the same labelled pool.],
  ) <tab:real-data-classification-results>
]

=== Results of E7 (image pipeline and processing-step choices)
The qualitative comparison of the resize interpolation function showed little visible effect. After visual inspection, the nearest neighbou, linear, quadratic, cubic, and Lanczos sampling all look very similar to indistinguishable across resolutions from 16x16 to 128x128. We therefore keep linear as the default and did not investigate the interpolation functions further.

Removing Gaussian smoothing had a large visible effect, because the resized ERP image then kept fine trial-to-trial noise that often hid the pattern of interest. In this experiment, and all of the others, the patterns became barely or not at all recognisable without smoothing. Furthermore the model performance often dropped drastically, close to a class bias. We also tried to replace the Gaussian kernel, for example with the median kernels. These looked visually very similar and needed more computational time, so we kept Gaussian smoothing as a mandatory step in the ERP image processing.

Amplitude binning behaved in a related way, where two to four levels removed the pattern while sixteen or more levels approached the continuous image colour range. We inspected them visually and then discarded this idea, so the ERP images kept their continuous amplitudes. The idea was to reduce noise by limiting the amplitude range.

Z-scoring before or after the resize mainly changed the colour range and left the visible structure almost unchanged. We decided to z-score before the smoothing and resize, so that unwanted vertical bands are removed from the ERP image as early as possible.

As @tab:processing-step-results shows, the reference pipeline of sort, z-score, smoothing, and resize already reached almost the same performance as the tested variants. Clipping the amplitudes to the 1st and 99th percentile as in the plotting, scaling the whole ERP image trial amplitudes to a fixed range from -1 to 1, and enforcing an exact class balance each gave hardly any performance gain.

#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (2fr, 0.85fr, 0.85fr, 0.9fr, 0.9fr),
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
        [Processing step],
        [1-layer CNN],
        [3-layer CNN],
        [10-layer CNN],
        [ResNet18],
      ),

      [Reference pipeline],
      [0.412],
      [0.412],
      [0.510],
      [0.857],

      [Per-image clip at 1st and 99th percentile],
      [0.412],
      [0.412],
      [0.658],
      [0.865],

      [Per-image scale to range -1 to 1],
      [0.412],
      [0.412],
      [0.569],
      [0.865],

      [Class balancing to 50/50],
      [0.494],
      [0.479],
      [0.515],
      [0.896],
    ),
    caption: [Macro-F1 by processing step and model input size at 64x64 on the reference fixations dataset under five-fold cross-validation. The reference pipeline is sort, z-score, Gaussian smoothing, and resize.],
  ) <tab:processing-step-results>
]

=== Results of E8 (noise reduction and morphological filtering)
The broad filter screen of 40 filters adds each filter as a second input channel next to the standard ERP image channel and finds only small differences in model performance gain. Total-variation denoising, the Laplacian, and a coarse difference-of-Gaussian rank highest, but all of them stay within a few points of the single-channel baseline without any extra filters. So the extra filter channel alone does not decide the outcome. 

Hence we moved back to the single-channel combination order with Gaussian smoothing. As @tab:filter-combination-order shows, the filters gave no notable performance gain for the pretrained ResNet18 over the Gaussian-only reference, as in the previous experiment. We therefore did not pursue these filtering approaches further after E8.

#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.7fr, 0.95fr, 0.95fr, 0.95fr, 0.95fr),
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
        [Filter pipeline],
        [Random init BAcc],
        [Pretrained BAcc],
        [Random init Macro-F1],
        [Pretrained Macro-F1],
      ),

      [Gaussian only (reference)],
      [0.828],
      [0.904],
      [0.822],
      [0.898],

      [Gaussian then filter],
      [0.872],
      [0.907],
      [0.874],
      [0.903],

      [Filter then Gaussian],
      [0.868],
      [0.906],
      [0.861],
      [0.906],

      [Filter only],
      [0.875],
      [0.828],
      [0.881],
      [0.823],
    ),
    caption: [Balanced accuracy and macro-F1 for the best of the eight morphological filters per pipeline on the reference fixations dataset, for a randomly initialised and a pretrained ResNet18.],
  ) <tab:filter-combination-order>
]

=== Results of E9 (image and ERP-specific augmentation with imbalance handling)
This experiment was one of the early to explore methods for augmentation strategies on real ERP images. Generic image augmentations did not preserve the ERP image label well. A small rotation shears the time and trial axes and leaves empty corners, and a crop drops part of a pattern, so both can move or remove the very pattern that defines the class. We therefore compared label-preserving ERP image specific augmentations, namely trial dropout which removes random trials before the image is built, pink-noise addition, time jitter which shifts trials slightly along the time axis, and a combination of these. As a separate branch we tested the imbalance strategies from @exp:augmentation without any augmentation, namely class-weighted cross-entropy, focal loss, and balanced batches.

None of these methods gave a notable performance gain over the reference baseline. The imbalance strategies even lowered the scores and drove them towards the class bias, where the model mostly predicts the majority class. We therefore did not pursue these methods further. We also considered self-supervised and semi-supervised learning to exploit the large pool of unlabelled ERP images, but discarded this direction for feasibility reasons.

=== Results of E10 (input resolution and model capacity)
@tab:resolution-results reports macro-F1 scores across resolutions. The 1- and 3-layer CNNs stay at the majority-class at every resolution, so extra visual information did not help a model that is too shallow. The 10-layer CNN improves steadily with resolution. The pretrained ResNet18 is the only model that separates the classes sufficient enough.

Model training time grows as fast with increased resolution. Each step from 16x16 up to 256x256 quadruples the pixel count, and the training time follows a similar steep, roughly exponential growth, most clearly for the deeper models. The 10-layer CNN needs about 5 s per fold at 64x64, about 12 s at 128x128, and about 49 s at 256x256, while the pretrained ResNet18 grows from about 3 s to about 4 s to about 16 s over the same steps. This added cost buys only a negligible accuracy gain. The 64x64 resolution therefore gives ResNet18 most of its accuracy at a fraction of the runtime, which supports the 64x64 default used mostly in this thesis.

#[
  #show figure: set block(breakable: false)
  #set text(size: 8pt)
  #set par(justify: false)

  #figure(
    table(
      columns: (1.2fr, 0.85fr, 0.85fr, 0.9fr, 0.9fr),
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
        [Resolution],
        [1-layer CNN],
        [3-layer CNN],
        [10-layer CNN],
        [ResNet18],
      ),

      [16x16],
      [0.412],
      [0.412],
      [-],
      [-],

      [32x32],
      [0.412],
      [0.412],
      [-],
      [-],

      [64x64],
      [0.412],
      [0.412],
      [0.510],
      [0.857],

      [128x128],
      [0.412],
      [0.412],
      [0.551],
      [0.897],

      [256x256],
      [0.412],
      [0.412],
      [0.663],
      [0.848],
    ),
    caption: [Macro-F1 by input resolution and model on the reference fixations dataset under five-fold cross-validation. A dash marks a model and resolution combination that was not trained because the deeper models need a minimum input size.],
  ) <tab:resolution-results>
]

== Further Findings <sec:further-findings>
A side observation across the simulation and real experiments is that every evaluated CNN, even the smaller ones, could recognise the source resolution of an ERP image. An image that was downscaled to the model input size from its original high resolution was reliably told apart from an image that was downscaled to the same input size from a lower resolution, mostly from trial slicing. As a consequence the models learned a class bias per recognised source resolution. This mainly motivated and made the uniform augmentation necessary @sec:real-to-real-augmentation.

A related attempt concerned the different sampling rates across the data sources. We looked for an augmentation strategy that could bring recordings with different sampling rates onto a common time axis. One idea was time-point dropping, where single columns are removed from the ERP image to shorten the time axis. We discarded this idea after testing, because the dropped columns left visible ridges. Such ridges are unnatural for an EEG signal, since they create disconnected jumps along the time axis, and a CNN could pick them up as another shortcut feature.

Another practical finding is that the classifier is efficient to train on a desktop PC with one GPU and no specialised or cluster hardware. Training a ResNet18 for one cross-validation fold took from a few seconds to under a minute, depending on the resolution and the pool size.

We also tried to move the ERP image processing onto the GPU, in particular the Gaussian smoothing, which is a typical GPU task. This works, but it needs batch processing where each batch is copied into GPU memory, processed, and copied back to RAM, and although the processing itself is fast, these transfers cost time. Because such a batched pipeline is both effortful to implement and needs careful optimisation to give a noticeable speed-up, we tried it but did not pursue it further.

// ----------------------------------------------------------------------------
// Chapter 5 - Interpret the findings, state limitations honestly, and derive
// realistic next steps.
// ----------------------------------------------------------------------------
#pagebreak()
= Discussion <chp:discussion>

This chapter interprets the results from @chp:results. First we revisit each hypothesis from @chp:introduction and decide whether the evidence supports it. Finally, we then state the limitations and outline the future work.

== Interpretation of the Main Findings

_#link(<rq:sim-to-real-sigmoid>)[RQ 1.1] — recognising simulated sigmoids in real data._ The results support both hypotheses behind this question. For H1.1a, a ResNet18 trained only on simulated images reaches 0.926 balanced accuracy on the reference fixation dataset (@tab:simulation-resnet18-64-results), far above the 0.5 chance and majority-class level, and the remaining simulation experiments point the same way. Therefore, the simulator renders sigmoids realistic enough for the learned concept to carry over to a real recording. For H1.1b, the same models that score almost perfectly on synthetic data (@tab:simulation-calibration-results) drop on every real source (@tab:cross-source-sim-to-real-performance), which indicates the predicted sim-to-real gap. Thus the transfer is strong on the calibration source but only partial once the model meets other datasets.

_#link(<rq:simulator-calibration>)[RQ 1.2] — can better calibration close the gap?_ Here the results do not support H1.2a. The structured strategies, Latin hypercube and Monte Carlo sampling, did not beat plain broad-random sampling: broad random gave the best balanced accuracy in the 64x64 search and kept that lead on the real sources, where the Monte Carlo model even collapsed to a single predicted class (@tab:simulation-resnet18-64-results, @tab:cross-source-sim-to-real-performance). This matches random search as a strong baseline in high-dimensional spaces @Bergstra2012. Calibration narrows the gap but does not close it. The likely reason is structural rather than a tuning failure: a single real dataset already generalises poorly to the others, so a simulator calibrated to that one source inherits the same limit. One parameter set and one general synthetic dataset cannot span the full range of real ERP images.

_#link(<rq:real-preprocessing>)[RQ 2.1] — preprocessing choices._ The results support H2.1. The steps and their order do matter @Roy2019, and Gaussian smoothing is the decisive one: without it the patterns fall below visual recognition and the model collapses towards a class bias (@tab:processing-step-results). The many additional noise-reduction and morphological filters add nothing measurable over the reference pipeline (@tab:filter-combination-order). The practical reading is to keep preprocessing minimal and let the convolutional layers carry out the pattern detection @Isensee2021nnUNet.

_#link(<rq:real-augmentation>)[RQ 2.2] — augmentation and class imbalance._ The results support H2.2 only in part. Pattern-preserving augmentation clearly helps, as trial slicing with inverse sort and polarity lifts the pretrained ResNet18 from 0.866 to 0.918 balanced accuracy (@tab:real-data-classification-results). This echoes the small-data augmentation gains of Iqbal et al. @Iqbal2026BrainRetinaNet. The imbalance-handling half of the hypothesis does not hold: the loss-based strategies pushed the model towards a class bias, and the ERP-specific variants of trial dropout, pink noise, and time jitter added no gain (@exp:augmentation). Therefore, augmentation helps only when it leaves the defining pattern intact.

_#link(<rq:real-model-choice>)[RQ 2.3] — model and resolution._ The results support H2.3. The accuracy-efficiency trade-off is pronounced: the 1- and 3-layer CNNs stay at the majority class, the 10-layer CNN improves only marginally, and only the pretrained ResNet18 separates the classes well (@tab:resolution-results). ResNet34 matches it at a higher cost, and accuracy barely moves above 64x64, so the pretrained ResNet18 at 64x64 is the preferred configuration.

Taken together, the two research questions, #link(<rq:sim-to-real>)[RQ 1] and #link(<rq:manual-labelling>)[RQ 2]: A simulator-trained CNN detects sigmoids well on the source it was calibrated to but generalises weakly across datasets, and better calibration does not remove that gap. Training on manually labelled real images instead yields stronger and more robust performance across the whole pool, reaching 0.918 balanced accuracy. Efficiency points the same way: the classifier is cheap to train and run on a single GPU, while image simulation dominates the runtime (@tab:simulation-resnet18-64-timing). For now, real labels give the more reliable detector, and simulation is a partial, not yet sufficient, substitute for the manual labelling effort.

== Limitations
The main limitation of this thesis is the sim-to-real gap. The simulator can generate labelled ERP images in large numbers, but it can only generate the kind of variability that was built into it. Real EEG also contains subject-specific responses, non-stationary noise, artefacts, imperfect event timing, and preprocessing effects. This means that strong performance on synthetic images is not enough to show that the model has generalised sufficiently for patterns from real ERP images @Krol2018SEREEGA, @Schepers2025. Therefore, the weak transfer result should be read as a genuine limitation of the setup, not just as a tuning problem.

A related risk is shortcut learning. CNNs often exploit whichever visual regularity is easiest for the training objective, even if that regularity is not the intended concept @Geirhos2020ShortcutLearning. In this project, such shortcuts could come from simulator-specific smoothness, noise texture, colour scaling, or unusually clean pattern boundaries. Domain randomisation can lower this risk, since varying the simulator parameters widely keeps the model from relying on any single simulator cue, but it cannot remove the risk completely @Tobin2017. The resize step is another example, where every tested CNN can tell apart ERP images by their source resolution, as @sec:further-findings reports.

The manual labeling is another limitation. The real-data evaluation uses only a limited number of manually labelled ERP images, and visual pattern labels are not objective. Borderline cases can be judged differently, for instance when a weak and noisy sigmoid appears. Reliability measures are normally used to quantify such disagreement when several raters label the same images @Artstein2008InterCoderAgreement, @Hallgren2012InterRaterKappa. Without that type of agreement analysis, disagreement between the classifier and the labels cannot always be interpreted as model error alone.

The results are also conditional on the chosen preprocessing pipeline. Sorting, z-scoring, smoothing, and resizing all change the image seen by the CNN. This is not a minor technical detail. ERP methods research shows that reasonable processing choices can lead to different measurements and conclusions @Clayson2021ERPMultiverse.

A further limitation is that the different sampling rates across the data sources were not handled. The sources range from 250 to 512 Hz. We tried to equalise the time axis with time-point dropping, but discarded it because it left unnatural ridges, as @sec:further-findings describes. In our pool the sampling-rate differences were less pronounced than the trial-count differences, and the trial-slicing augmentation already equalises that issue.

== Future Work
One useful next step is to try non-CNN architectures for the same classification task. Transformer-based image models have gained popularity in recent years and classify images through self-attention rather than the local convolutions a CNN relies on, which lets them relate distant regions of an image directly @Dosovitskiy2021ViT. So attention across the trial and time axes in an ERP image is a plausible alternative to convolution. A first comparison could pair the pretrained ResNet18 with a Vision Transformer. Attention models for EEG and ERP decoding already show that self-attention over channels and time can match convolutional models @Song2023EEGConformer, @Zelger2025BeyondAveraging.

A further direction is to make the simulation faster. Generating images for a single ERP pattern is already an efficiency concern with the current pipeline. Covering all six patterns with the current implementation would take roughly a week of runtime, which is not practical. Better use of multithreading is the most obvious lever. We only tuned the sigmoid in this thesis, so the other five patterns may transfer better under their own settings.

A third extension is localisation. The current classifier assigns one label to an entire ERP image, so it cannot mark where a pattern starts, ends, or overlaps with another structure. Detection-style biomedical imaging work such as Brain-RetinaNet shows how convolutional models can move from image-level classification towards localising relevant regions @Iqbal2026BrainRetinaNet. For ERP images, such a shift would require labels for pattern extents in trial-time space, not only image-level labels. This may also be beneficial for the vision transformer models.

A next step is to embed the trained CNN into an interactive tool for exploratory analysis, for example _ERPgnostics.jl_ @ERPgnosticsDocs, where its class probabilities act as the pattern measure for exploring new ERP datasets. Combined with the presented augmentation, the model could screen many subjects, channels, and sorting variables and surface the few images worth a closer look.

// ----------------------------------------------------------------------------
// Chapter 6 - End with direct answers, not a second discussion.
// ----------------------------------------------------------------------------
#pagebreak()
= Conclusion <chp:conclusion>

// Summarize the core takeaway in a few sentences and answer the research
// question directly.
This thesis addressed two questions, whether convolutional neural networks can detect visual ERP image patterns automatically, and whether simulated ERP images can reduce the need for manually labelled real data.

Regarding the first question, the results provide partial support for the simulation-based approach. A ResNet18 trained only on simulated sigmoid images reaches 0.926 balanced accuracy on the reference fixation dataset, well above 0.5 chance. Simulator calibration reduced the observed sim-to-real gap, although it did not eliminate it entirely. In our settings, broad random parameter sampling surprisingly transfered more reliably than the structured strategies we tested. A simulator-trained model is therefore a useful starting point, but it does not yet replace manual labels across datasets.

With respect to the second research question, the results suggest that training on manually labelled real ERP images constitutes an effective and efficient approach. A pretrained ResNet18 reaches 0.918 balanced accuracy on the labelled pool, performance improved most with Gaussian smoothing and by pattern-preserving augmentation such as trial slicing with inverse sort and polarity. At 64x64 the model keeps most of this accuracy at a fraction of the training cost, so the patterns are detectable both reliably and with modest computational requirements.

Taken together, CNNs were able to detect annotated ERP image patterns in the
evaluated datasets. Training on manually labelled real images gave the stronger
and more robust detector, so manual labels still outperform the simulation
approach. Simulation can nonetheless cut the labelling effort, yet closing the sim-to-real gap for all six patterns remains an open problem. A faster simulator with parameters tuned per pattern is a plausible next step.
