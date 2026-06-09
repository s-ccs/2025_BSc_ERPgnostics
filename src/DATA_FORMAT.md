# Data Format For Real ERP Data

This document describes how ERP data must be stored so it can be used by the
project code.

## Short Version

Each dataset is one JLD2 bundle:

```text
datasets/<dataset_key>/
├── events.jld2
├── labels.jld2
└── signals/
    ├── <channel_name>.jld2
    └── ...
```

Use one dataset folder per participant. Participants from the same study get
separate folders because their events differ.

## Stored Preprocessing State

The bundle stores preprocessed, event-locked EEG/ERP trials. It does not store
raw continuous EEG.

Before writing the bundle:

- unusable artefacts and segments have already been handled
- baseline handling has already been applied
- each trial has been cut to the post-event window
- all samples before the event onset have been removed
- each trial is aligned to the event onset

The bundle must not store ERP-image processing steps. Trial sorting,
per-timepoint z-scoring, smoothing, and resizing happen only after loading.

## signals/<channel_name>.jld2

Each file in signals/ stores the EEG data for one channel. It must contain
`data_time_trials` and `metadata`.

The file name is the channel name. Use the original EEG channel label when it is
available, for example Cz, Pz, Fp1, or ch001. The name must be unique inside the
signals/ folder.

### data_time_trials

data_time_trials is a Float32 matrix with EEG values in
timepoints x trials layout. E.g. if data_time_trials has size 501 x 346, the channel
contains 501 timepoints for 346 trials.

Required shape:

- number of rows equals n_timepoints_post in events.jld2 metadata
- number of columns equals the number of rows in events

The core alignment rule is:

```text
events row i  <->  data_time_trials[:, i]
```

Every trial gets its own trial_index. The index starts at 1 in each dataset
folder:

```text
trial_index = 1:nrow(events)
```

This means the stored bundle can be read in two equivalent ways:

```text
events row i  <->  data_time_trials[:, i]
events row i  <->  data_time_trials[:, events.trial_index[i]]
```

The row-position rule describes the physical storage layout. The trial_index
column keeps the original trial identity visible after events are sorted,
sliced, or augmented.

### metadata

metadata is a dictionary with channel-level metadata. These fields make each
signal file traceable and easier to assign to the correct dataset and source
channel, even when a channel file is inspected or moved independently.

Required fields:

- dataset_key: same dataset_key as in events.jld2 metadata
- channel_name: channel file name without .jld2
- channel_idx: original channel index in the source recording

## events.jld2

events.jld2 must contain `events` and `metadata`.

### events

events is a DataFrame with one row per trial.

Required columns:

- trial_index: trial number inside this dataset folder. It starts at 1.
- epoch_index: original epoch number before conversion. This helps trace a trial back to the source data

Example:

| trial_index | epoch_index | condition | fixation_duration_ms | rt_ms |
| ---: | ---: | --- | ---: | ---: |
| 1 | 1 | fixation | 186.0 | 531.0 |
| 2 | 2 | fixation | 224.0 | 487.0 |
| 3 | 3 | fixation | 171.0 | 612.0 |

Additional columns are allowed. They are used as sort variables, for example
condition, fixation_duration_ms, gaze_x, gaze_y, pupil size, reaction_time_ms,
word_frequency, or rt_ms.

### metadata

metadata is a dictionary with dataset-level metadata.

Required fields:

- dataset_key: unique dataset identifier; can be a combination of experiment and participant
- dataset_label: short readable dataset name for plots and tables
- subject_label: participant ID, for example sub-01
- n_trials: number of trials
- n_timepoints_post: number of timepoints post event onset
- time_start_s: start time of each stored trial in seconds, relative to the event onset, should be close to 0
- time_end_s: end time of each stored trial in seconds
- sampling_rate_hz: number of sampled timepoints per second

## labels.jld2

labels.jld2 is optional. Use it when labels are available.
If labels.jld2 exists, it must contain `labels` and `metadata`.

### labels

labels is a DataFrame with one row per labelled channel and sort variable. The
dataset_key comes from labels.jld2 metadata.

Required columns:

- channel_name: channel file name without .jld2. Labels use this name to find the signal file
- sort_variable: column name in events
- erp_class: ERP pattern class

Example:

| channel_name | sort_variable | erp_class |
| --- | --- | --- |
| ch001 | duration | no_class |
| ch002 | duration | sigmoid |
| ch003 | sac_amplitude | tilted_bar |

Allowed erp_class values:

- no_class
- sigmoid
- one_sided_fan
- two_sided_fan
- diverging_bar
- hourglass
- tilted_bar

Together with the dataset_key from metadata, each channel_name and sort_variable
combination should occur at most once.

### metadata

metadata is a dictionary with label-level metadata.

Required fields:

- dataset_key: same dataset_key as in events.jld2 metadata
