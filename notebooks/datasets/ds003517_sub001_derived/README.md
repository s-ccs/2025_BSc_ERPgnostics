# OpenNeuro 8bit EEG Dataset (Derived)

This folder contains a derived subject-level dataset built from OpenNeuro
`ds003517` for `sub-001`.

Sources
- OpenNeuro dataset DOI: 10.18112/openneuro.ds003517.v1.1.0
- Paper DOI: 10.1016/j.neuroimage.2016.02.075
- Public S3 export root: https://s3.amazonaws.com/openneuro.org/ds003517

Contents
- source dataset root: `/home/benjamin/Dokumente/BA2/notebooks/datasets/ds003517`
- `epochs.hdf5`: epoched EEG, written so Julia/HDF5 reads it as `(channels, time, trial)`
- `events.csv`: per-epoch metadata aligned to the HDF5 trials
- `metadata.json`: preprocessing summary and trial counts

Preprocessing
- drop `VEOG` and `HEOG`
- average reference
- band-pass filter `0.1` to `20.0` Hz
- epochs from `-0.5` s to `0.498` s around selected events
- baseline correction from `-0.2` s to `0.0` s
