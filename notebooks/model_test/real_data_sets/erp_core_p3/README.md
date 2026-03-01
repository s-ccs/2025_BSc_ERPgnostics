# ERP CORE P3 Dataset

This folder contains the processed ERP CORE P3 material used by
`notebooks/model_test/erp_core_p3.ipynb`.

## Contents

- `epochs.hdf5`: per-subject epoch tensors in `subjects/<sub>/epochs`
  with shape `(channel, time, trial)`
- `events_rt.csv`: one row per retained epoch with calculated reaction time
- `metadata.json`: source links and layout metadata
- `source/`: downloaded ERP CORE source files used to build the derived files

## Source

- OSF P3 component: https://osf.io/etdkz/
- OSF root listing used for downloads: https://files.osf.io/v1/resources/etdkz/providers/osfstorage/5f247351b084f60115c9aa10/
- ERP CORE P3 processing scripts: https://github.com/lucklab/ERP_CORE/tree/master/P3
- MNE EEGLAB epoch reader docs: https://mne.tools/stable/generated/mne.read_epochs_eeglab.html

## Notes

- Source epoch files are the processed EEGLAB files
  `*_P3_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set/.fdt`.
- Reaction times are calculated from `*_P3_Eventlist_For_RTs.txt`, which aligns
  directly with the retained epochs in the processed epoch file.
- The Julia notebook applies the remaining image pipeline steps:
  reaction-time sorting, per-timepoint z-scoring, Gaussian low-pass filtering,
  and resizing to the model input size.
