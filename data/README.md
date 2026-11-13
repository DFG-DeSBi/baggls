This directory contains datasets, trained models, and XAI outputs required for BaGGLS experiments and applications.

# Structure

data/
├── JASPAR_core_latest_version.txt 
├── testdata/
│   ├── testdata.npy         
│   ├── testlabels.npy
│   └── testpos.npy
├── vb_models/
│   └── vb_model.pkl
└── xai/
    └── attr_dm_0.0001.csv

# File Descriptions
- **JASPAR_core_latest_version.txt**: Transcription factor binding motifs for FIMO scanning incl split REST motif
- **testdata.npy**: Test sequences for model evaluation
- **testlabels.npy**: Binary response variable
- **testpos.npy**: Motif positions
- **vb_model.pkl**: Fitted BaGGLS model
- **attr_dm_0.0001.csv**: Aggregated attribution scores

Configure paths in `configs/paths.yml`.
