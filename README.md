# crxnfp_sample_prep

## Requirements

The required Python libraries and versions can be found in requirement.txt

## Prepare for the training data of crxnfp

1. Calculate the DRFP fingerprints for 3M USPTO reactions in a multi-process way (multi-process-text.py)
2. Sample qualified reaction pairs (sample-pairs-bin.py)
3. Write qualified reaction pairs to a file (prepare-training-data.py)
4. Visualize the distribution of tanimoto similarity of sampled reaction pairs (load-pair-view.py)

## Data

https://zenodo.org/records/16996192<img width="468" height="14" alt="image" src="https://github.com/user-attachments/assets/39eedfb0-14d8-42aa-9916-9b0eac7589f3" />

uspto_all_reactions_training.txt:  USPTO reactions 

radius_3_text.zip:  DRFP fingerprints for all USPTO reactions

train-srxnfp-data.txt:  Sampled 1M reaction pairs

