# crxnfp_sample_prep

## Prepare for the training data of crxnfp

1. Calculate the DRFP fingerprints for 3M USPTO reactions in a multi-process way (multi-process-text.py)
2. Sample qualified reaction pairs (sample-pairs-bin.py)
3. Write qualified reaction pairs to a file (prepare-training-data.py)
4. Visualize the distribution of tanimoto similarity of sampled reaction pairs (load-pair-view.py)

## Data

https://drive.switch.ch/index.php/s/10kLm31tdnailZS

uspto_all_reactions_training.txt:  USPTO reactions 

radius_3_text.zip:  DRFP fingerprints for all USPTO reactions

train-srxnfp-data.txt:  Sampled 1M reaction pairs

