# Create the training reaction pairs

import pickle
import glob
import os
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import numpy as np


def tanimoto_similarity(pair):
    try:
        fp1_condense, fp2_condense = pair
        if fp1_condense == "":
            fp1_condense = set()
        else:
            fp1_condense = set(fp1_condense.split(","))

        if fp2_condense == "":
            fp2_condense = set()
        else:
            fp2_condense = set(fp2_condense.split(","))

        if len(fp1_condense | fp2_condense) == 0:
            return 0
        else:
            original_tanimoto = len(fp1_condense & fp2_condense) / len(fp1_condense | fp2_condense)
            k = 5  # Adjust compression factor
            return 1 - np.exp(-k * original_tanimoto)

    except Exception as e:
        print(f"Error calculating similarity for pair {pair}: {e}")
        return 0.0


if __name__ == "__main__":

    with open("stored_pairs.pkl", "rb") as file:
        previously_sampled_stored = pickle.load(file)

    rxn_file = open("uspto_all_reactions_training.txt", "r")
    rxn_smiles = [line.strip() for line in rxn_file]

    directory = "radius_3_text/"
    files = glob.glob(directory + '*.text')

    sorted_files = sorted(files, key=lambda x: int(x.split("chunk_")[1].split(".text")[0]))

    fps_saved = []
    index = 0
    for file_name in sorted_files:
        print("processing file", index)
        file = open(file_name, "r", encoding='utf-8')
        fps_saved_tmp = [line.strip() for line in file]
        fps_saved += fps_saved_tmp
        index += 1

    fps_paired = []
    reactions_paired = []

    for pair in list(previously_sampled_stored):
        fps_paired.append((fps_saved[pair[0]], fps_saved[pair[1]]))
        reactions_paired.append((rxn_smiles[pair[0]], rxn_smiles[pair[1]]))

    print(len(fps_paired))

    pool_size = os.cpu_count()
    print("pool size:", pool_size)

    pair_similarity = []
    pool = multiprocessing.Pool(pool_size)

    for result in tqdm(pool.imap(func=tanimoto_similarity, iterable=fps_paired), total=len(fps_paired)):
        pair_similarity.append(result)

    pool.close()
    pool.join()

    print(len(pair_similarity))

    print(np.max(pair_similarity))

    assert len(reactions_paired) == len(pair_similarity)

    with open("train-srxnfp-data.txt", "w") as file:
        for i in range(len(pair_similarity)):
            line = reactions_paired[i][0] + "," + reactions_paired[i][1] + "," + str(pair_similarity[i]) + "\n"
            file.write(line)