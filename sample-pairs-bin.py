# The script to sample qualified reaction paris for each bin. Tanimoto similarity of drfp fingerprints of two reactions.
# Bins: [0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 1.0]

import glob
import random
import os
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math


# Function to generate unique, order-independent index pairs
def generate_unique_index_pairs(num_items, num_pairs, previously_sampled):
    unique_pairs = []
    while len(unique_pairs) < num_pairs:
        # Randomly sample two indices
        i, j = random.sample(range(num_items), 2)
        # Create an order-independent pair (sorted tuple)
        pair = tuple(sorted((i, j)))
        # Add to the unique_pairs set if not previously sampled
        if pair not in previously_sampled:
            unique_pairs.append(pair)
            # previously_sampled.add(pair)
    return unique_pairs


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
            # return math.sqrt(original_tanimoto)
            k = 5  # Adjust compression factor
            return 1 - np.exp(-k * original_tanimoto)

    except Exception as e:
        print(f"Error calculating similarity for pair {pair}: {e}")
        return 0.0


if __name__ == "__main__":

    directory = "radius_3_text/"
    files = glob.glob(directory + '*.text')

    sorted_files = sorted(files, key=lambda x: int(x.split("chunk_")[1].split(".text")[0]))

    rxn_file = open("uspto_all_reactions_training.txt", "r")
    rxn_smiles = [line.strip() for line in rxn_file]

    total_rxn_counts = len(rxn_smiles)
    pairs_count = 1000000

    previously_sampled_stored = set()
	
	# Starting from the second bin, uncomment the following two lines
    # with open("stored_pairs.pkl", "rb") as file:
    #     previously_sampled_stored = pickle.load(file)

    pairs = generate_unique_index_pairs(total_rxn_counts, pairs_count, previously_sampled_stored)

    fps_saved = []
    index = 0
    for file_name in sorted_files:
        print("processing file", index)
        file = open(file_name, "r", encoding='utf-8')
        fps_saved_tmp = [line.strip() for line in file]
        fps_saved += fps_saved_tmp
        index += 1

    fps_paried = []

    for pair in pairs:
        fps_paried.append((fps_saved[pair[0]], fps_saved[pair[1]]))

    print(len(fps_paried))

    pool_size = os.cpu_count()
    print("pool size:", pool_size)

    pair_similarity = []
    pool = multiprocessing.Pool(pool_size)

    for result in tqdm(pool.imap(func=tanimoto_similarity, iterable=fps_paried), total=len(fps_paried)):
        pair_similarity.append(result)

    pool.close()
    pool.join()

    print(len(pair_similarity))
    print(np.max(pair_similarity))

    bin = [0, 0.1]

    selected_index = [i for i in range(len(pair_similarity)) if bin[0] <= pair_similarity[i] <= bin[1]]

    bin_count = np.array([0])
    bin_remain = 1000000-np.sum(bin_count)

    if len(selected_index) >= bin_remain:
        selected_index = selected_index[:bin_remain]

    selected_pairs = [pairs[i] for i in selected_index]
    selected_similarity = [pair_similarity[i] for i in selected_index]

    previously_sampled_stored.update(selected_pairs)

    print("Remained pairs:", len(selected_pairs))

    with open("stored_pairs.pkl", "wb") as file:
        pickle.dump(previously_sampled_stored, file)

    print("stored samples:", len(previously_sampled_stored))

