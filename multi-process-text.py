import multiprocessing
from drfp.fingerprint import NoReactionError
from typing import Iterable, List, Tuple, Set, Dict, Union
import numpy as np
from rdkit.Chem import AllChem
from collections import defaultdict
from tqdm import tqdm
from hashlib import blake2b
import os
import pickle
import warnings
warnings.filterwarnings("ignore")


def shingling_from_mol(
        in_mol
):
    min_radius = 0
    radius = 3
    rings = True

    root_central_atom = True
    include_hydrogens = False

    get_atom_indices = False

    if include_hydrogens:
        in_mol = AllChem.AddHs(in_mol)

    shingling = []
    atom_indices = defaultdict(list)

    if rings:
        for ring in AllChem.GetSymmSSSR(in_mol):
            bonds = set()
            ring = list(ring)
            indices = set()
            for i in ring:
                for j in ring:
                    if i != j:
                        indices.add(i)
                        indices.add(j)
                        bond = in_mol.GetBondBetweenAtoms(i, j)
                        if bond is not None:
                            bonds.add(bond.GetIdx())

            ngram = AllChem.MolToSmiles(
                AllChem.PathToSubmol(in_mol, list(bonds)),
                canonical=True,
                allHsExplicit=True,
            ).encode("utf-8")

            shingling.append(ngram)

            if get_atom_indices:
                atom_indices[ngram].append(indices)

    if min_radius == 0:
        for i, atom in enumerate(in_mol.GetAtoms()):
            ngram = atom.GetSmarts().encode("utf-8")
            shingling.append(ngram)

            if get_atom_indices:
                atom_indices[ngram].append(set([atom.GetIdx()]))

    for index, _ in enumerate(in_mol.GetAtoms()):
        for i in range(1, radius + 1):
            p = AllChem.FindAtomEnvironmentOfRadiusN(
                in_mol, i, index, useHs=include_hydrogens
            )
            amap = {}
            submol = AllChem.PathToSubmol(in_mol, p, atomMap=amap)

            if index not in amap:
                continue

            smiles = ""

            if root_central_atom:
                smiles = AllChem.MolToSmiles(
                    submol,
                    rootedAtAtom=amap[index],
                    canonical=True,
                    allHsExplicit=True,
                )
            else:
                smiles = AllChem.MolToSmiles(
                    submol,
                    canonical=True,
                    allHsExplicit=True,
                )

            if smiles != "":
                shingling.append(smiles.encode("utf-8"))
                if get_atom_indices:
                    atom_indices[smiles.encode("utf-8")].append(set(amap.keys()))

    if not root_central_atom:
        for key in atom_indices:
            atom_indices[key] = list(set([frozenset(s) for s in atom_indices[key]]))

    # Set ensures that the same shingle is not hashed multiple times
    # (which would not change the hash, since there would be no new minima)
    if get_atom_indices:
        return list(set(shingling)), atom_indices
    else:
        return list(set(shingling))


def internal_encode(
        in_smiles
):
    get_atom_indices = False

    atom_indices = {}
    atom_indices["reactants"] = []
    atom_indices["products"] = []

    sides = in_smiles.split(">")
    if len(sides) < 3:
        raise NoReactionError(
            f"The following is not a valid reaction SMILES: '{in_smiles}'"
        )

    if len(sides[1]) > 0:
        sides[0] += "." + sides[1]

    left = sides[0].split(".")
    right = sides[2].split(".")

    left_shingles = set()
    right_shingles = set()

    for l in left:
        mol = AllChem.MolFromSmiles(l)

        if not mol:
            atom_indices["reactants"].append(None)
            continue

        if get_atom_indices:
            sh, ai = shingling_from_mol(
                mol
            )
            atom_indices["reactants"].append(ai)
        else:
            sh = shingling_from_mol(
                mol
            )

        for s in sh:
            right_shingles.add(s)

    for r in right:
        mol = AllChem.MolFromSmiles(r)

        if not mol:
            atom_indices["products"].append(None)
            continue

        if get_atom_indices:
            sh, ai = shingling_from_mol(
                mol
            )
            atom_indices["products"].append(ai)
        else:
            sh = shingling_from_mol(
                mol
            )

        for s in sh:
            left_shingles.add(s)

    # s = right_shingles.symmetric_difference(left_shingles)
    right_unique = right_shingles.difference(left_shingles)
    left_unique = left_shingles.difference(right_shingles)

    # if get_atom_indices:
    #     return DrfpEncoder.hash(list(s)), list(s), atom_indices
    # else:
    #     return DrfpEncoder.hash(list(s)), list(s)

    if get_atom_indices:
        return (hash(list(right_unique)), hash(list(left_unique))), list(
            right_unique.union(left_unique)), atom_indices
    else:
        return (hash(list(right_unique)), hash(list(left_unique))), list(
            right_unique.union(left_unique))


def hash(shingling: List[str]) -> np.ndarray:
    """Directly hash all the SMILES in a shingling to a 32-bit integerself.

    Arguments:
        shingling: A list of n-grams

    Returns:
        A list of hashed n-grams
    """

    hash_values = []

    for t in shingling:
        hash_values.append(int(blake2b(t, digest_size=4).hexdigest(), 16))

    return np.array(hash_values, dtype=np.int32)


def fold(
        hash_values: Tuple[np.ndarray, np.ndarray], length: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """Folds the hash values to a binary vector of a given length.

    Arguments:
        hash_value: An array containing the hash values
        length: The length of the folded fingerprint

    Returns:
        A tuple containing the folded fingerprint and the indices of the on bits
    """
    substrates_hash_values, products_hash_values = hash_values

    folded = np.zeros(2 * length, dtype=np.uint8)
    substrates_on_bits = substrates_hash_values % length
    products_on_bits = products_hash_values % length + length
    on_bits = np.concatenate((substrates_on_bits, products_on_bits), axis=0)
    folded[on_bits] = 1

    return folded, on_bits


def encode_single(x: str):

    n_folded_length = 2048

    hashed_diff, smiles_diff = internal_encode(
        x
    )

    difference_folded, on_bits = fold(
        hashed_diff,
        length=n_folded_length,
    )

    condensed_difference_folded = ",".join(str(s) for s in np.argwhere(difference_folded != 0).flatten().tolist())

    return condensed_difference_folded


if __name__ == "__main__":

    not_show_progress_bar = False,
    pool_size = os.cpu_count()
    print("pool size:", pool_size)

    # Load data from the file
    file_path = "uspto_all_reactions_training.txt"
    with open(file_path, 'r') as file:
        rxn_smiles = [line.strip() for line in file]
	
	# You can also update this part, so the program can be stopped and restarted
    chunk_id = 0
    item_count = len(rxn_smiles)
    start_id = 0

    while start_id < len(rxn_smiles):

        end_id = min(start_id + item_count, len(rxn_smiles))

        print("Processing", start_id, "-", end_id)

        X = rxn_smiles[start_id:end_id]
        fps = []

        pool = multiprocessing.Pool(pool_size)

        for result in tqdm(pool.imap(func=encode_single, iterable=X), total=len(X)):
            fps.append(result)

        pool.close()
        pool.join()

        with open("radius_3/ddrfp_radius_3_chunk_"+str(chunk_id)+".text", "w") as f:
            for line in fps:
                f.write(f"{line}\n")


        chunk_id += 1
        start_id += item_count
        fps = []

