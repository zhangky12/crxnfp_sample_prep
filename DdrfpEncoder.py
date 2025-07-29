# This script is adapted from DrfpEncoder, so that the fingerprints will be directed. 
# It's called directed DrfpEncoder (DdrfpEncoder)

from drfp.fingerprint import DrfpEncoder, NoReactionError
from typing import Iterable, List, Tuple, Set, Dict, Union
import numpy as np
from rdkit.Chem import AllChem
from collections import defaultdict
from tqdm import tqdm


class DdrfpEncoder(DrfpEncoder):

    def internal_encode(
            in_smiles: str,
            radius: int = 3,
            min_radius: int = 0,
            rings: bool = True,
            get_atom_indices: bool = False,
            root_central_atom: bool = True,
            include_hydrogens: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, List[Set[int]]]]]],
    ]:
        """Creates an drfp array from a reaction SMILES string.

        Arguments:
            in_smiles: A valid reaction SMILES string
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            min_radius: The minimum radius that is used to extract n-grams
            rings: Whether or not to include rings in the shingling

        Returns:
            A tuple with two arrays, the first containing the drfp hash values for substrates and products, the second the substructure SMILES
        """

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
                sh, ai = DrfpEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                atom_indices["reactants"].append(ai)
            else:
                sh = DrfpEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            for s in sh:
                right_shingles.add(s)

        for r in right:
            mol = AllChem.MolFromSmiles(r)

            if not mol:
                atom_indices["products"].append(None)
                continue

            if get_atom_indices:
                sh, ai = DrfpEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                atom_indices["products"].append(ai)
            else:
                sh = DrfpEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            for s in sh:
                left_shingles.add(s)
		
		# Distinguish forward and reverse reactions
        right_unique = right_shingles.difference(left_shingles)
        left_unique = left_shingles.difference(right_shingles)


        if get_atom_indices:
            return (DrfpEncoder.hash(list(right_unique)), DrfpEncoder.hash(list(left_unique))), list(
                right_unique.union(left_unique)), atom_indices
        else:
            return (DrfpEncoder.hash(list(right_unique)), DrfpEncoder.hash(list(left_unique))), list(
                right_unique.union(left_unique))

    @staticmethod
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

    @staticmethod
    def encode(
            X: Union[Iterable, str],
            n_folded_length: int = 2048,
            min_radius: int = 0,
            radius: int = 3,
            rings: bool = True,
            mapping: bool = False,
            atom_index_mapping: bool = False,
            root_central_atom: bool = True,
            include_hydrogens: bool = False,
            show_progress_bar: bool = False,
    ) -> Union[
        List[np.ndarray],
        Tuple[List[np.ndarray], Dict[int, Set[str]]],
        Tuple[List[np.ndarray], Dict[int, Set[str]]],
        List[Dict[str, List[Dict[str, List[Set[int]]]]]],
    ]:
        """Encodes a list of reaction SMILES using the drfp fingerprint.

        Args:
            X: An iterable (e.g. List) of reaction SMILES or a single reaction SMILES to be encoded
            n_folded_length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            radius: The maximum radius of a substructure
            rings: Whether to include full rings as substructures
            mapping: Return a feature to substructure mapping in addition to the fingerprints
            atom_index_mapping: Return the atom indices of mapped substructures for each reaction
            root_central_atom: Whether to root the central atom of substructures when generating SMILES
            show_progress_bar: Whether to show a progress bar when encoding reactions

        Returns:
            A list of drfp fingerprints or, if mapping is enabled, a tuple containing a list of drfp fingerprints and a mapping dict.
        """
        if isinstance(X, str):
            X = [X]

        show_progress_bar = not show_progress_bar

        # If mapping is required for atom_index_mapping
        if atom_index_mapping:
            mapping = True

        result = []
        result_map = defaultdict(set)
        atom_index_maps = []

        for _, x in tqdm(enumerate(X), total=len(X), disable=show_progress_bar):
            if atom_index_mapping:
                hashed_diff, smiles_diff, atom_index_map = DrfpEncoder.internal_encode(
                    x,
                    min_radius=min_radius,
                    radius=radius,
                    rings=rings,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
            else:
                hashed_diff, smiles_diff = DdrfpEncoder.internal_encode(
                    x,
                    min_radius=min_radius,
                    radius=radius,
                    rings=rings,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            difference_folded, on_bits = DdrfpEncoder.fold(
                hashed_diff,
                length=n_folded_length,
            )

            if mapping:
                for unfolded_index, folded_index in enumerate(on_bits):
                    result_map[folded_index].add(
                        smiles_diff[unfolded_index].decode("utf-8")
                    )

            if atom_index_mapping:
                aidx_bit_map = {}
                aidx_bit_map["reactants"] = []
                aidx_bit_map["products"] = []

                for reactant in atom_index_map["reactants"]:
                    r = defaultdict(list)
                    for key, value in reactant.items():
                        if key in smiles_diff:
                            idx = smiles_diff.index(key)
                            r[on_bits[idx]].append(value)
                    aidx_bit_map["reactants"].append(r)

                for product in atom_index_map["products"]:
                    r = defaultdict(list)
                    for key, value in product.items():
                        if key in smiles_diff:
                            idx = smiles_diff.index(key)
                            r[on_bits[idx]].append(value)
                    aidx_bit_map["products"].append(r)

                atom_index_maps.append(aidx_bit_map)

            # result.append(difference_folded)
            # return condensed result
            condensed_difference_folded = set(np.argwhere(difference_folded != 0).flatten())
            result.append(condensed_difference_folded)

        r = [result]

        if mapping:
            r.append(result_map)

        if atom_index_mapping:
            r.append(atom_index_maps)

        if len(r) == 1:
            return r[0]
        else:
            return tuple(r)
