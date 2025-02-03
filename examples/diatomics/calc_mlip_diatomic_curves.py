"""Predict pair-repulsion curves for diatomic molecules with MACE-MP.

Thanks to Tamas Stenczel who first did this type of PES smoothness and physicality
analysis in https://github.com/stenczelt/MACE-MP-work for the MACE-MP paper
https://arxiv.org/abs/2401.00096 (see fig. 56).
"""

# %%
from __future__ import annotations

import json
import lzma
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from mace.calculators import MACECalculator, mace_mp
from tqdm import tqdm


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from ase.calculators.calculator import Calculator

__date__ = "2024-03-31"
module_dir = os.path.dirname(__file__)


# %%
@contextmanager
def timer(label: str = "") -> Generator[None, None, None]:
    """Context manager for timing code execution."""
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label} took {time.perf_counter() - start:.3}s.")


def generate_diatomics(
    elem1: str, elem2: str, distances: Sequence[float]
) -> list[Atoms]:
    """Build diatomic molecules in vacuum for given distances.

    Args:
        elem1: Chemical symbol of the first element.
        elem2: Chemical symbol of the second element.
        distances: Distances to sample at.

    Returns:
        list[Atoms]: List of diatomic molecules.
    """
    return [
        Atoms(f"{elem1}{elem2}", positions=[[0, 0, 0], [dist, 0, 0]], pbc=False)
        for dist in distances
    ]


def calc_one_pair(
    elem1: str, elem2: str, calc: MACECalculator, distances: Sequence[float]
) -> list[float]:
    """Calculate potential energy for a pair of elements at given distances.

    Args:
        elem1: Chemical symbol of the first element.
        elem2: Chemical symbol of the second element.
        calc: MACECalculator instance.
        distances: Distances to calculate potential energy at.

    Returns:
        list[float]: Potential energies at each distance.
    """
    return [
        calc.get_potential_energy(at)
        for at in generate_diatomics(elem1, elem2, distances)
    ]


def calc_homo_diatomics(
    calculator: Calculator, model_name: str, atomic_numbers: Sequence[int]
) -> dict[str, list[float]]:
    """Generate potential energy data for homonuclear diatomic molecules.

    Args:
        calculator: ASE calculator instance.
        model_name: Name of the model for the output file.
        atomic_numbers: List of atomic numbers to calculate for.

    Returns:
        dict[str, list[float]]: Potential energy data for homonuclear diatomic
            molecules or None if the file already exists.
    """
    out_path = f"{module_dir}/homo-nuclear-{model_name}.json.xz"
    distances = np.linspace(0.1, 6.0, 119)
    # saving the results in a dict: "z0-z1" -> [energy] & saved the distances
    results = {"distances": list(distances)}
    # homo-nuclear diatomics
    pbar = tqdm(atomic_numbers, desc=f"Homo-nuclear diatomics for {model_name}")
    for z0 in pbar:
        elem1, elem2 = chemical_symbols[z0], chemical_symbols[z0]
        formula = f"{elem1}-{elem2}"
        pbar.set_postfix_str(formula)
        results[formula] = calc_one_pair(elem1, elem2, calculator, distances)

    with lzma.open(out_path, mode="wt") as file:
        json.dump(results, file)
    print(f"Saved results to {out_path}")

    return results


def calc_hetero_diatomics(
    z0: int, calculator: Calculator, model_name: str, atomic_numbers: Sequence[int]
) -> dict[str, list[float]]:
    """Generate potential energy data for hetero-nuclear diatomic molecules with a
    fixed first element and save to .json.xz file.

    Args:
        z0 (int): Atomic number of the fixed first element.
        calculator (Calculator): ASE calculator instance.
        model_name (str): Name of the model for the output file.
        atomic_numbers (list[int]): List of atomic numbers to calculate for.

    Returns:
        dict[str, list[float]]: Potential energy data for hetero-nuclear
            diatomic molecules or None if the file already exists.
    """
    out_path = f"{module_dir}/hetero-nuclear-diatomics-{z0}-{model_name}.json.xz"
    if os.path.isfile(out_path):
        print(f"Skipping {z0} because {out_path} already exists")
        with lzma.open(out_path, mode="rt") as file:
            return json.load(file)

    print(f"Starting {z0}")
    distances = np.linspace(0.1, 6.0, 119)
    # saving the results in a dict: "z0-z1" -> [energy] & saved the distances
    results = {"distances": list(distances)}
    # hetero-nuclear diatomics
    pbar = tqdm(atomic_numbers, desc=f"Hetero-nuclear diatomics for {model_name}")
    for z1 in pbar:
        elem1, elem2 = chemical_symbols[z0], chemical_symbols[z1]
        formula = f"{elem1}-{elem2}"
        pbar.set_postfix_str(formula)
        results[formula] = calc_one_pair(elem1, elem2, calculator, distances)

    with lzma.open(out_path, mode="wt") as file:
        json.dump(results, file)
    print(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    mace_chkpt_url = "https://github.com/ACEsuit/mace-mp/releases/download"
    checkpoints = {
        "mace-mpa-0-medium": f"{mace_chkpt_url}/mace_mpa_0/mace-mpa-0-medium.model",
        "mace-omat-0-medium": f"{mace_chkpt_url}/mace_omat_0/mace-omat-0-medium.model",
    }
    for model_name, checkpoint in checkpoints.items():
        if model_name.startswith("mace"):
            calculator = mace_mp(model=checkpoint, default_dtype="float64")
            atomic_numbers = calculator.z_table.zs
        else:
            raise ValueError(f"Unknown model: {model_name}")

        calc_homo_diatomics(calculator, model_name, atomic_numbers=atomic_numbers)

        # calculate all hetero-nuclear diatomics (takes a long time)
        for z0 in calculator.z_table.zs:
            calc_hetero_diatomics(
                z0, calculator, model_name, atomic_numbers=atomic_numbers
            )
