import os
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Sequence
from glob import glob
from importlib.metadata import version
from typing import Any, Final

import numpy as np
import ray
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import BadInputSetWarning, MPStaticSet
from tqdm import tqdm


warnings.filterwarnings("ignore", category=BadInputSetWarning)
module_dir = os.path.dirname(os.path.abspath(__file__))


def get_element_pairs(
    base_dir: str = f"{module_dir}/diatomic_calcs",
) -> list[tuple[str, str]]:
    """Get all element pairs from the directory structure.

    Args:
        base_dir (str): Base directory containing the calculations.

    Returns:
        list[tuple[str, str]]: List of (element1, element2) pairs.
    """
    pairs = []
    for el1_dir in glob(f"{base_dir}/*/"):
        el1 = os.path.basename(os.path.normpath(el1_dir))
        for pair_dir in glob(f"{el1_dir}/{el1}-*/"):
            el2 = os.path.basename(os.path.normpath(pair_dir)).split("-")[1]
            pairs.append((el1, el2))
    return pairs


def is_calculation_complete(dist_dir: str) -> bool:
    """Check if a VASP calculation is complete by looking for OUTCAR.

    Args:
        dist_dir (str): Directory to check

    Returns:
        bool: True if calculation is complete
    """
    outcar = os.path.join(dist_dir, "OUTCAR")
    if not os.path.exists(outcar):
        return False

    # Check if calculation finished successfully
    with open(outcar) as f:
        return "reached required accuracy" in f.read()


def find_closest_completed_calc(
    current_dist: float, distances: Sequence[str]
) -> str | None:
    """Find the closest completed calculation to copy WAVECAR from.

    Args:
        current_dist (float): Current distance being calculated
        distances (list[str]): List of distance directories

    Returns:
        str | None: Path to closest completed calculation, or None if none found
    """
    completed_dists = []
    for dist_dir in distances:
        if is_calculation_complete(dist_dir):
            dist = float(os.path.basename(dist_dir).split("=")[1])
            completed_dists.append((abs(dist - current_dist), dist_dir))

    return min(completed_dists)[1] if completed_dists else None


def create_vasp_inputs_locally(
    *,
    elements: set[str],
    distances: Sequence[float] = np.logspace(0, 1, 40),  # 1-10 Å, 40 points
    box_size: tuple[float, float, float] = (10, 10, 20),
) -> str:
    """Create VASP input files locally for all element pairs.

    Args:
        elements (set[str]): Elements to include
        distances (Sequence[float]): Distances to calculate in Å
        box_size (tuple[float, float, float]): Size of the cubic box in Å

    Returns:
        str: Path to temporary directory containing input files
    """
    tmp_dir = tempfile.mkdtemp(prefix="vasp_diatomics_")
    print(f"Creating VASP inputs in {tmp_dir}")

    for elem1 in elements:
        for elem2 in elements:
            pair_dir = os.path.join(tmp_dir, elem1, f"{elem1}-{elem2}")
            os.makedirs(pair_dir, exist_ok=True)

            for dist in distances:
                dist_dir = os.path.join(pair_dir, f"dist={dist:.3f}")
                os.makedirs(dist_dir, exist_ok=True)

                # Create the structure
                box = Lattice.orthorhombic(*box_size)
                center = np.array(box_size) / 2
                coords_1 = center - np.array([0, 0, dist / 2])
                coords_2 = center + np.array([0, 0, dist / 2])

                dimer = Structure(
                    box, [elem1, elem2], (coords_1, coords_2), coords_are_cartesian=True
                )

                # Generate VASP input files
                vasp_input_set = MPStaticSet(
                    dimer,
                    user_kpoints_settings=Kpoints(),  # sample a single k-point at Gamma
                    # disable symmetry since spglib in VASP sometimes detects false
                    # symmetries in dimers and fails
                    user_incar_settings={"ISYM": 0, "LH5": True},
                    user_potcar_settings={"W": "W_sv"},
                )
                vasp_input_set.write_input(dist_dir)

    return tmp_dir


@ray.remote(num_cpus=1)
def run_vasp_calculation(
    *,  # force keyword-only arguments
    elem1: str,
    elem2: str,
    task_id: int,
    out_dir: str,
    distances: Sequence[float] = np.logspace(0, 1, 40),  # 1-10 Å, 40 points
) -> tuple[str, dict[str, Any]]:
    """Run VASP calculations for a single element pair directory.

    Args:
        elem1 (str): First element symbol
        elem2 (str): Second element symbol
        task_id (int): Task ID for logging
        out_dir (str): Output directory
        distances (Sequence[float]): Distances to calculate in Å

    Returns:
        tuple[str, dict[str, Any]]: Element pair and results dictionary
    """
    pair_name = f"{elem1}-{elem2}"
    print(f"Processing {pair_name}")

    # Create directories
    pair_dir = os.path.join(out_dir, elem1, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    results_dict = {"completed_calcs": [], "errors": [], "error_traceback": []}

    # Run calculations for each distance
    for dist in distances:
        dist_dir = os.path.join(pair_dir, f"dist={dist:.3f}")
        os.makedirs(dist_dir, exist_ok=True)

        if not is_calculation_complete(dist_dir):
            # Copy input files from working directory
            src_dir = os.path.join(elem1, pair_name, f"dist={dist:.3f}")
            for file in ("INCAR", "KPOINTS", "POSCAR", "POTCAR"):
                shutil.copy2(os.path.join(src_dir, file), dist_dir)

            # Try to copy WAVECAR from closest completed calculation
            closest_calc = find_closest_completed_calc(
                dist, [os.path.join(pair_dir, f"dist={d:.3f}") for d in distances]
            )
            if closest_calc:
                subprocess.run(["cp", f"{closest_calc}/WAVECAR", dist_dir], check=True)

            # Run VASP
            try:
                vasp_cmd = "vasp_std"
                print(f"Running {vasp_cmd} in {dist_dir}")
                os.chdir(dist_dir)
                subprocess.run([vasp_cmd], check=True)
                results_dict["completed_calcs"] += [dist_dir]

                # Clean up unnecessary files to save space, keep WAVECAR for next calculation
                for pattern in (
                    "CHG*",
                    "EIGENVAL",
                    "PROCAR",
                    "PCDAT",
                    "REPORT",
                    "XDATCAR",
                    "DOSCAR",
                ):
                    subprocess.run(f"rm -f {pattern}", shell=True, check=True)

            except subprocess.CalledProcessError as exc:
                results_dict["errors"] += [f"VaspError: {exc!r}"]
                results_dict["error_traceback"] += [str(exc)]

    return pair_name, results_dict


def main(*, debug: bool = False) -> None:
    """Run VASP calculations for all element pairs using Ray.

    Args:
        debug (bool): If True, only run 10 jobs for testing
    """
    # skip superheavy elements (most have no POTCARs and are radioactive)
    skip_elements = {
        *"Am At Bk Cf Cm Es Fr Fm Md No Lr Rf Po Db Sg Bh Hs Mt Ds Cn Nh Fl Mc Lv Ra "
        "Rg Ts Og Rn".split()
    }
    elements = sorted({*map(str, Element)} - skip_elements)
    pairs = [(el1, el2) for el1 in elements for el2 in elements]
    if debug:
        pairs = pairs[:10]
        print(f"Debug mode: running only {len(pairs)} pairs")
    else:
        print(f"Found {len(pairs)} element pairs to run")

    # Create VASP input files locally
    tmp_dir = create_vasp_inputs_locally(elements=set(elements))

    # Initialize Ray (point this at the head node of the cluster)
    node_name = os.getenv("RAY_NODE_NAME", "lambda-staging-with-ray-2.40")
    ray_ip: Final[str] = {
        "lambda-staging-with-ray-2.40": "100.82.154.22",
    }.get(node_name)
    ray_address = os.getenv("RAY_ADDRESS", f"ray://{ray_ip}:10001")
    print(f"{ray_address=}")

    # Ray initialization
    if ray_address:
        # Connect to existing Ray cluster
        ray.init(
            address=ray_address,
            runtime_env={
                "uv": ["ase", "pymatgen", "pymatviz"],
                "working_dir": tmp_dir,
            },
        )
    else:
        # Start Ray locally with optimized settings for M3 Max
        ray.init(
            num_cpus=8,  # Use 8/14 cores (leaving some for system + efficiency cores)
        )

    print(f"\nConnected to Ray cluster: {ray.cluster_resources()}")
    ray_resources = ray.available_resources()
    ray_mem = ray_resources.get("memory", 0) / 1e9
    print(f"Available memory: {ray_mem:.1f} GB")
    obj_store_mem = ray_resources.get("object_store_memory", 0)
    print(f"Object store memory: {obj_store_mem / 1e9:.1f} GB")

    out_dir = os.getenv("SBATCH_OUTPUT", os.path.join(module_dir, "vasp-diatomics"))
    os.makedirs(out_dir, exist_ok=True)

    # Save run parameters
    run_params = dict(
        n_pairs=len(pairs),
        versions={dep: version(dep) for dep in ("numpy", "ray", "ase")},
    )

    with open(os.path.join(out_dir, "run_params.json"), mode="w") as file:
        import json

        json.dump(run_params, file, indent=4)

    # Process pairs in parallel
    futures = [
        run_vasp_calculation.remote(
            elem1=el1,
            elem2=el2,
            task_id=idx,
            out_dir=out_dir,
        )
        for idx, (el1, el2) in enumerate(pairs)
    ]

    # Process results as they complete
    results: dict[str, dict[str, Any]] = {}

    for future in tqdm(futures, desc="Running VASP calculations"):
        pair_name, result_dict = ray.get(future)
        results[pair_name] = result_dict

        # Save intermediate results
        with open(os.path.join(out_dir, "results.json"), mode="w") as file:
            json.dump(results, file, indent=4)

    # Clean up temporary directory
    shutil.rmtree(tmp_dir)
    print(f"\nResults saved to {out_dir!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Run only 10 jobs for testing"
    )
    args = parser.parse_args()

    main(debug=args.debug)
