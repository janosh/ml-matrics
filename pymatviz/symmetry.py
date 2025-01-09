"""Symmetry functions using moyopy."""

from typing import TYPE_CHECKING, Any

from pymatgen.core import Structure


if TYPE_CHECKING:
    import moyopy


def get_anonymous_moyo_dataset(
    structure: Structure, dummy_specie: str = "H", **kwargs: Any
) -> "moyopy.MoyoDataset":
    """Get anonymous moyopy dataset for a structure after replacing all species
    with a dummy element. This gives the anonymous space group that only depends on
    atomic positions, not chemical identity.

    Args:
        structure: Pymatgen Structure object
        dummy_specie: Element to replace all species with. Defaults to "H".
        **kwargs: Additional keyword arguments for moyopy.MoyoDataset.

    Returns:
        moyopy.MoyoDataset
    """
    import moyopy

    # Create a copy to avoid modifying the input structure
    struct_copy = structure.copy()
    # Get list of current species
    species = struct_copy.species
    # Replace all species with dummy element
    struct_copy.replace_species({sp: dummy_specie for sp in species})

    cell = moyopy.Cell(
        struct_copy.lattice.matrix, struct_copy.frac_coords, struct_copy.atomic_numbers
    )

    return moyopy.MoyoDataset(cell, **kwargs)
