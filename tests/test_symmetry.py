"""Tests for symmetry module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatviz.symmetry import get_anonymous_moyo_dataset


if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_get_anonymous_moyo_dataset_basic(cubic_struct: Structure) -> None:
    """Test basic functionality of get_anonymous_moyo_dataset."""
    dataset = get_anonymous_moyo_dataset(cubic_struct)
    assert dataset.number == 229
    assert dataset.hall_number == 529
    assert dataset.angle_tolerance is None
    assert dataset.symprec == 0.0001


@pytest.mark.parametrize("symprec", [0.01, 0.1, 1.0])
def test_get_anonymous_moyo_dataset_symprec(
    cubic_struct: Structure, symprec: float
) -> None:
    """Test get_anonymous_moyo_dataset with different symprec values."""
    dataset = get_anonymous_moyo_dataset(cubic_struct, symprec=symprec)
    assert dataset is not None
    assert hasattr(dataset, "number")


def test_get_anonymous_moyo_dataset_perturbed(cubic_struct: Structure) -> None:
    """Test get_anonymous_moyo_dataset with perturbed structure."""
    # Slightly distort the structure
    perturbed = cubic_struct.copy()
    perturbed.perturb(0.1)

    # With tight tolerance
    dataset_tight = get_anonymous_moyo_dataset(perturbed, symprec=0.01)
    # With loose tolerance
    dataset_loose = get_anonymous_moyo_dataset(perturbed, symprec=0.2)

    assert dataset_tight.number != dataset_loose.number


@pytest.mark.parametrize("dummy_specie", ["H", "He", "Li"])
def test_get_anonymous_moyo_dataset_different_dummy(
    cubic_struct: Structure, dummy_specie: str
) -> None:
    """Test get_anonymous_moyo_dataset with different dummy species."""
    dataset = get_anonymous_moyo_dataset(cubic_struct, dummy_specie=dummy_specie)
    assert dataset.number == 229
    assert dataset.hall_number == 529
    assert dataset.angle_tolerance is None
    assert dataset.symprec == 0.0001


def test_get_anonymous_moyo_dataset_preserves_structure(
    cubic_struct: Structure,
) -> None:
    """Test that get_anonymous_moyo_dataset doesn't modify input structure."""
    original_positions = cubic_struct.cart_coords.copy()
    original_species = [site.species_string for site in cubic_struct]

    get_anonymous_moyo_dataset(cubic_struct)

    assert (cubic_struct.cart_coords == original_positions).all()
    assert [site.species_string for site in cubic_struct] == original_species


def test_get_anonymous_moyo_dataset_primitive_vs_conventional(
    cubic_struct: Structure,
) -> None:
    """Test get_anonymous_moyo_dataset with primitive and conventional cells."""
    spg_analyzer = SpacegroupAnalyzer(cubic_struct)
    primitive_structure = spg_analyzer.get_primitive_standard_structure()

    dataset_conv = get_anonymous_moyo_dataset(cubic_struct)
    dataset_prim = get_anonymous_moyo_dataset(primitive_structure)

    assert dataset_conv.number == dataset_prim.number


def test_get_anonymous_moyo_dataset_invalid_structure() -> None:
    """Test handling of invalid structure input."""
    with pytest.raises((AttributeError, TypeError)):
        get_anonymous_moyo_dataset("not a structure")


def test_get_anonymous_moyo_dataset_invalid_dummy_specie(
    cubic_struct: Structure,
) -> None:
    """Test handling of invalid dummy_specie input."""
    with pytest.raises(IndexError, match="list index out of range"):
        get_anonymous_moyo_dataset(cubic_struct, dummy_specie="NotAnElement")
