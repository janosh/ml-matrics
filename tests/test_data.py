from __future__ import annotations

import re
from typing import ClassVar, get_args

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pymatviz._data import SupportedDataType, ptable_data_preprocessor
from pymatviz.enums import Key


class TestDataPreprocessor:
    test_dict: ClassVar = {
        "H": 1,  # int
        "He": [2.0, 4.0],  # float list
        "Li": np.array([6.0, 8.0]),  # float array
        "Na": 11.0,  # float
        "Mg": {"a": -1, "b": 14.0}.values(),  # dict_values
        "Al": {-1, 2.3},  # mixed int/float set
    }

    @staticmethod
    def _validate_output_df(output_df: pd.DataFrame) -> None:
        assert isinstance(output_df, pd.DataFrame)

        assert list(output_df) == [Key.heat_val]
        assert list(output_df.index) == ["H", "He", "Li", "Na", "Mg", "Al"]

        assert_allclose(output_df.loc["H", Key.heat_val], [1.0])
        assert_allclose(output_df.loc["He", Key.heat_val], [2.0, 4.0])
        assert_allclose(output_df.loc["Li", Key.heat_val], [6.0, 8.0])
        assert_allclose(output_df.loc["Na", Key.heat_val], [11.0])
        assert_allclose(output_df.loc["Mg", Key.heat_val], [-1.0, 14.0])

        assert output_df.attrs["vmin"] == -1.0
        assert output_df.attrs["vmax"] == 14.0

    def test_from_pd_dataframe(self) -> None:
        input_df: pd.DataFrame = pd.DataFrame(
            self.test_dict.items(), columns=[Key.element, Key.heat_val]
        ).set_index(Key.element)

        output_df: pd.DataFrame = ptable_data_preprocessor(input_df)

        self._validate_output_df(output_df)

    def test_from_bad_pd_dataframe(self) -> None:
        """Test auto-fix of badly formatted pd.DataFrame."""
        test_dict = {
            "He": [2.0, 4.0],  # float list
            "Li": np.array([6.0, 8.0]),  # float array
            "Mg": {"a": -1, "b": 14.0}.values(),  # dict_values
        }

        input_df_0 = pd.DataFrame(test_dict)

        # Elements as a row, and no proper row/column names
        output_df_0 = ptable_data_preprocessor(input_df_0)

        assert_allclose(output_df_0.loc["He", Key.heat_val], [2.0, 4.0])
        assert_allclose(output_df_0.loc["Li", Key.heat_val], [6.0, 8.0])
        assert_allclose(output_df_0.loc["Mg", Key.heat_val], [-1.0, 14.0])

        # Elements as a column, and no proper row/column names
        input_df_1 = input_df_0.copy().transpose()
        output_df_1 = ptable_data_preprocessor(input_df_1)

        assert_allclose(output_df_1.loc["He", Key.heat_val], [2.0, 4.0])
        assert_allclose(output_df_1.loc["Li", Key.heat_val], [6.0, 8.0])
        assert_allclose(output_df_1.loc["Mg", Key.heat_val], [-1.0, 14.0])

    def test_from_pd_series(self) -> None:
        input_series: pd.Series = pd.Series(self.test_dict)

        output_df = ptable_data_preprocessor(input_series)

        self._validate_output_df(output_df)

    def test_from_dict(self) -> None:
        input_dict = self.test_dict

        output_df = ptable_data_preprocessor(input_dict)

        self._validate_output_df(output_df)

    def test_unsupported_type(self) -> None:
        for invalid_data in ([0, 1, 2], range(5), "test", None):
            err_msg = (
                f"{type(invalid_data).__name__} unsupported, "
                f"choose from {get_args(SupportedDataType)}"
            )
            with pytest.raises(TypeError, match=re.escape(err_msg)):
                ptable_data_preprocessor(invalid_data)

    def test_get_vmin_vmax(self) -> None:
        # Test without nested list/array
        test_dict_0 = {"H": 1, "He": [2, 4], "Li": np.array([6, 8])}

        output_df_0 = ptable_data_preprocessor(test_dict_0)

        assert output_df_0.attrs["vmin"] == 1
        assert output_df_0.attrs["vmax"] == 8

        # Test with nested list/array
        test_dict_1 = {
            "H": 1,
            "He": [[2, 3], [4, 5]],
            "Li": [np.array([6, 7]), np.array([8, 9])],
        }

        output_df_1 = ptable_data_preprocessor(test_dict_1)

        assert output_df_1.attrs["vmin"] == 1
        assert output_df_1.attrs["vmax"] == 9


class TestMissingAnomalyHandle:
    # TODO: finish this unit test
    def test_handle_missing(self) -> None:
        pass

    def test_handle_infinity(self) -> None:
        pass
