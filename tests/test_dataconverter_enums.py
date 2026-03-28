"""
Tests for dataconverter enums and base class edge cases.
"""

import pytest
from pyDataconverter.dataconverter import (
    InputType, QuantizationMode, OutputType, ADCBase, DACBase,
)


class TestInputTypeEnum:

    def test_single_value(self):
        assert InputType.SINGLE.value == 'single'

    def test_differential_value(self):
        assert InputType.DIFFERENTIAL.value == 'differential'

    def test_members(self):
        assert set(InputType.__members__) == {'SINGLE', 'DIFFERENTIAL'}


class TestQuantizationModeEnum:

    def test_floor_value(self):
        assert QuantizationMode.FLOOR.value == 'floor'

    def test_symmetric_value(self):
        assert QuantizationMode.SYMMETRIC.value == 'symmetric'


class TestOutputTypeEnum:

    def test_single_value(self):
        assert OutputType.SINGLE.value == 'single'

    def test_differential_value(self):
        assert OutputType.DIFFERENTIAL.value == 'differential'


class TestADCBaseEdgeCases:
    """Edge cases for ADCBase not covered in test_ADCBase.py."""

    def _make_adc_class(self):
        class ConcreteADC(ADCBase):
            def _convert_input(self, vin):
                return 0
        return ConcreteADC

    def test_n_bits_1_valid(self):
        cls = self._make_adc_class()
        adc = cls(n_bits=1, v_ref=1.0)
        assert adc.n_bits == 1

    def test_n_bits_32_valid(self):
        cls = self._make_adc_class()
        adc = cls(n_bits=32, v_ref=1.0)
        assert adc.n_bits == 32

    def test_n_bits_negative_raises(self):
        cls = self._make_adc_class()
        with pytest.raises(ValueError):
            cls(n_bits=-1)

    def test_invalid_input_type_raises(self):
        cls = self._make_adc_class()
        with pytest.raises(TypeError):
            cls(n_bits=12, input_type='single')

    def test_vref_zero_raises(self):
        cls = self._make_adc_class()
        with pytest.raises(ValueError):
            cls(n_bits=12, v_ref=0.0)

    def test_integer_vref_accepted(self):
        cls = self._make_adc_class()
        adc = cls(n_bits=12, v_ref=2)
        assert adc.v_ref == 2


class TestDACBaseEdgeCases:
    """Edge cases for DACBase not covered in test_DACBase.py."""

    def _make_dac_class(self):
        class ConcreteDAC(DACBase):
            def _convert_input(self, digital_input):
                return 0.0
        return ConcreteDAC

    def test_n_bits_1_valid(self):
        cls = self._make_dac_class()
        dac = cls(n_bits=1, v_ref=1.0)
        assert dac.n_bits == 1

    def test_n_bits_32_valid(self):
        cls = self._make_dac_class()
        dac = cls(n_bits=32, v_ref=1.0)
        assert dac.n_bits == 32

    def test_n_bits_negative_raises(self):
        cls = self._make_dac_class()
        with pytest.raises(ValueError):
            cls(n_bits=-1)

    def test_invalid_output_type_raises(self):
        cls = self._make_dac_class()
        with pytest.raises(TypeError):
            cls(n_bits=12, output_type='single')

    def test_lsb_calculation(self):
        cls = self._make_dac_class()
        dac = cls(n_bits=8, v_ref=1.0)
        assert dac.lsb == pytest.approx(1.0 / 255)

    def test_convert_boolean_raises(self):
        """Boolean is a subclass of int in Python but convert should handle it."""
        cls = self._make_dac_class()
        dac = cls(n_bits=8, v_ref=1.0)
        # bool is subclass of int, so True (=1) should work
        result = dac.convert(True)
        assert result == 0.0  # our mock returns 0.0
