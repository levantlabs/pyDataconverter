"""Simple test structure for ADCBase class"""

import unittest
from typing import Union, Tuple
from pyDataconverter.dataconverter import ADCBase, InputType


class TestADCBase(unittest.TestCase):
    """Test cases for ADCBase class"""

    def setUp(self):
        # Create a concrete test class since ADCBase is abstract
        class TestADC(ADCBase):
            def _convert_input(self, voltage: Union[float, Tuple[float, float]]) -> int:
                if self.input_type == InputType.SINGLE:
                    voltage_input = voltage
                elif self.input_type == InputType.DIFFERENTIAL:
                    voltage_input = voltage[0] - voltage[1]
                return int(voltage_input * 2 ** self.n_bits / self.v_ref)

        self.ADCClass = TestADC

    def test_initialization(self):
        """Test valid and invalid initialization parameters"""
        # Valid initialization
        adc = self.ADCClass(n_bits=12, v_ref=1.0)
        self.assertEqual(adc.n_bits, 12)
        self.assertEqual(adc.v_ref, 1.0)
        self.assertEqual(adc.input_type, InputType.DIFFERENTIAL)

        # Invalid n_bits
        with self.assertRaises(TypeError):
            self.ADCClass(n_bits=12.5)
        with self.assertRaises(ValueError):
            self.ADCClass(n_bits=0)


        # Invalid v_ref
        with self.assertRaises(ValueError):
            self.ADCClass(n_bits=12, v_ref=-1.0)
        with self.assertRaises(TypeError):
            self.ADCClass(n_bits=12, v_ref="1.0")

    def test_input_types(self):
        """Test single-ended and differential inputs"""
        # Single-ended
        adc_se = self.ADCClass(12, input_type=InputType.SINGLE)
        self.assertEqual(adc_se.convert(0.5), 2048)  # Mid-scale for 12 bits

        # Invalid single-ended input
        with self.assertRaises(TypeError):
            adc_se.convert((0.5, 0))

        # Differential
        adc_diff = self.ADCClass(12, input_type=InputType.DIFFERENTIAL)
        self.assertEqual(adc_diff.convert((0.75, 0.25)), 2048)  # 0.5V differential

        # Invalid differential input
        with self.assertRaises(TypeError):
            adc_diff.convert(0.5)
        with self.assertRaises(TypeError):
            adc_diff.convert((0.5,))



    def test_string_representation(self):
        """Test string representation"""
        adc = self.ADCClass(12, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
        expected_str = "TestADC(n_bits=12, v_ref=1.0, input_type=DIFFERENTIAL)"
        self.assertEqual(str(adc), expected_str)


if __name__ == '__main__':
    unittest.main()