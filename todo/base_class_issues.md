ADCBase/DACBase Base Class Issues
=================================

Date: 2026-03-25

LSP/TYPE ERRORS
---------------
1. _convert_input return type mismatch (ADCBase, line 87)
   - Abstract method declares return type as "None" via "pass"
   - Should be "int" for ADC, "Union[float, Tuple[float, float]]" for DAC

2. Parameter name inconsistency
   - ADCBase._convert_input uses parameter name "vin"
   - SimpleADC._convert_input uses "analog_input"
   - This causes LSP "Parameter name mismatch" errors

LSP ERRORS ACROSS FILES
-----------------------
SimpleADC.py:
  - Line 167: float() called on Tuple[float, float] - type error
  - Line 170: Unpacking tuple without type narrowing
  - convert() return type mismatch (returns int, base returns None)

FlashADC.py:
  - _convert_input return type mismatch (returns int, base returns None)
  - Line 266, 296: matplotlib list[None] type issues

SimpleDAC.py:
  - Line 190: Unpacking float as tuple in __main__ example

RECOMMENDED FIXES
-----------------
1. Update ADCBase._convert_input signature:
   @abstractmethod
   def _convert_input(self, vin: Union[float, Tuple[float, float]]) -> int:
       pass

2. Update DACBase._convert_input signature:
   @abstractmethod
   def _convert_input(self, digital_input: int) -> Union[float, Tuple[float, float]]:
       pass

3. Standardize parameter names across all implementations

4. Add type narrowing for differential input unpacking

5. Fix SimpleDAC.__main__ example to properly handle differential output
