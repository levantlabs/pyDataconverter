"""pyDataconverter: a Python toolbox for modeling and analyzing data converters.

The public API is organised into fully-namespaced subpackages:

- ``pyDataconverter.architectures`` — ADC/DAC classes (SARADC, FlashADC, …)
- ``pyDataconverter.components``    — building blocks (Comparator, CDAC, …)
- ``pyDataconverter.dataconverter`` — base classes and enums (ADCBase, DACBase,
  InputType, OutputType, QuantizationMode)
- ``pyDataconverter.utils``         — utility submodules (signal_gen,
  fft_analysis, characterization, metrics, visualizations, nodal_solver)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyDataconverter")
except PackageNotFoundError:
    __version__ = "unknown"
