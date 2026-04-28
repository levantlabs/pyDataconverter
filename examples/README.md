# pyDataconverter examples

Runnable scripts demonstrating individual converter architectures and
analysis helpers.  Each script is self-contained and can be invoked
directly:

```bash
python examples/saradc_demo.py
```

| Script | What it shows |
|---|---|
| `saradc_demo.py` | 4-bit single-ended and differential SAR transfer functions; cycle-by-cycle conversion trace. |
| `flashadc_demo.py` | 3-bit Flash ADC transfer function with non-idealities, time-domain response, and the `visualize_FlashADC` static + animated visualisations. |
| `simpleadc_demo.py` | 12-bit `SimpleADC` ideal / noisy / aperture-jittered conversions. |
| `simpledac_demo.py` | 12-bit `SimpleDAC` ideal / noisy / differential conversions. |
| `comparator_demo.py` | Monte-Carlo P(out=1) curve for `DifferentialComparator` with offset + noise + hysteresis. |
| `fft_analysis_demo.py` | Windowing, spectral leakage, and frequency-resolution comparisons using `pyDataconverter.utils.fft_analysis`. |
| `signal_gen_demo.py` | Sine / two-tone / IMD / ramp / step / multitone analog and digital signal generators. |

These were previously embedded as `if __name__ == "__main__":` blocks
in the corresponding library modules; they were extracted to keep the
library code focused on the public API.
