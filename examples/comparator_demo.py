"""
DifferentialComparator demo — Monte-Carlo transfer characteristic.

Run::

    python examples/comparator_demo.py

Sweeps a comparator with offset + noise + hysteresis across a small
differential-input range and plots the empirical P(out=1) curve from
500 trials, with the configured offset marked.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataconverter.components.comparator import DifferentialComparator


def main() -> None:
    comp = DifferentialComparator(
        offset=0.001,
        noise_rms=0.0005,
        hysteresis=0.002,
    )
    print(comp)

    v_diff = np.linspace(-0.01, 0.01, 1000)
    n_trials = 500
    results = []
    for _ in range(n_trials):
        comp.reset()
        results.append([comp.compare(v, 0.0) for v in v_diff])

    prob_high = np.mean(results, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(v_diff * 1e3, prob_high)
    plt.axvline(comp.offset * 1e3, color='r', linestyle='--',
                label=f'Offset: {comp.offset*1e3:.1f} mV')
    plt.xlabel('Differential Input (mV)')
    plt.ylabel('P(output = 1)')
    plt.title('DifferentialComparator Transfer Characteristic')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
