"""
Comparator Component Module
==========================

This module provides a basic comparator model for use in data converter simulations.

Classes:
    Comparator: Basic comparator model with configurable non-idealities

Version History:
---------------
1.0.0 (2024-02-07):
    - Initial release
    - Basic comparator with offset, noise, bandwidth, and hysteresis
    - Support for individual comparison operations
1.0.1 (2024-02-07):
    - Added time constant parameter for future temporal modeling

Notes:
------
The comparator model includes several non-idealities that can be configured:
    - DC offset
    - Input-referred noise
    - Bandwidth limitations
    - Hysteresis
    - Time constant (for future temporal modeling)
Future versions may include:
    - Temperature effects
    - Power supply sensitivity
    - Slew rate limitations
    - Temporal behavior modeling
"""

from typing import Optional
import numpy as np


class Comparator:
    """
    Basic comparator model with configurable non-idealities.

    Attributes:
        offset: DC offset voltage (in volts)
        noise_rms: RMS noise voltage (in volts)
        bandwidth: -3dB bandwidth (in Hz), None for infinite bandwidth
        hysteresis: Hysteresis voltage (in volts)
        time_constant: Time constant for temporal behavior (in seconds)

    Notes:
        The comparator model assumes:
        - Gaussian noise distribution
        - First-order bandwidth limitation
        - Symmetric hysteresis around the threshold
        - Time constant parameter reserved for future temporal modeling
    """

    def __init__(self,
                 offset: float = 0.0,
                 noise_rms: float = 0.0,
                 bandwidth: Optional[float] = None,
                 hysteresis: float = 0.0,
                 time_constant: float = 0.0):
        """
        Initialize comparator with specified non-idealities.

        Args:
            offset: DC offset voltage (in volts)
            noise_rms: RMS noise voltage (in volts)
            bandwidth: -3dB bandwidth (in Hz), None for infinite bandwidth
            hysteresis: Hysteresis voltage (in volts)
            time_constant: Time constant for temporal behavior (in seconds)
        """
        self.offset = offset
        self.noise_rms = noise_rms
        self.bandwidth = bandwidth
        self.hysteresis = hysteresis
        self.time_constant = time_constant
        self._last_output = 0  # For hysteresis

        # Initialize bandwidth-related variables if needed
        if bandwidth is not None:
            self._last_input = 0.0
            self._tau = 1 / (2 * np.pi * bandwidth)

    def compare(self, v_pos: float, v_neg: float, time_step: float = None) -> int:
        """
        Compare two voltages with non-idealities.

        Args:
            v_pos: Positive input voltage
            v_neg: Negative input voltage
            time_step: Time step for bandwidth calculations (if bandwidth is specified)

        Returns:
            1 if v_pos > v_neg (accounting for non-idealities), else 0

        Raises:
            ValueError: If bandwidth is specified but time_step is None
        """
        # Calculate input difference
        v_diff = v_pos - v_neg + self.offset

        # Apply bandwidth limitation if specified
        if self.bandwidth is not None:
            if time_step is None:
                raise ValueError("time_step must be provided when bandwidth is specified")
            alpha = time_step / (time_step + self._tau)
            v_diff = (1 - alpha) * self._last_input + alpha * v_diff
            self._last_input = v_diff

        # Add noise if specified
        if self.noise_rms > 0:
            v_diff += np.random.normal(0, self.noise_rms)

        # Apply hysteresis
        if self.hysteresis > 0:
            if self._last_output == 1:
                threshold = -self.hysteresis / 2
            else:
                threshold = self.hysteresis / 2
            result = 1 if v_diff > threshold else 0
        else:
            result = 1 if v_diff > 0 else 0

        self._last_output = result
        return result

    def reset(self):
        """
        Reset comparator state (for hysteresis and bandwidth calculations).
        """
        self._last_output = 0
        if self.bandwidth is not None:
            self._last_input = 0.0

    def __repr__(self) -> str:
        """Return string representation of comparator configuration."""
        params = [
            f"offset={self.offset:.2e}V",
            f"noise_rms={self.noise_rms:.2e}V",
            f"hysteresis={self.hysteresis:.2e}V"
        ]
        if self.bandwidth is not None:
            params.append(f"bandwidth={self.bandwidth:.2e}Hz")
        if self.time_constant > 0:
            params.append(f"time_constant={self.time_constant:.2e}s")
        return f"Comparator({', '.join(params)})"


# Rest of the comparator.py file remains the same...

if __name__ == "__main__":
    """Example usage of the Comparator class"""
    import matplotlib.pyplot as plt

    # Create a comparator with some non-idealities
    comp = Comparator(
        offset=0.001,  # 1mV offset
        noise_rms=0.0005,  # 0.5mV RMS noise
        hysteresis=0.002,  # 2mV hysteresis
        time_constant=1e-9  # 1ns time constant (for future use)
    )
    print(f"Comparator configuration:\n{comp}\n")

    # Test with a sweep of differential inputs
    v_diff = np.linspace(-0.01, 0.01, 1000)  # ±10mV sweep
    results = []

    # Perform comparison over multiple trials to show noise effects
    n_trials = 1000
    for _ in range(n_trials):
        comp.reset()  # Reset comparator state
        trial_results = [comp.compare(v, 0) for v in v_diff]
        results.append(trial_results)

    # Calculate probability of '1' output at each voltage
    prob_high = np.mean(results, axis=0)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(v_diff * 1000, prob_high, 'b-', label='Probability of High Output')
    plt.axvline(x=comp.offset * 1000, color='r', linestyle='--',
                label=f'Offset: {comp.offset * 1000:.1f}mV')
    if comp.hysteresis > 0:
        plt.axvspan(
            -comp.hysteresis / 2 * 1000,
            comp.hysteresis / 2 * 1000,
            alpha=0.2, color='g',
            label=f'Hysteresis: {comp.hysteresis * 1000:.1f}mV'
        )

    plt.grid(True)
    plt.xlabel('Differential Input (mV)')
    plt.ylabel('Probability of High Output')
    plt.title('Comparator Transfer Characteristic')
    plt.legend()
    plt.show()

    # Example of noise effect on a specific input
    print("Testing comparator at 2mV input:")
    test_results = [comp.compare(0.002, 0) for _ in range(1000)]
    prob_high = np.mean(test_results)
    print(f"Probability of high output: {prob_high:.3f}")