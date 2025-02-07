""" Fun utility to visualize flash ADC operation """

from pyDataconverter.architectures.FlashADC import FlashADC
from typing import Optional
import numpy as np


def visualize_flash_adc(adc: FlashADC,
                        input_voltage: Optional[float] = None,
                        show_comparator_details: bool = True,
                        fig=None,
                        ax=None) -> None:
    """
    Visualize the Flash ADC architecture and state.

    Args:
        adc: FlashADC instance to visualize
        input_voltage: Optional input voltage to show current state
        show_comparator_details: If True, shows comparator offsets and states
        fig: Optional matplotlib figure (creates new one if None)
        ax: Optional matplotlib axis (creates new one if None)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        show_plot = True
    else:
        show_plot = False

    # Plot reference ladder
    ladder_x = 0.2
    for i, ref in enumerate(adc.reference_voltages):
        # Draw resistor
        rect = patches.Rectangle(
            (ladder_x - 0.05, ref - 0.02),
            0.1, 0.04,
            facecolor='lightgray',
            edgecolor='black'
        )
        ax.add_patch(rect)

        # Add reference voltage label
        ax.text(ladder_x - 0.15, ref, f'{ref:.3f}V')

    # Plot comparators
    comp_x = 0.5
    triangle_size = 0.06  # Reduced size

    if input_voltage is not None:
        comparator_outputs = [
            comp.compare(input_voltage, ref)
            for comp, ref in zip(adc.comparators, adc.reference_voltages)
        ]
        code = adc._convert_input(input_voltage)
    else:
        comparator_outputs = [None] * len(adc.comparators)

    for i, (comp, ref, output) in enumerate(zip(adc.comparators,
                                                adc.reference_voltages,
                                                comparator_outputs)):
        # Draw comparator triangle
        triangle = patches.RegularPolygon(
            (comp_x, ref),  # center point
            numVertices=3,  # triangle
            radius=triangle_size,  # size
            orientation=-np.pi / 2,  # Rotated 90 degrees counterclockwise
            facecolor='white',
            edgecolor='black'
        )
        ax.add_patch(triangle)

        # Draw connection lines
        ax.plot([ladder_x + 0.05, comp_x - triangle_size/2], [ref, ref], 'k-', linewidth=1)

        if show_comparator_details:
            # Show offset
            ax.text(comp_x + 0.15, ref,
                    f'Offset: {comp.offset * 1000:.1f}mV')

        # Show output state if input provided
        if output is not None:
            color = 'green' if output else 'red'
            ax.plot(comp_x, ref, 'o', color=color, markersize=8)

    # Plot input voltage if provided
    if input_voltage is not None:
        ax.axhline(y=input_voltage, color='blue', linestyle='--',
                   label=f'Input: {input_voltage:.3f}V')
        ax.text(1.0, input_voltage + 0.05,
                f'Output Code: {code:0{adc.n_bits}b} ({code})',
                va='bottom')

    # Set plot limits and labels
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, adc.v_ref * 1.1)
    ax.set_title(f'{adc.n_bits}-bit Flash ADC')
    ax.set_xlabel('Architecture Components')
    ax.set_ylabel('Input Voltage (V)')

    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.grid(True)

    if show_plot:
        plt.show()

    return fig, ax


def animate_flash_adc(adc: FlashADC,
                      input_voltages: np.ndarray,
                      interval: float = 1.0) -> None:
    """
    Animate Flash ADC operation for a sequence of input voltages.

    Args:
        adc: FlashADC instance
        input_voltages: Array of input voltages
        interval: Time between updates in seconds (default: 1.0)
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(12, 8))

    def init():
        ax.clear()
        return []

    def update(frame):
        ax.clear()
        v_in = input_voltages[frame]
        visualize_flash_adc(adc, v_in, show_comparator_details=True, fig=fig, ax=ax)
        return []

    anim = FuncAnimation(fig, update, frames=len(input_voltages),
                         init_func=init, interval=interval * 1000,  # Convert to milliseconds
                         blit=True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create ADC
    adc = FlashADC(
        n_bits=3,
        v_ref=1.0,
        comparator_params={
            'noise_rms': 0.001,
            'hysteresis': 0.002
        }
    )

    # Static visualization
    visualize_flash_adc(adc, input_voltage=0.4)

    # Animation example
    t = np.linspace(0, 2 * np.pi, 50)
    input_voltages = 0.5 + 0.4 * np.sin(t)  # Sine wave centered at 0.5V
    animate_flash_adc(adc, input_voltages, interval=0.1)