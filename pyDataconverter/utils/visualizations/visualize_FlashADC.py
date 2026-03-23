"""Utility to visualize Flash ADC operation."""

from pyDataconverter.architectures.FlashADC import FlashADC
from pyDataconverter.dataconverter import InputType
from typing import Optional, Union, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _effective_thresholds(adc: FlashADC) -> np.ndarray:
    """
    Return the effective switching threshold for each comparator.

    Single-ended:  threshold[i] = reference_voltages[i]
    Differential:  threshold[i] = comp_refs[i] - comp_refs[n-1-i]
                   (opposite ends of the ladder → full [-v_ref/2, +v_ref/2] range)
    """
    refs = adc.reference_voltages
    if adc.input_type == InputType.DIFFERENTIAL:
        n = len(refs)
        return np.array([refs[i] - refs[n - 1 - i] for i in range(n)])
    return refs.copy()


def _resolve_input(adc: FlashADC, input_voltage):
    """
    Return (v_pos, v_neg, v_display) from a scalar or (v_pos, v_neg) tuple.
    v_display is the value drawn on the y-axis (v_diff for diff, vin for SE).
    """
    if adc.input_type == InputType.DIFFERENTIAL:
        if isinstance(input_voltage, (tuple, list)) and len(input_voltage) == 2:
            v_pos, v_neg = float(input_voltage[0]), float(input_voltage[1])
        else:
            v = float(input_voltage)
            v_pos, v_neg = v / 2, -v / 2
        return v_pos, v_neg, v_pos - v_neg
    else:
        v = float(input_voltage)
        return v, 0.0, v


def _eval_comparators(adc: FlashADC, v_pos: float, v_neg: float) -> list:
    """Evaluate each comparator using static (noiseless) reference taps."""
    refs = adc.reference_voltages
    n    = len(refs)
    if adc.input_type == InputType.DIFFERENTIAL:
        return [comp.compare(v_pos, v_neg, refs[i], refs[n - 1 - i])
                for i, comp in enumerate(adc.comparators)]
    else:
        return [comp.compare(v_pos, 0.0, ref, 0.0)
                for comp, ref in zip(adc.comparators, refs)]


def _y_limits(adc: FlashADC) -> Tuple[float, float]:
    thresh = _effective_thresholds(adc)
    span   = thresh[-1] - thresh[0]
    return thresh[0] - span * 0.15, thresh[-1] + span * 0.15


# ---------------------------------------------------------------------------
# Static drawing (ladder, triangles, wires) — called once
# ---------------------------------------------------------------------------

def _draw_static(adc: FlashADC, ax, show_comparator_details: bool,
                 ladder_x: float, comp_x: float) -> dict:
    """
    Draw all fixed elements and return a dict of updateable artists:
      'input_line'  – the horizontal input-voltage line
      'comp_dots'   – list of dot artists (one per comparator)
      'code_text'   – annotation showing the output code
    """
    import matplotlib.patches as patches

    thresh  = _effective_thresholds(adc)
    raw_refs = adc.reference_voltages
    n        = len(thresh)
    y_min, y_max = _y_limits(adc)
    span    = y_max - y_min
    tri_r   = min(0.05, span / n * 0.38)

    # Reference ladder rail
    ax.plot([ladder_x, ladder_x], [thresh[0], thresh[-1]],
            'k-', linewidth=2, zorder=1)

    for i, (t, tap) in enumerate(zip(thresh, raw_refs)):
        # Resistor rectangle
        ax.add_patch(patches.Rectangle(
            (ladder_x - 0.045, t - 0.012),
            0.09, 0.024,
            facecolor='lightgray', edgecolor='black', zorder=2))

        # Label
        label = f'{t:.3f} V'
        if adc.input_type == InputType.DIFFERENTIAL:
            label += f'  (±{abs(tap):.3f} V)'
        ax.text(ladder_x - 0.06, t, label,
                va='center', ha='right', fontsize=7.5)

        # Comparator triangle
        ax.add_patch(patches.RegularPolygon(
            (comp_x, t), numVertices=3, radius=tri_r,
            orientation=-np.pi / 2,
            facecolor='white', edgecolor='black', zorder=3))

        # Wire
        ax.plot([ladder_x + 0.045, comp_x - tri_r / 2], [t, t],
                'k-', linewidth=0.8, zorder=1)

        # Offset label
        if show_comparator_details:
            ax.text(comp_x + tri_r + 0.04, t,
                    f'{adc.comparators[i].offset * 1e3:+.1f} mV',
                    va='center', fontsize=7, color='gray')

    # Updateable artists — initially invisible
    input_line, = ax.plot([], [], color='steelblue', linestyle='--',
                          linewidth=1.6, zorder=5)
    comp_dots = [ax.plot(comp_x + tri_r * 0.65, t, 'o',
                         color='lightgray', markersize=8, zorder=4)[0]
                 for t in thresh]
    code_text = ax.text(comp_x + tri_r + 0.04, y_max,
                        '', color='steelblue', fontsize=9,
                        va='top', fontweight='bold')

    # Axes formatting
    is_diff = (adc.input_type == InputType.DIFFERENTIAL)
    mode    = 'Differential' if is_diff else 'Single-Ended'
    ax.set_title(f'{adc.n_bits}-bit Flash ADC ({mode}) — Comparator Bank')
    ax.set_xlabel('Architecture Components')
    ax.set_ylabel('Effective Threshold / Input (V)')
    ax.set_xlim(-0.25, 1.1)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([ladder_x, comp_x], labels=['Reference\nLadder', 'Comparator\nBank'])
    ax.grid(True, axis='y', alpha=0.3)

    return {'input_line': input_line, 'comp_dots': comp_dots, 'code_text': code_text}


# ---------------------------------------------------------------------------
# State update (called every time input changes)
# ---------------------------------------------------------------------------

def _update_state(adc: FlashADC, v_display: float,
                  comp_outputs: list, code: int, artists: dict,
                  y_min: float, y_max: float):
    """Update the dynamic artists to reflect the current input voltage."""
    # Input line
    artists['input_line'].set_data([-0.25, 1.1], [v_display, v_display])

    # Comparator dots
    for dot, fired in zip(artists['comp_dots'], comp_outputs):
        dot.set_color('green' if fired else 'red')

    # Code annotation
    artists['code_text'].set_text(
        f'Input: {v_display:.3f} V\nCode: {code:0{adc.n_bits}b}  ({code})')
    artists['code_text'].set_y(max(y_min + (y_max - y_min) * 0.88, v_display + (y_max - y_min) * 0.04))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_flash_adc(adc: FlashADC,
                        input_voltage: Optional[Union[float, Tuple[float, float]]] = None,
                        show_comparator_details: bool = True,
                        interactive: bool = False,
                        fig=None,
                        ax=None):
    """
    Visualize the Flash ADC comparator bank.

    For single-ended ADCs pass a scalar; for differential pass a (v_pos, v_neg)
    tuple or a scalar v_diff (split symmetrically as ±v_diff/2).

    Args:
        adc:                    FlashADC instance.
        input_voltage:          Input voltage(s) for the snapshot view.
                                Ignored when interactive=True.
        show_comparator_details: Annotate each comparator with its offset.
        interactive:            If True, add a slider for real-time exploration.
        fig, ax:                Optional existing figure/axes.

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    ladder_x, comp_x = 0.2, 0.5
    y_min, y_max = _y_limits(adc)

    if interactive:
        from matplotlib.widgets import Slider

        fig, (ax, ax_slider) = plt.subplots(
            2, 1, figsize=(12, max(7, len(adc.comparators) * 0.9)),
            gridspec_kw={'height_ratios': [10, 1]})
        fig.subplots_adjust(hspace=0.4)

        artists = _draw_static(adc, ax, show_comparator_details, ladder_x, comp_x)

        # Slider range = effective input range
        is_diff  = (adc.input_type == InputType.DIFFERENTIAL)
        v_lo     = -adc.v_ref / 2 if is_diff else 0.0
        v_hi     =  adc.v_ref / 2 if is_diff else adc.v_ref
        v_init   = (v_lo + v_hi) / 2
        label    = 'v_diff (V)' if is_diff else 'v_in (V)'

        slider = Slider(ax_slider, label, v_lo, v_hi, valinit=v_init, color='steelblue')

        def on_slide(val):
            v = slider.val
            v_pos, v_neg, v_disp = _resolve_input(adc, v)
            outputs = _eval_comparators(adc, v_pos, v_neg)
            code    = adc._convert_input(
                (v_pos, v_neg) if is_diff else v_pos)
            _update_state(adc, v_disp, outputs, code, artists, y_min, y_max)
            fig.canvas.draw_idle()

        slider.on_changed(on_slide)
        on_slide(v_init)   # initialise display
        plt.show()

    else:
        created = fig is None or ax is None
        if created:
            fig, ax = plt.subplots(figsize=(12, max(6, len(adc.comparators) * 0.9)))

        artists = _draw_static(adc, ax, show_comparator_details, ladder_x, comp_x)

        if input_voltage is not None:
            is_diff = (adc.input_type == InputType.DIFFERENTIAL)
            v_pos, v_neg, v_disp = _resolve_input(adc, input_voltage)
            outputs = _eval_comparators(adc, v_pos, v_neg)
            code    = adc._convert_input((v_pos, v_neg) if is_diff else v_pos)
            _update_state(adc, v_disp, outputs, code, artists, y_min, y_max)

        if created:
            plt.tight_layout()
            plt.show()

    return fig, ax


def animate_flash_adc(adc: FlashADC,
                      input_voltages: np.ndarray,
                      interval: float = 0.1) -> None:
    """
    Animate Flash ADC operation over a sequence of input voltages.

    Draws static elements once; updates only the dynamic artists each frame.

    Args:
        adc:            FlashADC instance.
        input_voltages: 1-D array of scalars (SE) or list of (v_pos, v_neg) tuples (diff).
        interval:       Time between frames in seconds (default 0.1 s).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ladder_x, comp_x = 0.2, 0.5
    y_min, y_max = _y_limits(adc)
    is_diff = (adc.input_type == InputType.DIFFERENTIAL)

    fig, ax = plt.subplots(figsize=(12, max(7, len(adc.comparators) * 0.9)))
    artists  = _draw_static(adc, ax, show_comparator_details=True,
                            ladder_x=ladder_x, comp_x=comp_x)

    def update(frame):
        iv = input_voltages[frame]
        v_pos, v_neg, v_disp = _resolve_input(adc, iv)
        outputs = _eval_comparators(adc, v_pos, v_neg)
        code    = adc._convert_input((v_pos, v_neg) if is_diff else v_pos)
        _update_state(adc, v_disp, outputs, code, artists, y_min, y_max)
        return list(artists.values()) + artists['comp_dots']

    FuncAnimation(fig, update, frames=len(input_voltages),
                  interval=interval * 1000, blit=False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Single-ended — interactive slider
    adc_se = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.SINGLE,
                      offset_std=0.008)
    visualize_flash_adc(adc_se, interactive=True)

    # Differential — interactive slider
    adc_diff = FlashADC(n_bits=3, v_ref=1.0, input_type=InputType.DIFFERENTIAL,
                        offset_std=0.008)
    visualize_flash_adc(adc_diff, interactive=True)
