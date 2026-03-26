"""
SAR ADC Visualizer
==================

Visualises the internal operation of a SAR ADC in two complementary panels:

  Left  — Binary search funnel
          Shows the successive approximation algorithm as a narrowing voltage
          interval, one bar per bit cycle (MSB → LSB).  Green shading marks
          the half kept after each comparison; red marks the half discarded.
          The orange tick inside each bar is the trial DAC voltage; the blue
          dashed line is the held input voltage.

  Right — C-DAC capacitor state
          Shows each bit's voltage contribution as a bar.  Green = bit decided
          1 (capacitor connected to v_ref); grey = bit decided 0 (connected to
          GND); blue = not yet decided.  An orange dashed line marks the
          running DAC output.

Functions:
    visualize_sar_adc:       Static snapshot or interactive slider.
    animate_sar_conversion:  Reveals the binary search one bit cycle at a time
                             for a single input voltage — pedagogically the most
                             useful mode.
    animate_sar_adc:         Sweeps through a sequence of input voltages,
                             showing the complete trace at each step.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from pyDataconverter.architectures.SARADC import SARADC
from pyDataconverter.dataconverter import InputType


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_input(adc: SARADC, input_voltage) -> Tuple[float, object]:
    """
    Return (v_display, vin_arg).

    v_display is the value plotted on the voltage axis.
    vin_arg   is passed directly to convert_with_trace().
    """
    if adc.input_type == InputType.DIFFERENTIAL:
        if isinstance(input_voltage, (tuple, list)) and len(input_voltage) == 2:
            v_pos, v_neg = float(input_voltage[0]), float(input_voltage[1])
        else:
            v = float(input_voltage)
            v_pos, v_neg = v / 2.0, -v / 2.0
        return v_pos - v_neg, (v_pos, v_neg)
    else:
        v = float(input_voltage)
        return v, v


def _voltage_range(adc: SARADC) -> Tuple[float, float]:
    """Full-scale limits for the voltage axis."""
    if adc.input_type == InputType.DIFFERENTIAL:
        return -adc.v_ref / 2.0, adc.v_ref / 2.0
    return 0.0, adc.v_ref


def _compute_intervals(adc: SARADC, trace: dict) -> List[Tuple[float, float]]:
    """
    Return a list of N+1 (lower, upper) bounds.

    Entry k is the search interval at the START of bit cycle k.
    Entry N is the final interval after all decisions (width ≈ 1 LSB).
    """
    v_lo, v_hi = _voltage_range(adc)
    lower, upper = v_lo, v_hi
    intervals = []
    for k in range(adc.n_bits):
        intervals.append((lower, upper))
        v_trial = trace['dac_voltages'][k]
        if trace['bit_decisions'][k] == 1:
            lower = v_trial
        else:
            upper = v_trial
    intervals.append((lower, upper))   # final interval (extra entry)
    return intervals


def _bit_contributions(adc: SARADC) -> np.ndarray:
    """
    Voltage contribution of each bit to the effective DAC output.

    contribution[k] = change in (v_refp − v_refn) when bit k is set from 0→1.
    For an ideal single-ended binary DAC this equals 2^(N-1-k) / 2^N * v_ref.
    For a differential DAC the formula is the same (both sides switch).
    Handles mismatch naturally via cdac.get_voltage().
    """
    n_bits = adc.n_bits
    v_base = adc.cdac.get_voltage(0)
    diff_base = v_base[0] - v_base[1]
    contribs = np.zeros(n_bits)
    for k in range(n_bits):
        single_code = 1 << (n_bits - 1 - k)
        v = adc.cdac.get_voltage(single_code)
        contribs[k] = (v[0] - v[1]) - diff_base
    return contribs


# ---------------------------------------------------------------------------
# Panel drawing (always clears the axes before redrawing)
# ---------------------------------------------------------------------------

_C_KEPT      = '#a8d5a2'   # light green  — chosen half (bit = 1)
_C_CLEAR     = '#f4a5a5'   # light red    — discarded half (bit = 0)
_C_UNDECIDED = '#d0e4f7'   # light blue   — not yet searched
_C_BACKGROUND= '#ebebeb'   # light grey   — full interval background
_C_TRIAL     = 'darkorange'
_C_INPUT     = 'steelblue'
_C_BIT1      = 'seagreen'
_C_BIT0      = '#cccccc'


def _draw_funnel(ax, adc: SARADC, trace: dict, v_display: float,
                 n_cycles: Optional[int] = None) -> None:
    """
    Draw the binary search funnel.

    n_cycles: draw only the first n_cycles decisions (None = all).
    Remaining cycles shown as a placeholder block in light blue.
    """
    n_bits   = adc.n_bits
    n_show   = n_bits if n_cycles is None else min(n_cycles, n_bits)
    v_lo, v_hi = _voltage_range(adc)
    intervals  = _compute_intervals(adc, trace)
    bar_w      = 0.38
    margin     = (v_hi - v_lo) * 0.10

    ax.cla()

    for k in range(n_show):
        lo, hi   = intervals[k]
        v_trial  = trace['dac_voltages'][k]
        decision = trace['bit_decisions'][k]
        x        = k + 1

        # Full interval background
        ax.add_patch(Rectangle(
            (x - bar_w / 2, lo), bar_w, hi - lo,
            facecolor=_C_BACKGROUND, edgecolor='#999999',
            linewidth=0.7, zorder=1))

        # Chosen sub-interval
        if decision == 1:
            cho_lo, cho_hi = v_trial, hi
            cho_color      = _C_KEPT
        else:
            cho_lo, cho_hi = lo, v_trial
            cho_color      = _C_CLEAR

        ax.add_patch(Rectangle(
            (x - bar_w / 2, cho_lo), bar_w, cho_hi - cho_lo,
            facecolor=cho_color, edgecolor='none', alpha=0.85, zorder=2))

        # Trial DAC voltage tick
        ax.plot([x - bar_w / 2, x + bar_w / 2], [v_trial, v_trial],
                color=_C_TRIAL, linewidth=2.0, zorder=3)

        # Decision marker (▲ = kept, ▼ = cleared)
        ax.scatter([x], [v_trial],
                   color=_C_TRIAL, s=55, zorder=4,
                   marker='^' if decision == 1 else 'v')

        # DAC voltage label
        sign = 1 if decision == 1 else -1
        ax.text(x, v_trial + sign * (v_hi - v_lo) * 0.022,
                f'{v_trial:.3f}', ha='center', fontsize=6.5, color=_C_TRIAL,
                va='bottom' if decision == 1 else 'top')

        # Bit value above the bar
        bit_lbl   = '1' if decision else '0'
        bit_color = 'seagreen' if decision else 'firebrick'
        ax.text(x, hi + (v_hi - v_lo) * 0.025, bit_lbl,
                ha='center', fontsize=9, fontweight='bold', color=bit_color)

    # Undecided placeholder bars
    if n_show < n_bits:
        rem_lo, rem_hi = intervals[n_show]
        for k in range(n_show, n_bits):
            ax.add_patch(Rectangle(
                (k + 1 - bar_w / 2, rem_lo), bar_w, rem_hi - rem_lo,
                facecolor=_C_UNDECIDED, edgecolor='#999999',
                linewidth=0.7, linestyle=':', alpha=0.55, zorder=1))

    # Dotted line connecting trial voltages (decided cycles only)
    if n_show >= 2:
        ax.plot(range(1, n_show + 1), trace['dac_voltages'][:n_show],
                color=_C_TRIAL, linewidth=0.9, linestyle=':', alpha=0.6, zorder=2)

    # Input voltage line
    ax.axhline(v_display, color=_C_INPUT, linewidth=2.0, linestyle='--', zorder=5)

    # Title
    code = trace['code']
    mode = ('Differential' if adc.input_type == InputType.DIFFERENTIAL
            else 'Single-Ended')
    if n_show == n_bits:
        suffix = f'→  Code {code}  ({code:0{n_bits}b}b)'
    else:
        suffix = f'[Bit {n_show} of {n_bits} decided]'

    ax.set_title(f'{n_bits}-bit SAR ADC ({mode}) — Binary Search\n'
                 f'Input: {v_display:.4f} V   {suffix}', fontsize=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_C_KEPT,       label='Bit = 1 (search above)'),
        mpatches.Patch(facecolor=_C_CLEAR,      label='Bit = 0 (search below)'),
        mpatches.Patch(facecolor=_C_BACKGROUND, label='Remaining interval'),
        Line2D([0], [0], color=_C_TRIAL, linewidth=1.8, label='Trial DAC voltage'),
        Line2D([0], [0], color=_C_INPUT, linewidth=1.8, linestyle='--',
               label=f'Input: {v_display:.4f} V'),
    ]
    if n_show < n_bits:
        legend_elements.append(
            mpatches.Patch(facecolor=_C_UNDECIDED, label='Not yet decided'))
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # Axes
    ax.set_xlim(0.5, n_bits + 0.5)
    ax.set_ylim(v_lo - margin, v_hi + margin)
    ax.set_xticks(range(1, n_bits + 1))
    ax.set_xticklabels(
        ['Bit {}\n{}'.format(
            n_bits - k,
            '(MSB)' if k == 0 else ('(LSB)' if k == n_bits - 1 else ''))
         for k in range(n_bits)],
        fontsize=8)
    ax.set_xlabel('Bit Cycle  (MSB → LSB)')
    ax.set_ylabel(
        'Differential Input (V)' if adc.input_type == InputType.DIFFERENTIAL
        else 'Input Voltage (V)')
    ax.grid(True, axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_cdac(ax, adc: SARADC, trace: dict,
               n_cycles: Optional[int] = None) -> None:
    """
    Draw the C-DAC capacitor state.

    Each bar represents one bit; its height is the voltage contribution of
    that bit when set to 1.  Colour indicates the decision so far.
    """
    n_bits    = adc.n_bits
    n_show    = n_bits if n_cycles is None else min(n_cycles, n_bits)
    decisions = trace['bit_decisions']
    contribs  = _bit_contributions(adc)

    ax.cla()

    colors = []
    for k in range(n_bits):
        if k < n_show:
            colors.append(_C_BIT1 if decisions[k] == 1 else _C_BIT0)
        else:
            colors.append(_C_UNDECIDED)

    x    = np.arange(n_bits)
    bars = ax.bar(x, contribs, color=colors, edgecolor='black', linewidth=0.6,
                  zorder=2)

    # Highlight the most recently decided bit with an orange border
    if n_cycles is not None and 1 <= n_cycles <= n_bits:
        bars[n_cycles - 1].set_edgecolor(_C_TRIAL)
        bars[n_cycles - 1].set_linewidth(2.2)

    # Running DAC output (sum of set bits up to n_show)
    dac_out = sum(contribs[k] for k in range(n_show) if decisions[k] == 1)
    ax.axhline(dac_out, color=_C_TRIAL, linewidth=1.8, linestyle='--', zorder=3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_C_BIT1, edgecolor='black', label='Bit = 1 (→ v_ref)'),
        mpatches.Patch(facecolor=_C_BIT0, edgecolor='black', label='Bit = 0 (→ GND)'),
        Line2D([0], [0], color=_C_TRIAL, linewidth=1.8, linestyle='--',
               label=f'DAC output: {dac_out:.4f} V'),
    ]
    if n_show < n_bits:
        legend_elements.append(
            mpatches.Patch(facecolor=_C_UNDECIDED, label='Not yet decided'))
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(
        ['Bit {}\n{}'.format(
            n_bits - k,
            'MSB' if k == 0 else ('LSB' if k == n_bits - 1 else ''))
         for k in range(n_bits)],
        fontsize=7)
    ax.set_ylabel('Bit contribution to DAC (V)')
    ax.set_title('C-DAC Capacitor State', fontsize=10)
    ax.set_ylim(0, max(contribs) * 1.35)
    ax.grid(True, axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_sar_adc(adc: SARADC,
                      input_voltage: Optional[Union[float, Tuple]] = None,
                      interactive: bool = False,
                      fig=None,
                      axes=None):
    """
    Visualise SAR ADC operation for a given input voltage.

    Left panel:  binary search funnel.
    Right panel: C-DAC capacitor state.

    Args:
        adc:           SARADC instance.
        input_voltage: Scalar (single-ended) or (v_pos, v_neg) tuple
                       (differential).  Ignored when interactive=True.
        interactive:   Add a slider for real-time exploration.
        fig, axes:     Optional existing (fig, (ax_funnel, ax_cdac)).

    Returns:
        (fig, (ax_funnel, ax_cdac))
    """
    if interactive:
        from matplotlib.widgets import Slider

        v_lo, v_hi = _voltage_range(adc)
        v_init     = (v_lo + v_hi) / 2.0

        fig = plt.figure(figsize=(14, 7))
        gs  = gridspec.GridSpec(2, 2,
                                height_ratios=[10, 1], hspace=0.45,
                                width_ratios=[6, 4])
        ax_funnel = fig.add_subplot(gs[0, 0])
        ax_cdac   = fig.add_subplot(gs[0, 1])
        ax_slider = fig.add_subplot(gs[1, :])

        is_diff = adc.input_type == InputType.DIFFERENTIAL
        label   = 'v_diff (V)' if is_diff else 'v_in (V)'
        slider  = Slider(ax_slider, label, v_lo, v_hi,
                         valinit=v_init, color=_C_INPUT)

        def _on_slide(val):
            v_disp, vin_arg = _resolve_input(adc, slider.val)
            trace = adc.convert_with_trace(vin_arg)
            _draw_funnel(ax_funnel, adc, trace, v_disp)
            _draw_cdac(ax_cdac, adc, trace)
            fig.canvas.draw_idle()

        slider.on_changed(_on_slide)
        _on_slide(v_init)
        plt.show()

    else:
        created = fig is None or axes is None
        if created:
            fig, axes_arr = plt.subplots(
                1, 2, figsize=(14, 6),
                gridspec_kw={'width_ratios': [6, 4]})
            ax_funnel, ax_cdac = axes_arr
        else:
            ax_funnel, ax_cdac = axes

        if input_voltage is not None:
            v_disp, vin_arg = _resolve_input(adc, input_voltage)
            trace = adc.convert_with_trace(vin_arg)
            _draw_funnel(ax_funnel, adc, trace, v_disp)
            _draw_cdac(ax_cdac, adc, trace)

        if created:
            plt.tight_layout()
            plt.show()

    return fig, (ax_funnel, ax_cdac)


def animate_sar_conversion(adc: SARADC,
                           input_voltage: Union[float, Tuple],
                           interval: float = 0.7):
    """
    Animate a single SAR conversion, revealing one bit cycle per frame.

    Frame k shows the first k bit decisions; remaining cycles appear as
    placeholder blocks.  The most recently decided bit is highlighted in
    orange so the viewer can follow the search step by step.

    Args:
        adc:           SARADC instance.
        input_voltage: Input voltage (scalar for SE, tuple for diff).
        interval:      Seconds between frames (default 0.7 s).

    Returns:
        FuncAnimation object (keep a reference to prevent garbage collection).
    """
    from matplotlib.animation import FuncAnimation

    v_disp, vin_arg = _resolve_input(adc, input_voltage)
    trace   = adc.convert_with_trace(vin_arg)
    n_bits  = adc.n_bits

    fig, axes_arr = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [6, 4]})
    ax_funnel, ax_cdac = axes_arr
    plt.tight_layout()

    def _update(frame):
        n_cycles = frame + 1   # frame 0 = after MSB, frame N-1 = all decided
        _draw_funnel(ax_funnel, adc, trace, v_disp, n_cycles=n_cycles)
        _draw_cdac(ax_cdac, adc, trace, n_cycles=n_cycles)
        fig.canvas.draw_idle()
        return []

    anim = FuncAnimation(fig, _update, frames=n_bits,
                         interval=interval * 1000, repeat=True, blit=False)
    plt.show()
    return anim


def animate_sar_adc(adc: SARADC,
                    input_voltages,
                    interval: float = 0.12):
    """
    Animate SAR ADC operation across a sequence of input voltages.

    Each frame shows the complete binary search trace for one input, making
    it easy to see how the search pattern shifts as the input sweeps.

    Args:
        adc:            SARADC instance.
        input_voltages: 1-D array of scalars (SE) or list of (v_pos, v_neg)
                        tuples (differential).
        interval:       Seconds between frames (default 0.12 s).

    Returns:
        FuncAnimation object.
    """
    from matplotlib.animation import FuncAnimation

    fig, axes_arr = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [6, 4]})
    ax_funnel, ax_cdac = axes_arr
    plt.tight_layout()

    def _update(frame):
        v_disp, vin_arg = _resolve_input(adc, input_voltages[frame])
        trace = adc.convert_with_trace(vin_arg)
        _draw_funnel(ax_funnel, adc, trace, v_disp)
        _draw_cdac(ax_cdac, adc, trace)
        fig.canvas.draw_idle()
        return []

    anim = FuncAnimation(fig, _update, frames=len(input_voltages),
                         interval=interval * 1000, repeat=True, blit=False)
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Static snapshot — single-ended
    adc_se = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.SINGLE)
    visualize_sar_adc(adc_se, input_voltage=0.37)
    plt.show()

    # Static snapshot — differential
    adc_diff = SARADC(n_bits=4, v_ref=1.0, input_type=InputType.DIFFERENTIAL)
    visualize_sar_adc(adc_diff, input_voltage=0.22)
    plt.show()

    # Interactive slider — single-ended
    visualize_sar_adc(adc_se, interactive=True)

    # Bit-by-bit animation for a single conversion
    animate_sar_conversion(adc_se, input_voltage=0.62)

    # Sweep animation across a sine wave
    t    = np.linspace(0, 2 * np.pi, 40)
    v_in = 0.5 + 0.45 * np.sin(t)
    animate_sar_adc(adc_se, input_voltages=v_in)
