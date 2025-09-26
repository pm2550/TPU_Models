#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a two-lane (USB, TPU) schematic for partial off-chip execution.
The USB lane shows H2D -> Weight streaming -> D2H, while the TPU lane
shows compute. Long H2D spans are visually compressed with an axis break.
"""
import argparse
import math
import json
import sys
import subprocess
from pathlib import Path

# Global font delta applied to all text sizes (in px)
FONT_DELTA = 0.0

def svg_rect(x, y, w, h, fill, stroke="#000", sw=1, rx=3, ry=3, opacity=1.0):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx}" ry="{ry}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{sw}" />')

def svg_text(x, y, s, size=12, anchor='middle', weight='normal', family='DejaVu Sans, Arial'):
    # Apply global font delta
    final_size = max(6.0, float(size) + float(FONT_DELTA))
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{final_size}" '
            f'font-weight="{weight}">{s}</text>')

def main():
    ap = argparse.ArgumentParser(description='USB/TPU partial off-chip schematic (SVG).')
    ap.add_argument('--h2d', type=float, default=15.152, help='H2D duration (ms)')
    ap.add_argument('--compute', type=float, default=0.568, help='Compute duration (ms)')
    ap.add_argument('--d2h', type=float, default=0.060, help='D2H duration (ms)')
    ap.add_argument('--wstream', type=float, default=3.206, help='Weight streaming duration (ms)')
    ap.add_argument('--outfile', type=str, default='results/plots/partial_offchip_pipeline.svg')
    ap.add_argument('--h2d-compress', type=float, default=0.30, help='Visual compression factor for H2D (0-1]')
    ap.add_argument('--other-compress', type=float, default=1.0, help='Visual compression for the other segments (0-1]')
    # Axis break/ticks controls
    ap.add_argument('--axis-left-end', type=float, default=1.5, help='Right-most tick on left segment before break (ms)')
    ap.add_argument('--axis-right-start', type=float, default=10.5, help='Left-most tick on right segment after break (ms)')
    ap.add_argument('--tick-left-step', type=float, default=1.0, help='Tick step on the left segment (ms)')
    ap.add_argument('--tick-right-step', type=float, default=1.0, help='Tick step on the right segment (ms)')
    ap.add_argument('--break-gap', type=float, default=28.0, help='Pixel width of the axis break gap')
    ap.add_argument('--break-stub-len', type=float, default=10.0, help='Horizontal stub length at both sides of the break (px)')
    ap.add_argument('--break-slash-clear', type=float, default=4.0, help='Half clearance around slashes center (px)')
    ap.add_argument('--axis-left-overhang', type=float, default=0.5, help='Extra visible range beyond left end (ms), not labeled')
    ap.add_argument('--axis-right-overhang', type=float, default=0.5, help='Extra visible range before right start (ms), not labeled')
    ap.add_argument('--break-left-vis', type=float, default=1.5, help='Explicit left visible endpoint before break (ms), e.g., 2.5')
    ap.add_argument('--break-right-vis', type=float, default=10.5, help='Explicit right visible start after break (ms), e.g., 9.5')
    ap.add_argument('--summary-text', type=str, default=None, help='Custom summary text shown below the axis; if omitted, a default is generated')
    ap.add_argument('--copies', type=int, default=2, help='How many stacked copies of the schematic to draw')
    ap.add_argument('--copy-gap', type=float, default=36.0, help='Vertical gap between copies (px)')
    ap.add_argument('--use-measured', action='store_true', help='Use measured durations to draw geometry (applies to all copies unless --copy-modes is set)')
    ap.add_argument('--swow-ms', type=float, default=None, help='Streaming without off-chip weight duration (ms); defaults to max(H2D - wstream, 0)')
    ap.add_argument('--input-streaming', type=float, default=None, help='Input streaming duration (ms) for top copy; splits H2D')
    ap.add_argument('--copy-modes', type=str, default='original,measured', help='Comma-separated per-copy modes: original or measured, e.g., "original,measured"')
    ap.add_argument('--axis-mode', type=str, default='auto', choices=['auto','original','measured'], help='Which mode to use for axis timeline; auto uses the first copy mode')
    ap.add_argument('--font-delta', type=float, default=0.0, help='Increase all font sizes by this many px (e.g., 2.0 to go up one size)')
    ap.add_argument('--legend-gap-below', type=float, default=60.0, help='Extra vertical gap between the legend and the lanes (px)')
    ap.add_argument('--legend-gap-between', type=float, default=28.0, help='Horizontal gap between legend items (px)')
    ap.add_argument('--legend-text-width-scale', type=float, default=1.12, help='Multiplier to text width estimate to avoid overlap (e.g., 1.12)')
    ap.add_argument('--legend-label-tail-gap', type=float, default=8.0, help='Extra pixels after each label to separate legend cards')
    ap.add_argument('--legend-row-gap', type=float, default=8.0, help='Vertical gap between legend rows (px)')
    ap.add_argument('--legend-scale', type=float, default=1.0, help='Scale factor for legend text/swatch height (e.g., 1.2)')
    ap.add_argument('--legend-match-lane', action='store_true', help='Make legend swatch height equal to lane height')
    ap.add_argument('--legend-swatch-width', type=float, default=28.0, help='Legend swatch width in px (kept moderate)')
    # Bottom-copy overrides (apply to copy index 1 if provided)
    ap.add_argument('--bottom-swow-ms', type=float, default=None, help='Bottom copy: streaming without off-chip weight duration (ms)')
    ap.add_argument('--bottom-compute', type=float, default=None, help='Bottom copy: compute duration (ms)')
    ap.add_argument('--bottom-d2h', type=float, default=None, help='Bottom copy: D2H duration (ms)')
    ap.add_argument('--bottom-wstream', type=float, default=None, help='Bottom copy: off-chip weight streaming duration (ms)')
    ap.add_argument('--bottom-h2d', type=float, default=None, help='Bottom copy: H2D duration (ms) (used only if mode=original for bottom)')
    ap.add_argument('--bottom-input-streaming', type=float, default=None, help='Bottom copy: input streaming duration (ms)')
    # Compute tail-fixed extension: draw compute with a longer duration by moving its head left, tail unchanged
    ap.add_argument('--compute-tailfixed-ms', type=float, default=None, help='Top copy: draw compute with this duration but keep its end time unchanged (extends head left)')
    ap.add_argument('--bottom-compute-tailfixed-ms', type=float, default=None, help='Bottom copy: draw compute with this duration but keep its end time unchanged (extends head left)')
    # Axis unit label (e.g., ms) near the arrow on the x-axis
    ap.add_argument('--axis-unit', type=str, default='ms', help='Unit label for the axis (e.g., ms). Empty string to hide')
    ap.add_argument('--axis-unit-size', type=float, default=16.0, help='Font size for the axis unit label (px)')
    # Optional per-copy titles rendered at the far left of each copy (left of USB/TPU labels)
    ap.add_argument('--title-top', type=str, default='Partial on-chip', help='Title for the top copy (e.g., "Partial on-chip")')
    ap.add_argument('--title-bottom', type=str, default='Fully on-chip', help='Title for the bottom copy (e.g., "Fully on-chip")')
    ap.add_argument('--block-label-size', type=float, default=16.0, help='Font size for numeric labels on blocks (px)')
    ap.add_argument('--lane-label-size', type=float, default=None, help='Font size for lane labels USB/TPU (px); default matches legend text size')
    ap.add_argument('--title-size', type=float, default=None, help='Font size for per-copy titles (px); default matches legend text size')
    ap.add_argument('--tick-font-size', type=float, default=None, help='Font size for axis tick labels (px); default matches legend text size')
    ap.add_argument('--label-margin', type=float, default=6.0, help='Horizontal gap between lane labels and the bar start (px)')
    ap.add_argument('--pad-left', type=float, default=None, help='Override left padding (px); if not set, auto-expands when titles are present')
    ap.add_argument('--lane-gap', type=float, default=None, help='Vertical gap between USB and TPU lanes (px)')
    ap.add_argument('--axis-gap', type=float, default=8.0, help='Gap between TPU lane and the axis baseline (px)')
    ap.add_argument('--title-offset-x', type=float, default=-6.0, help='Horizontal offset for titles relative to the USB/TPU label center (px)')
    ap.add_argument('--title-margin', type=float, default=6.0, help='Gap between title and bar start (pad_l) in px')
    ap.add_argument('--title-usb-gap', type=float, default=8.0, help='Min gap between title end and the first letter of USB (px)')
    ap.add_argument('--lane-label-swap-offset', type=float, default=60.0, help='When swapping, distance (px) to place USB/TPU left of the bars')
    # Dataset registry and split export
    ap.add_argument('--dataset-file', type=str, default='data/partial_offchip_schematic.json', help='JSON file storing named datasets')
    ap.add_argument('--dataset-keys', type=str, default='partial,fully', help='Comma-separated dataset keys to use (order matters)')
    ap.add_argument('--split-into-singles', action='store_true', help='Generate separate single-copy images for each dataset key')
    ap.add_argument('--outfile-base', type=str, default=None, help='Base path for split outputs; defaults to --outfile without extension')
    args = ap.parse_args()

    # Split-export mode: generate single-copy images for datasets defined in a registry
    if args.split_into_singles:
        ds_path = Path(args.dataset_file)
        if not ds_path.exists():
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            default = {
                "datasets": {
                    "partial": {"title": "Partial on-chip", "h2d": 15.152, "wstream": 3.206, "compute": 0.568, "d2h": 0.060, "input": 0.413},
                    "fully": {"title": "Fully on-chip", "h2d": 14.161, "wstream": 0.0, "compute": 3.120, "d2h": 0.060, "input": 0.414}
                }
            }
            ds_path.write_text(json.dumps(default, indent=2), encoding='utf-8')
        data = json.loads(ds_path.read_text(encoding='utf-8'))
        datasets = data.get('datasets', {})
        keys = [k.strip() for k in args.dataset_keys.split(',') if k.strip()]
        base_noext = Path(args.outfile_base) if args.outfile_base else Path(args.outfile).with_suffix('')
        outputs = []
        for k in keys:
            if k not in datasets:
                continue
            d = datasets[k]
            out_svg = str(Path(f"{base_noext}_{k}.svg"))
            child_argv = [
                sys.executable, __file__,
                '--outfile', out_svg,
                '--h2d', str(d.get('h2d', 0.0)),
                '--wstream', str(d.get('wstream', 0.0)),
                '--compute', str(d.get('compute', 0.0)),
                '--d2h', str(d.get('d2h', 0.0)),
                '--input-streaming', str(d.get('input', 0.0)),
                '--copies', '1',
                '--title-top', d.get('title', k),
                '--axis-unit', args.axis_unit,
                '--axis-unit-size', str(args.axis_unit_size),
                '--break-left-vis', str(args.break_left_vis if args.break_left_vis is not None else args.axis_left_end),
                '--break-right-vis', str(args.break_right_vis if args.break_right_vis is not None else args.axis_right_start),
                '--axis-left-end', str(args.axis_left_end),
                '--axis-right-start', str(args.axis_right_start),
                '--tick-left-step', str(args.tick_left_step),
                '--tick-right-step', str(args.tick_right_step)
            ]
            subprocess.run(child_argv, check=True)
            outputs.append(out_svg)
        print("\n".join(outputs))
        return

    pad_l, pad_r, pad_t, pad_b = 90.0, 60.0, 40.0, 60.0
    # Set global font delta
    global FONT_DELTA
    FONT_DELTA = float(args.font_delta)
    lane_h = 26.0
    lane_gap = (24.0 if args.lane_gap is None else float(args.lane_gap))
    lane_round = 4
    scale = 60.0  # px per ms (before compression)
    # Default legend swatch height to match lane height unless overridden
    if not getattr(args, 'legend_match_lane', False):
        args.legend_match_lane = True

    # Allow manual override or auto-expand left padding when titles are present
    if args.pad_left is not None:
        pad_l = float(args.pad_left)
    else:
        def est_text_w(s: str, size_px: float) -> float:
            # Rough monospace-ish estimate used elsewhere in this script
            return len(s) * 7.8 * ((size_px + FONT_DELTA) / 12.0)
        max_title_w = 0.0
        if args.title_top:
            max_title_w = max(max_title_w, est_text_w(args.title_top, 12.0))
        if args.title_bottom:
            max_title_w = max(max_title_w, est_text_w(args.title_bottom, 12.0))
        if max_title_w > 0.0:
            # We right-align the title at x = pad_l/2 - 10, so ensure ~10px margin on the left
            min_pad_for_titles = 2.0 * (max_title_w + 20.0)
            if pad_l < min_pad_for_titles:
                pad_l = min_pad_for_titles

    # Actual timeline boundaries
    ch = max(0.05, min(1.0, args.h2d_compress))  # kept for CLI compatibility (not used in x mapping now)
    co = max(0.05, min(1.0, args.other_compress))

    t0 = 0.0
    # Prepare duration calculators
    swow = args.swow_ms if args.swow_ms is not None else max(0.0, args.h2d - args.wstream)
    def compute_durations(mode: str, idx: int):
        # Determine per-copy base values
        if idx == 1:  # bottom copy overrides if provided
            wstream = args.bottom_wstream if args.bottom_wstream is not None else args.wstream
            compute = args.bottom_compute if args.bottom_compute is not None else args.compute
            d2h = args.bottom_d2h if args.bottom_d2h is not None else args.d2h
            # H2D geometry always uses the per-copy 'streaming without' (swow)
            if mode == 'measured' and args.bottom_swow_ms is not None:
                h2d_duration = args.bottom_swow_ms
            else:
                # derive from provided bottom h2d/wstream if available, else fall back to global swow
                base_h2d = args.bottom_h2d if args.bottom_h2d is not None else args.h2d
                base_wstream = wstream
                h2d_duration = max(0.0, base_h2d - base_wstream)
            input_stream = args.bottom_input_streaming if args.bottom_input_streaming is not None else 0.0
        else:
            wstream = args.wstream
            compute = args.compute
            d2h = args.d2h
            # H2D geometry uses global 'swow' when measured, otherwise compute difference as well
            if mode == 'measured' and args.swow_ms is not None:
                h2d_duration = args.swow_ms
            else:
                h2d_duration = max(0.0, args.h2d - args.wstream)
            input_stream = args.input_streaming if args.input_streaming is not None else 0.0
        # Split H2D into input + on-chip
        input_stream = max(0.0, min(h2d_duration, float(input_stream)))
        onchip = max(0.0, h2d_duration - input_stream)
        return {
            'input': input_stream,
            'onchip': onchip,
            'wstream': wstream,
            'compute': compute,
            'd2h': d2h,
        }

    # Determine per-copy modes
    copies = max(1, int(args.copies))
    if args.copy_modes:
        raw_modes = [s.strip().lower() for s in args.copy_modes.split(',') if s.strip()]
        copy_modes = []
        for i in range(copies):
            m = raw_modes[i] if i < len(raw_modes) else (raw_modes[-1] if raw_modes else ('measured' if args.use_measured else 'original'))
            copy_modes.append('measured' if m == 'measured' else 'original')
    else:
        copy_modes = [('measured' if args.use_measured else 'original') for _ in range(copies)]

    # Axis mode selection
    if args.axis_mode == 'auto':
        axis_mode = copy_modes[0]
    else:
        axis_mode = args.axis_mode

    # Axis timeline durations (shared mapping)
    axis_durs = compute_durations(axis_mode, 0)
    h2d_s, h2d_e = 0.0, axis_durs['input'] + axis_durs['onchip']
    ws_s, ws_e = h2d_e, h2d_e + axis_durs['wstream']
    comp_s, comp_e = ws_e, ws_e + axis_durs['compute']
    d2h_s, d2h_e = comp_e, comp_e + axis_durs['d2h']
    total_actual = d2h_e
    # Extend axis to cover the longest copy so trailing segments (e.g., bottom D2H) are visible
    copy_totals = []
    for i in range(copies):
        di = compute_durations(copy_modes[i], i)
        copy_totals.append((di['input'] + di['onchip']) + di['wstream'] + di['compute'] + di['d2h'])
    if copy_totals:
        total_actual = max(total_actual, max(copy_totals))

    # Broken-axis configuration
    left_end_val = max(0.0, args.axis_left_end)
    right_start_val = max(left_end_val + 1e-3, args.axis_right_start)
    if right_start_val > total_actual:
        right_start_val = max(left_end_val + 1e-3, min(total_actual, args.axis_right_start))

    # Visual overhangs (extend visible line a bit beyond tick ranges without labeling)
    left_overhang = max(0.0, args.axis_left_overhang)
    right_overhang = max(0.0, args.axis_right_overhang)
    # Allow explicit visible endpoints to position the break (e.g., 2.5 and 9.5)
    if args.break_left_vis is not None and args.break_right_vis is not None:
        left_vis_end = float(args.break_left_vis)
        right_vis_start = float(args.break_right_vis)
    else:
        left_vis_end = min(total_actual, left_end_val + left_overhang)
        right_vis_start = max(0.0, right_start_val - right_overhang)
    if right_vis_start <= left_vis_end + 1e-6:
        # Ensure a non-zero gap in time domain
        mid = (left_vis_end + right_vis_start) / 2.0
        left_vis_end = mid - 1e-3
        right_vis_start = mid + 1e-3

    gap = max(8.0, float(args.break_gap))  # pixels between left and right axis segments

    # Compute overall width using uniform scale on both sides (use visual endpoints)
    axis_visual_width = scale * (left_vis_end - t0) + gap + scale * (total_actual - right_vis_start)
    # Baseline geometry (copy 0)
    legend_gap_below = max(0.0, float(args.legend_gap_below))
    usb_y0 = pad_t + legend_gap_below
    tpu_y0 = usb_y0 + lane_h + lane_gap
    # One copy vertical extent from its USB top to below summary text
    # y_base = tpu_y + lane_h + 12, tick labels go to +28, summary at +50
    per_copy_height = (tpu_y0 - usb_y0) + lane_h + 62.0  # (lane_h+lane_gap) + lane_h + 62
    per_copy_height = 2 * lane_h + lane_gap + 62.0
    copy_gap = max(0.0, float(args.copy_gap))
    # Compute tight total height based on the bottom-most content of the last copy
    last_offset = (copies - 1) * (per_copy_height + copy_gap)
    usb_y_last = usb_y0 + last_offset
    tpu_y_last = tpu_y0 + last_offset
    y_base_last = tpu_y_last + lane_h + 12.0
    # Tick labels extend to y_base + 28; add a tiny safety margin
    bottom_content = y_base_last + 28.0
    # If summary text is present, it sits lower at y_base + 50
    if args.summary_text:
        bottom_content = max(bottom_content, y_base_last + 50.0)
    total_h = bottom_content + pad_b

    col_h2d = '#1f77b4'  # On-chip weight streaming
    col_d2h = '#d62728'
    col_comp = '#7f7f7f'
    col_stream = '#f3dcb4'
    col_input = '#9ecae1'  # Input streaming uses a distinct lighter blue

    # Prepare legend metrics first to ensure the canvas is wide enough to contain it
    # Legend baseline Y; ensure swatch top is not outside the canvas
    legend_y = 16.0
    lg_scale = max(0.5, float(args.legend_scale))
    sw_w = max(10.0, float(args.legend_swatch_width))
    sw_h = (lane_h if args.legend_match_lane else 16.0 * lg_scale)
    # If legend_y - sw_h + 2 < 0, the swatch would be cropped; bump legend_y
    if legend_y - sw_h + 2 < 0:
        legend_y = sw_h + 2.0
    label_gap = 7.0 * (lane_h / 26.0 if args.legend_match_lane else lg_scale)
    gap_between = max(0.0, float(args.legend_gap_between))
    mono_family = 'DejaVu Sans Mono, Menlo, Consolas, monospace'
    # Legend items: top row shows D2H, Compute, Off-chip streaming; bottom row shows On-chip + Input
    items = [
        (col_stream, 'Off-chip weight streaming', 'wstreamCross'),
        (col_comp, 'Compute', None),
        (col_d2h, 'D2H', 'd2hHatch'),
        (col_h2d, 'On-chip weight streaming', 'h2dHatch'),
        (col_input, 'Input streaming', 'inputHatch'),
    ]
    # Scale label width estimate by legend text size change (baseline 12px)
    legend_font_size = 12.0 * (lane_h / 16.0) if args.legend_match_lane else 12.0 * lg_scale
    font_scale = (legend_font_size + FONT_DELTA) / 12.0
    tws = max(0.9, float(args.legend_text_width_scale))
    tail_gap = max(0.0, float(args.legend_label_tail_gap))
    # Default other text sizes to match legend text size if not explicitly set
    lane_label_px = float(args.lane_label_size) if args.lane_label_size is not None else float(legend_font_size)
    title_px = float(args.title_size) if args.title_size is not None else float(legend_font_size)
    tick_px = float(args.tick_font_size) if args.tick_font_size is not None else float(legend_font_size)
    widths = [sw_w + label_gap + len(name)*7.8 * font_scale * tws + tail_gap for _c, name, _p in items]
    # Top row: D2H, Compute, Off-chip; Bottom row: On-chip, Input
    # (indices correspond to the 'items' array above)
    row1_idx = [2,1,0]
    row2_idx = [3,4]
    row1_w = sum(widths[i] for i in row1_idx) + (len(row1_idx)-1)*gap_between
    row2_w = sum(widths[i] for i in row2_idx) + (len(row2_idx)-1)*gap_between if row2_idx else 0.0
    legend_w = max(row1_w, row2_w)

    # Ensure the canvas is wide enough to fit both axis and legend (with margins)
    min_canvas_w = legend_w + 40.0  # ~20px margins on both sides
    total_w = max(pad_l + axis_visual_width + pad_r + 80.0, min_canvas_w)

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{total_h:.0f}">')
    out.append('<defs>')
    out.append('<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
               '<path d="M0,0 L0,7 L9,3.5 z" fill="#4d4d4d" /></marker>')
    out.append('<pattern id="h2dHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
               '<rect width="6" height="6" fill="none" />'
               '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.55" />'
               '</pattern>')
    out.append('<pattern id="d2hHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(135)">'
               '<rect width="6" height="6" fill="none" />'
               '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.55" />'
               '</pattern>')
    out.append('<pattern id="wstreamCross" patternUnits="userSpaceOnUse" width="6" height="6">'
               '<rect width="6" height="6" fill="none" />'
               '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.1" opacity="0.55" />'
               '<path d="M 0 0 L 6 0" stroke="#000000" stroke-width="1.1" opacity="0.55" />'
               '</pattern>')
    out.append('<pattern id="inputHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
               '<rect width="6" height="6" fill="none" />'
               '<path d="M 3 0 L 3 6" stroke="#000000" stroke-width="1.1" opacity="0.55" />'
               '</pattern>')
    out.append('</defs>')

    # Raise legend higher (further from bars)
    # Two rows of legend items, horizontally centered to the bars/axis area
    row_gap = max(0.0, float(args.legend_row_gap))
    # Row 1 (center to the visible axis span rather than full canvas)
    axis_left_x = pad_l
    axis_right_x = pad_l + axis_visual_width
    axis_center_x = (axis_left_x + axis_right_x) / 2.0
    lx1 = axis_center_x - row1_w/2.0
    xc = lx1
    for i in row1_idx:
        fill_color, name, pattern_id = items[i]
        rect_y = legend_y - sw_h + 2
        out.append(svg_rect(xc, rect_y, sw_w, sw_h, fill_color, sw=0.8, rx=2, ry=2, opacity=0.95))
        if pattern_id:
            out.append(svg_rect(xc, rect_y, sw_w, sw_h, f'url(#{pattern_id})', stroke='none', sw=0, rx=2, ry=2, opacity=1.0))
        out.append(svg_text(xc + sw_w + label_gap, legend_y, name, anchor='start', size=legend_font_size, family=mono_family))
        # Advance by swatch + label + extra gap. Scale label width by current font.
        xc += sw_w + label_gap + len(name)*7.8 * font_scale * tws + tail_gap + gap_between
    # Row 2
    if row2_idx:
        legend_y2 = legend_y + sw_h + row_gap
        lx2 = axis_center_x - row2_w/2.0
        xc = lx2
        for i in row2_idx:
            fill_color, name, pattern_id = items[i]
            rect_y = legend_y2 - sw_h + 2
            out.append(svg_rect(xc, rect_y, sw_w, sw_h, fill_color, sw=0.8, rx=2, ry=2, opacity=0.95))
            if pattern_id:
                out.append(svg_rect(xc, rect_y, sw_w, sw_h, f'url(#{pattern_id})', stroke='none', sw=0, rx=2, ry=2, opacity=1.0))
            out.append(svg_text(xc + sw_w + label_gap, legend_y2, name, anchor='start', size=legend_font_size, family=mono_family))
            # Advance by swatch + label + extra gap. Scale label width by current font.
            xc += sw_w + label_gap + len(name)*7.8 * font_scale * tws + tail_gap + gap_between

    # Lane labels will be drawn per copy

    def map_time_broken(actual: float) -> float | None:
        """Map an actual time (ms) to x with a broken axis.
        Left segment shows [0 .. left_end_val], right segment shows [right_start_val .. total_actual].
        Returns None if actual is in the elided gap (left_end_val, right_start_val).
        """
        if actual <= left_vis_end + 1e-9:
            return pad_l + scale * (actual - t0)
        if actual >= right_vis_start - 1e-9:
            return pad_l + scale * (left_vis_end - t0) + gap + scale * (actual - right_vis_start)
        return None

    def draw_block(y: float, start_t: float, end_t: float, color: str, hatch: str | None,
                   label: str | None = None, label_offset_y: float = -6.0, opacity: float = 0.95):
        """Draw a horizontal block, keeping visual continuity across the broken axis.
        - If the block lies fully in left or right segment: draw normally.
        - If it spans across the omitted middle: draw a single rectangle from mapped(start) to mapped(end),
          visually continuous over the break (axis shows the break).
        - If it is partially in the gap: draw only the visible portion (closest boundary).
        """
        # Normalize
        start_t = max(start_t, t0)
        end_t = min(end_t, total_actual)
        if end_t - start_t <= 1e-9:
            return

        def map_endpoint(t: float):
            return map_time_broken(t)

        # Cases
        if end_t <= left_vis_end + 1e-9:
            # Entirely in left
            x1 = map_endpoint(start_t)
            x2 = map_endpoint(end_t)
        elif start_t >= right_vis_start - 1e-9:
            # Entirely in right
            x1 = map_endpoint(start_t)
            x2 = map_endpoint(end_t)
        elif start_t <= left_vis_end + 1e-9 and end_t >= right_vis_start - 1e-9:
            # Spans across the gap – draw one continuous rect from left start to right end
            x1 = map_endpoint(start_t)
            x2 = map_endpoint(end_t)
        elif start_t <= left_vis_end + 1e-9 and end_t > left_vis_end + 1e-9 and end_t < right_vis_start - 1e-9:
            # Ends in gap – draw up to left boundary
            x1 = map_endpoint(start_t)
            x2 = map_endpoint(left_vis_end)
        elif start_t > left_vis_end + 1e-9 and start_t < right_vis_start - 1e-9 and end_t >= right_vis_start - 1e-9:
            # Starts in gap – draw from right boundary
            x1 = map_endpoint(right_vis_start)
            x2 = map_endpoint(end_t)
        else:
            # Entirely in gap – nothing visible
            return

        if x1 is None or x2 is None:
            return

        x = min(x1, x2)
        w = abs(x2 - x1)
        out.append(svg_rect(x, y, w, lane_h, color, rx=lane_round, ry=lane_round, opacity=opacity))
        if hatch:
            out.append(svg_rect(x, y, w, lane_h, f'url(#{hatch})', stroke='none', sw=0, rx=lane_round, ry=lane_round, opacity=1.0))
        if label:
            out.append(svg_text(x + w/2, y + label_offset_y, label, size=float(args.block_label_size)))

    def render_one(y_offset: float, idx: int):
        usb_y = usb_y0 + y_offset
        tpu_y = tpu_y0 + y_offset

        # Place USB/TPU adjacent to the bars (near_bars_x), and put the title to their left
        near_bars_x = pad_l - max(0.0, float(args.label_margin))
        out.append(svg_text(near_bars_x, usb_y + lane_h*0.65, 'USB', anchor='end', size=lane_label_px, weight='bold'))
        out.append(svg_text(near_bars_x, tpu_y + lane_h*0.65, 'TPU', anchor='end', size=lane_label_px, weight='bold'))

        # Optional per-copy title: place to the LEFT of the USB/TPU labels area
        # Right-align so the text grows leftwards from (pad_l/2 - title_margin)
        # Title sits to the left of USB/TPU labels, and must end before USB starts
        # Estimate USB label width to enforce the constraint: title_end_x < usb_start_x
        usb_char_w = 7.8 * ((lane_label_px + FONT_DELTA) / 12.0)
        usb_w = 3.0 * usb_char_w  # width of 'USB'
        min_margin = usb_w + max(0.0, float(args.title_usb_gap))
        effective_margin = max(max(0.0, float(args.title_margin)), min_margin)
        title_x = near_bars_x - effective_margin
        title_y = usb_y + lane_h + (lane_gap / 2.0)
        if idx == 0 and args.title_top:
            out.append(svg_text(title_x, title_y, args.title_top, anchor='end', size=title_px, weight='bold'))
        if idx == 1 and args.title_bottom:
            out.append(svg_text(title_x, title_y, args.title_bottom, anchor='end', size=title_px, weight='bold'))

        # Blocks
        mode = copy_modes[idx]
        durs = compute_durations(mode, idx)
        # Local starts/ends for this copy's blocks (but mapping still uses axis timeline)
        # Place On-chip weight streaming first, then Input streaming, then Off-chip streaming
        lonc_s, lonc_e = 0.0, durs.get('onchip', 0.0)
        lin_s, lin_e = lonc_e, lonc_e + durs.get('input', 0.0)
        lws_s, lws_e = lin_e, lin_e + durs['wstream']
        lcomp_s, lcomp_e = lws_e, lws_e + durs['compute']
        # Optionally keep tail fixed and extend compute head left to reach a desired duration
        if idx == 0 and args.compute_tailfixed_ms is not None:
            desired = float(args.compute_tailfixed_ms)
            if desired > 0:
                lcomp_s = max(0.0, lcomp_e - desired)
                compute_display = desired
            else:
                compute_display = durs['compute']
        elif idx == 1 and args.bottom_compute_tailfixed_ms is not None:
            desired = float(args.bottom_compute_tailfixed_ms)
            if desired > 0:
                lcomp_s = max(0.0, lcomp_e - desired)
                compute_display = desired
            else:
                compute_display = durs['compute']
        else:
            compute_display = durs['compute']
        # Place D2H after compute so compute is between streaming and D2H
        ld2h_s, ld2h_e = lcomp_e, lcomp_e + durs['d2h']

        # Draw input and on-chip (split H2D)
        if durs.get('onchip', 0.0) > 0:
            draw_block(usb_y, lonc_s, lonc_e, col_h2d, 'h2dHatch', label=f"{durs['onchip']:.3f}", label_offset_y=-6.0, opacity=0.95)
        if durs.get('input', 0.0) > 0:
            draw_block(usb_y, lin_s, lin_e, col_input, 'inputHatch', label=f"{durs['input']:.3f}", label_offset_y=-6.0, opacity=0.95)
        # Place Off-chip weight streaming label above the bar to avoid overlap with TPU compute below
        draw_block(usb_y, lws_s, lws_e, col_stream, 'wstreamCross', label=f"{durs['wstream']:.3f}", label_offset_y=-6.0, opacity=0.92)
        draw_block(usb_y, ld2h_s, ld2h_e, col_d2h, 'd2hHatch', label=f"{durs['d2h']:.3f}", label_offset_y=-6.0, opacity=0.95)
        draw_block(tpu_y, lcomp_s, lcomp_e, col_comp, None, label=f"{compute_display:.3f}", label_offset_y=-6.0, opacity=0.65)

        # Axis (only for the first copy to share one axis)
        if idx == 0:
            base_x1 = pad_l
            y_base = tpu_y + lane_h + float(args.axis_gap)
            left_end_x = map_time_broken(left_vis_end) or base_x1
            right_start_x = map_time_broken(right_vis_start) or (base_x1 + scale * (left_vis_end - t0) + gap)
            axis_end_line = map_time_broken(total_actual) or right_start_x
            axis_end = axis_end_line + 60.0

            # Break with stubs
            break_center = (left_end_x + right_start_x) / 2.0
            stub_len = max(0.0, float(args.break_stub_len))
            slash_clear = max(0.0, float(args.break_slash_clear))
            gap_width = max(0.0, right_start_x - left_end_x)
            # If explicit break endpoints are given, enforce symmetric spacing around the break
            if args.break_left_vis is not None and args.break_right_vis is not None:
                # Ensure slash_clear does not exceed half the gap
                max_clear = max(0.0, (right_start_x - left_end_x) / 2.0 - 2.0)
                if slash_clear > max_clear:
                    slash_clear = max_clear
                left_axis_end_draw = break_center - slash_clear
                right_axis_start_draw = break_center + slash_clear
            else:
                left_axis_end_draw = min(left_end_x + stub_len, break_center - slash_clear)
                right_axis_start_draw = max(right_start_x - stub_len, break_center + slash_clear)
            if right_axis_start_draw - left_axis_end_draw < 6.0:
                left_axis_end_draw = left_end_x
                right_axis_start_draw = right_start_x
                slash_clear = max(4.0, gap_width / 4.0)

            out.append(f'<line x1="{base_x1:.2f}" y1="{y_base:.2f}" x2="{left_axis_end_draw:.2f}" y2="{y_base:.2f}" stroke="#4d4d4d" stroke-width="1.3" />')
            out.append(f'<path d="M {break_center-5:.2f},{(y_base-3):.2f} L {break_center-1:.2f},{(y_base+3):.2f}" stroke="#4d4d4d" stroke-width="1.3" />')
            out.append(f'<path d="M {break_center+1:.2f},{(y_base-3):.2f} L {break_center+5:.2f},{(y_base+3):.2f}" stroke="#4d4d4d" stroke-width="1.3" />')
            out.append(f'<line x1="{right_axis_start_draw:.2f}" y1="{y_base:.2f}" x2="{axis_end:.2f}" y2="{y_base:.2f}" stroke="#4d4d4d" stroke-width="1.3" marker-end="url(#arrowhead)" />')

            # Axis unit label near the arrow (e.g., 'ms')
            if args.axis_unit:
                unit_x = axis_end - 8.0
                unit_y = y_base - 10.0
                out.append(svg_text(unit_x, unit_y, args.axis_unit, anchor='end', size=float(args.axis_unit_size)))

            # Ticks
            lstep = max(1e-6, args.tick_left_step)
            rstep = max(1e-6, args.tick_right_step)
            val = 0.0
            while val <= left_end_val + 1e-9:
                x_pos = map_time_broken(val)
                if x_pos is not None:
                    out.append(f'<line x1="{x_pos:.2f}" y1="{(y_base-7):.2f}" x2="{x_pos:.2f}" y2="{(y_base+7):.2f}" stroke="#4d4d4d" stroke-width="1.1" />')
                    out.append(svg_text(x_pos, y_base + 28.0, f"{int(round(val))}", anchor='middle', size=tick_px))
                val += lstep

            start = math.ceil(right_start_val / rstep) * rstep
            start = round(start, 6)
            val = start
            ticks_max = math.ceil(total_actual)
            while val <= ticks_max + 1e-9:
                x_pos = map_time_broken(val)
                if x_pos is not None:
                    out.append(f'<line x1="{x_pos:.2f}" y1="{(y_base-7):.2f}" x2="{x_pos:.2f}" y2="{(y_base+7):.2f}" stroke="#4d4d4d" stroke-width="1.1" />')
                    out.append(svg_text(x_pos, y_base + 28.0, f"{int(round(val))}", anchor='middle', size=tick_px))
                val = round(val + rstep, 6)

            # Optional summary only if user provides text
            if args.summary_text:
                mono_family = 'DejaVu Sans Mono, Menlo, Consolas, monospace'
                x_center = (pad_l + axis_end_line) / 2.0
                out.append(svg_text(x_center, y_base + 50.0, args.summary_text, anchor='middle', size=12, family=mono_family))

    # Render requested number of copies stacked vertically
    for i in range(copies):
        y_offset = i * (per_copy_height + copy_gap)
        render_one(y_offset, i)

    out.append('</svg>')

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print(out_path)

if __name__ == '__main__':
    main()
