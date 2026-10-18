"""Generate Figure 2: the preprocessing chain inside the PREPROCESSING card of pipeline.svg.

Same glass style as make_pipeline_svg.py. Numbers come from firmware/model/params.json (v0.5-distill).
Run: python docs/media/make_signal_svg.py docs/media
"""
import sys
from pathlib import Path

from make_pipeline_svg import ICONS, THEMES, esc

W, H = 1240, 560
CW, CH = 208, 150
COLS = [24, 270, 516, 762, 1008]
ROW1, ROW2 = 70, 340

HUES = {"in": "6,182,212", "sample": "244,63,94", "window": "234,120,40", "npu": "245,158,11"}

# (col, row_y, hue, step, title, formula, note lines); step "" = icon card
CARDS = [
    (0, ROW1, "in", "wave", "UART IN", "22 × int16 counts", ["one frame per sample, 250 Hz", "from bridge.py"]),
    (1, ROW1, "sample", "1", "DECODE", "µV = count × 0.0488", ["exact BCI IV 2a ADC step,", "lossless"]),
    (2, ROW1, "sample", "2", "CAR", "x − mean of 22 ch", ["common average reference:", "removes the shared reference"]),
    (3, ROW1, "sample", "3", "BAND-PASS", "7–30 Hz · 4 biquads", ["causal, filter state kept;", "keeps mu and beta rhythms"]),
    (4, ROW1, "sample", "4", "DECIMATE", "keep every 2nd sample", ["250 Hz → 125 Hz;", "band-pass is the anti-alias"]),
    (0, ROW2, "window", "5", "WINDOW", "22 × 256 samples", ["2.048 s at 125 Hz,", "cue + 0.5 s to + 2.5 s"]),
    (1, ROW2, "window", "6", "ALIGN (EA)", "X ← A_subject · X", ["22 × 22 matrix per subject,", "from 24 unlabeled trials"]),
    (2, ROW2, "window", "7", "SCALE", "X ← X / 0.969", ["unit variance, same as", "the training data"]),
    (3, ROW2, "window", "8", "QUANTIZE", "round(X/0.0696) − 12", ["int8, clipped to −128…127,", "exactly as on the host"]),
    (4, ROW2, "npu", "npu", "ETHOS-U85", "TinyEEGNet-wide", ["int8 22 × 256 → 2 logits", "left / right"]),
]

BANDS = [
    (ROW1, "sample", "M55 · PER SAMPLE", "every 4 ms, 250 Hz", "≈ 250 cycles = 0.6 µs at 400 MHz"),
    (ROW2, "window", "M55 · PER WINDOW", "once per trial (or every 256 ms)", "≈ 100 k cycles = 0.25 ms at 400 MHz"),
]


def glass(t, hue, x, y, w, h, rx, fill=None):
    c = HUES[hue]
    fa = t["fill_a"] if fill is None else fill
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="rgba({c},{fa})" '
            f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.5"/>'
            f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{w - 3}" height="{h * 0.45}" rx="{rx - 1.5}" fill="url(#sheen)"/>')


def flow(t, d):
    return (f'<path d="{d}" fill="none" stroke="{t["arrow"]}" stroke-width="2.2" stroke-dasharray="5 5" '
            f'marker-end="url(#ah)"><animate attributeName="stroke-dashoffset" from="10" to="0" dur="0.6s" '
            f'repeatCount="indefinite"/></path>')


def card(t, col, y, hue, step, title, formula, notes):
    x = COLS[col]
    c = HUES[hue]
    p = [glass(t, hue, x, y, CW, CH, 16)]
    if step.isdigit():
        p.append(f'<circle cx="{x + 30}" cy="{y + 30}" r="14" fill="rgba({c},{t["chip_a"]})" '
                 f'stroke="rgba({c},{t["edge_a"]})" stroke-width="1.2"/>'
                 f'<text x="{x + 30}" y="{y + 35}" font-size="14" font-weight="800" text-anchor="middle" '
                 f'fill="{t["title"]}">{step}</text>')
    else:
        p.append(f'<g transform="translate({x + 17},{y + 17}) scale(1.1)" fill="none" stroke="{t["stroke"]}" '
                 f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
                 f'{ICONS[{"wave": "wave", "npu": "npu"}[step]].replace("{stroke}", t["stroke"])}</g>')
    p.append(f'<text x="{x + 54}" y="{y + 36}" font-size="15" font-weight="700" letter-spacing="0.4" '
             f'fill="{t["title"]}">{esc(title)}</text>')
    p.append(f'<rect x="{x + 14}" y="{y + 56}" width="{CW - 28}" height="30" rx="8" '
             f'fill="rgba({c},{t["chip_a"] * 0.6:.2f})"/>')
    p.append(f'<text x="{x + CW / 2}" y="{y + 76}" font-size="13" font-weight="600" text-anchor="middle" '
             f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" '
             f'fill="{t["tools"]}">{esc(formula)}</text>')
    for i, line in enumerate(notes):
        p.append(f'<text x="{x + 16}" y="{y + 110 + 18 * i}" font-size="12.5" fill="{t["role"]}">{esc(line)}</text>')
    return "".join(p)


def build(theme):
    t = THEMES[theme]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         "font-family=\"Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif\">",
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["arrow"]}"/></marker>'
         '<linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0" stop-color="#ffffff" stop-opacity="{t["sheen_a"]}"/>'
         '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient></defs>']

    # band frames behind each row, steps only (the UART and NPU cards sit outside the M55 work)
    for y, hue, name, when, cost in BANDS:
        c = HUES[hue]
        x0 = COLS[1] - 12 if y == ROW1 else COLS[0] - 12
        x1 = COLS[4] + CW + 12 if y == ROW1 else COLS[3] + CW + 12
        p.append(f'<rect x="{x0}" y="{y - 46}" width="{x1 - x0}" height="{CH + 58}" rx="20" '
                 f'fill="rgba({c},0.04)" stroke="rgba({c},0.5)" stroke-width="1.2" stroke-dasharray="6 5"/>')
        tx = x0 + 18 if y == ROW1 else COLS[0] + CW / 2 + 18
        p.append(f'<text x="{tx}" y="{y - 19}" font-size="14" font-weight="700" letter-spacing="0.6" '
                 f'fill="{t["title"]}">{esc(name)}</text>'
                 f'<text x="{tx + 11 * len(name) + 10}" y="{y - 19}" font-size="13" fill="{t["role"]}">'
                 f'{esc(when)}</text>'
                 f'<text x="{x1 - 18}" y="{y - 19}" font-size="13" text-anchor="end" fill="{t["role"]}">'
                 f'{esc(cost)}</text>')

    for c in CARDS:
        p.append(card(t, *c))

    mid1, mid2 = ROW1 + CH / 2, ROW2 + CH / 2
    for col in range(4):
        x1 = COLS[col] + CW
        p.append(flow(t, f"M{x1 + 4},{mid1} H{COLS[col + 1] - 4}"))
        p.append(flow(t, f"M{x1 + 4},{mid2} H{COLS[col + 1] - 4}"))
    # end of row 1 back to the start of row 2 through the gap between the rows
    gy = ROW1 + CH + (ROW2 - 46 - ROW1 - CH) / 2 + 4
    p.append(flow(t, f"M{COLS[4] + CW / 2},{ROW1 + CH + 4} V{gy} H{COLS[0] + CW / 2} V{ROW2 - 4}"))
    p.append(f'<text x="{COLS[2] + CW / 2}" y="{gy - 8}" font-size="12.5" text-anchor="middle" '
             f'fill="{t["label"]}">ring buffer 22 × 512 at 125 Hz</text>')

    p.append("</svg>")
    return "\n".join(p) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "signal.svg").write_text(build("light"))
    (out / "signal-dark.svg").write_text(build("dark"))
    print("wrote", out / "signal.svg", out / "signal-dark.svg")
