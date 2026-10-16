<p align="center"><img src="docs/banner.png" alt="ArmExecuTorchEEG — Brainwaves into action" width="100%"></p>

# Edge_Deploy

**Hackathon submission repo.** Live motor-imagery EEG → Alif Ensemble E8 → Ethos-U85 inference → predicted hand, with IMU-based artifact gating. Judged release: tag `v1.0-submission`.

> Arm + ExecuTorch Edge AI Challenge · SFSU · Oct 17–18 2026 · Team ArmExecuTorchEEG

## What it does

A stroke-rehab brain-computer interface. The user imagines moving the left or right hand; the board classifies which, in real time, on the NPU. If the board's accelerometer sees head movement, the prediction is suppressed instead of trusting a motion-corrupted window.

## Architecture

<p align="center"><picture>
<source media="(prefers-color-scheme: dark)" srcset="docs/media/pipeline-dark.svg">
<img src="docs/media/pipeline.svg" alt="Pipeline: EEG source and host bridge on the laptop, UART into the Alif E8, Cortex-M55 preprocessing, TinyEEGNet int8 on the Ethos-U85, IMU motion gate, LEDs and JSON out to the demo UI" width="900">
</picture></p>

### Workload placement

| Work | Where | Why |
|---|---|---|
| EEG acquisition | Headset → laptop | Headset is Bluetooth/USB; board has neither radio nor the vendor driver |
| Framing to UART | Laptop (`bridge.py`) | Thin: no filtering, no ML |
| Band-pass, normalize, window | Cortex-M55 | Must be bit-identical to training preprocessing; runs in < 1 ms |
| Inference | Ethos-U85 | 100 % of conv/linear ops delegated |
| IMU read + motion gate | Cortex-M55 | On-board ICM-42670 over I2C |
| Display | Laptop | Not scored; keeps board workload honest |

**Replay mode** (`--source replay`) feeds recorded BCI IV 2a trials through the same UART path. Same firmware, no headset required. Used for development and as demo fallback.

## Input

| Source | Format into the board |
|---|---|
| Live: EEG headset via `bridge.py --source <device>` | UART frames: `0xAA55, uint8 n_ch, uint16 seq, int16[n_ch] µV×10`, 250 Hz |
| Replay: `bridge.py --source replay --subject 1 --session 2` | Same frames from `.npz` |
| IMU: on-board ICM-42670 | 100 Hz accel, read by firmware |

Channel count is configured at build time (`CONFIG_N_CH`, default 22). Fewer channels than training → zero-filled, accuracy drops as per Experiment 2 in `Reproduce_Baselines`.

## Output

| Channel | Content |
|---|---|
| UART → laptop | `{"t":ms,"cls":0|1,"conf":0.xx,"lat_us":n,"motion":0|1}` every 250 ms (window stride) |
| LEDs | green left, blue right, red motion-gated, off idle |
| `results/board_metrics.md` | Latency (preproc, NPU, total), arena bytes, `.pte` size, power if measured |
| `results/live_accuracy.md` | Replay-mode accuracy on board vs PyTorch (must match), live-session confusion matrix |

## Expectations

| Metric | Target |
|---|---|
| End-to-end latency, EEG sample in → class out | < 50 ms (excl. headset's own lag) |
| NPU inference | < 10 ms |
| Replay accuracy on board | = PyTorch int8 accuracy, ±0 |
| Live accuracy, one subject, 40 cued trials | > 60 % (realistic for an uncalibrated session) |
| Motion gate | Red LED within 100 ms of a head shake; no false gate while still |
| Uptime | 30 min demo loop without reset |

## Steps

### A. Build and flash (board)

```bash
git clone https://github.com/ArmExecuTorchEEG/Edge_Deploy && cd Edge_Deploy
./tools/fetch_model.sh v0.2-fvp          # pulls .pte + preprocess_params.json from Emulator_Deploy release
```
1. Tools: VS Code + Keil Studio Pack, Alif SETOOLS ≥ 1.110, SEGGER J-Link ≥ 8.42 (starter README §1–2).
2. Board: USB-C on **PRG USB**, jumpers JP5 1-2, JP7 3-4. SW4 → **SEUART**.
3. First time only: Terminal > Run Task > *Alif: Install M55_HP debug stubs*.
4. SW4 → **UART4**. CMSIS view > Manage Solution > target-type **DevKit-E8** > Apply.
5. Build > Run. Console at 115200 shows `READY n_ch=22 model=tinyeegnet_int8`.

### B. Live run

```bash
pip install -r host/requirements.txt
python host/bridge.py --source openbci --port /dev/tty.usbserial-XXXX --board /dev/tty.usbmodemYYYY
python host/demo_ui.py --board /dev/tty.usbmodemYYYY       # second terminal
```

### C. Replay run (no headset)

```bash
python host/bridge.py --source replay --subject 1 --session 2 --board /dev/tty.usbmodemYYYY
```

### D. Verify numbers

```bash
python host/measure.py --board ... --trials 200 > results/board_metrics.md
```

## Layout

```
firmware/
  src/        main.cpp, uart_rx.cpp, preprocess.cpp (IIR from params.json), imu.cpp, infer.cpp, leds.cpp
  ai_layer/   generated
  model/      .pte, params.json (fetched)
  tests/      preprocess_test.cpp — compares against reference vectors
host/
  bridge.py   headset or replay → UART frames
  demo_ui.py  live plot, class, confidence, latency, motion flag
  measure.py  latency and accuracy harness
  drivers/    openbci.py, muse.py (LSL), replay.py
results/
docs/
  architecture.md   this diagram, expanded
  demo_script.md    the 3-minute judged demo, step by step, with fallback branches
  video/            backup recording of the full demo
```

## Submission checklist (rules)

| Requirement | Where |
|---|---|
| Public repo | this |
| Fixed release | tag `v1.0-submission`, with `.pte` and firmware `.axf` as assets |
| Build, deploy, run instructions | Steps A–C above |
| Architecture and components | this README + `docs/architecture.md` |
| Workload placement | table above |
| Performance measurements | `results/board_metrics.md` |
| Live demo on Alif E8 | `docs/demo_script.md`; video backup in `docs/video/` |
| English | yes |

## Judging map

| Criterion (pts) | Evidence |
|---|---|
| Functionality & reliability (20) | 30-min uptime, replay accuracy = PyTorch |
| Performance & efficiency (20) | `board_metrics.md`: NPU ms, arena, `.pte` KB |
| Effective hardware use (15) | All ML on Ethos-U85; CPU does only DSP and I/O |
| Multimodal & sensor integration (15) | EEG + on-board IMU fusion |
| Innovation & use-case (10) | Stroke rehab BCI on a $100-class MCU |
| Technical ambition (10) | Bit-exact on-board preprocessing, quantized pipeline from dataset to NPU |
| Engineering quality & demo (10) | Tests, release discipline, scripted demo, fallbacks |

## Team

| Name | Role in this repo |
|---|---|
| Moamen Soliman | Firmware DSP, UART path, IMU gate |
| Jihyung Ku | Model, replay harness |
| | |

## License

MIT. Starter code © Arm Limited under its original license, see `LICENSE-ARM`.

## Status

| Step | Done |
|---|---|
| Builds for DevKit-E8 | |
| Replay mode end-to-end on board | |
| Live headset frames arriving | |
| Live prediction | |
| IMU gate | |
| Demo UI | |
| Metrics recorded | |
| Demo rehearsed twice, video recorded | |
| Tag `v1.0-submission` | |
