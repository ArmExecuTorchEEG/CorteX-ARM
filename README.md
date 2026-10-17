<p align="center"><img src="docs/banner.png" alt="ArmExecuTorchEEG — Brainwaves into action" width="100%"></p>

# Edge_Deploy

**Hackathon submission repo.** Motor-imagery EEG streamed in real time → Alif Ensemble E8 → Ethos-U85 inference → predicted hand, with IMU-based artifact gating. Judged release: tag `v1.0-submission`.

> Arm + ExecuTorch Edge AI Challenge · SFSU · Oct 17–18 2026 · Team ArmExecuTorchEEG

## What it does

A stroke-rehab brain-computer interface. The user imagines moving the left or right hand; the board classifies which, in real time, on the NPU. If the board's accelerometer sees movement, the prediction is suppressed instead of trusting a motion-corrupted window.

**No headset.** The team has no EEG hardware. The demo streams held-out BCI IV 2a trials (session 2, never seen in training) from the laptop at 250 Hz, sample by sample, over the same UART path a headset would use. The firmware cannot tell the difference, so a headset driver can be added later without touching the board. The IMU is on the board, so it measures board motion: in a wearable build the board sits on the head; in the demo we move the board to show the gate.

## Architecture

<p align="center"><picture>
<source media="(prefers-color-scheme: dark)" srcset="docs/media/pipeline-dark.svg">
<img src="docs/media/pipeline.svg" alt="Pipeline: BCI IV 2a replay and host bridge on the laptop, UART into the Alif E8, Cortex-M55 preprocessing, TinyEEGNet int8 on the Ethos-U85, IMU motion gate, LEDs and JSON out to the demo UI" width="900">
</picture></p>

One `.pte` (TinyEEGNet-wide student wide32_ft, release `v0.5-distill`, 11,456 B, 125 Hz input, 256 samples = 2.048 s; 74.6 % int8 mean over 9 subjects and 3 seeds in Reproduce_Baselines) serves all nine BCI IV 2a subjects; `SET_SUBJECT` only switches the subject's 22×22 Euclidean-alignment matrix. The previous model, `v0.3-distill`, has the same contract (shapes, params schema, UART) and still fetches with `./tools/fetch_model.sh v0.3-distill`. Block diagram, per-sample pipeline, memory map and timing budget: [`docs/architecture.md`](docs/architecture.md).

### Workload placement

| Work | Where | Why |
|---|---|---|
| EEG source | Laptop (`bridge.py --source replay`) | No headset; recorded trials paced at 250 Hz like a live stream |
| Framing to UART | Laptop (`bridge.py`) | Thin: no filtering, no ML |
| CAR, band-pass (4 SOS), decimate to 125 Hz, window | Cortex-M55, per sample, float64 | Must match `host/pipeline_ref.py` bit for bit; 243 cycles/sample (FVP) |
| Euclidean alignment (per subject), scale, int8 quantize | Cortex-M55, per window | 99.7 k cycles ≈ 0.25 ms at 400 MHz (FVP) |
| Inference | Ethos-U85 | All conv/linear ops delegated; 128 k NPU cycles per window (FVP). The `.pte` I/O is float, quantize/dequantize run on the M55 |
| IMU read + motion gate | Cortex-M55 | On-board ICM-42670-P over I3C, polled at 100 Hz |
| Display, metrics | Laptop (`demo_ui.py`, `measure.py`) | Not scored; keeps board workload honest |

**Replay mode** (`--source replay`) is the only EEG source: it feeds recorded BCI IV 2a trials through the UART path in real time. It is used for development and for the judged demo.

## Input

UART4 (the board's stdio USART, SW4 → UART4, PRG USB), **921600 baud** 8N1. 250 Hz × 49 B × 10 bit = 122.5 kbit/s does not fit in 115200. Little endian.

| Frame | Bytes | Meaning |
|---|---|---|
| Data | `0xAA 0x55, uint8 n_ch (22), uint16 seq, int16[n_ch]` | One 250 Hz sample. Value = µV × 20.48 (the native BCI IV 2a step 100/2048 µV, so lossless; range ±1600 µV). The firmware decodes µV = count × 0.048828125. Channels in `channel_order` of `params.json`. `seq` +1 per sample, wraps at 65536 |
| Control | `0xAA 0x56, uint8 kind, uint8 arg, uint16 seq` | kind 1 `SET_SUBJECT` (arg 1–9, required before the first cue); 2 `CUE` (arg = true label 0 left, 1 right, 255 unknown; the cue sample is the data frame with this `seq`, sent right after); 3 `RESET` (filter state, counters, ring buffer, pending cues); 4 `FREE_RUN` (arg 0/1) |

A session is `RESET`, `SET_SUBJECT`, then a continuous stream from `--lead-in` (10 s) before the first replayed cue, with no `seq` gaps. The replay data are `host/data/A0{s}_session2.npz` (git-ignored), written once by `host/export_replay.py`.

| Source | Into the board |
|---|---|
| EEG: `bridge.py --source replay --subject 3 --session 2` | Frames above, paced at 250 Hz |
| IMU: on-board ICM-42670-P | Accel, ±4 g at 100 Hz ODR (own I3C driver in `firmware/src/imu.cpp`, every transfer bounded to 5 ms), polled at 100 Hz |

## Output

| Channel | Content |
|---|---|
| UART → laptop, at boot | `READY n_ch=22 model=tinyeegnet_int8 pte_bytes=11456 fs=250 fs_model=125 win=256 baud=921600` |
| UART → laptop, per inference | `{"t":ms,"seq":n,"trial":i,"subject":s,"cls":0\|1\|-1,"raw_cls":0\|1,"conf":0.xx,"logits":[l0,l1],"lat_us":n,"motion":0\|1,"label":0\|1\|-1}`, plus `"imu":0` when the IMU is not running |
| UART → laptop, diagnostics | lines starting with `#` (`# imu ok …`, `# frame_err n`, gap fills) |
| LEDs | green left, blue right, red motion-gated, off idle (held 500 ms) |
| `results/board_metrics.md` | Accuracy, agreement, `lat_us` mean/p50/p95/max, gated count (`host/measure.py`) |
| `results/replay_accuracy.md` | Host-side ground truth for all 9 subjects: offline int8, sim replay, expected board output |

One inference per cue, when the 125 Hz window `[cue_125 + 62, cue_125 + 318)` is complete (window end = cue + 2.544 s; `cue_125 = n_250 // 2`). With `FREE_RUN 1` also one every 256 ms on the newest window (`"trial":-1`). `seq` is the last sample of the window, `trial` the cue index since `RESET`. `raw_cls` = argmax(logits), `conf` = its softmax probability, `cls` = −1 when `motion` = 1, else `raw_cls`. `lat_us` runs from the last UART byte of the window's last sample (stamped in the UART ISR) to just before the JSON line is formatted; it excludes formatting and queueing (a few µs) and is chunk-granular (up to ~90 µs). `label` echoes the `CUE` arg.

Motion gate: `motion` = 1 if, within the 500 ms before the inference, `max | |a| − 1 g | > 0.25 g` or any axis changed by more than 0.25 g between consecutive samples (`firmware/src/motion_config.h`).

Only `bridge.py` opens the serial port. It republishes every board line, plus its own `{"event":"cue"}` and other records, as UDP datagrams to `127.0.0.1:5555`; `demo_ui.py` and `measure.py` listen there.

## Expectations

| Metric | Target |
|---|---|
| End-to-end latency, last UART byte of a window in → JSON line queued | < 50 ms (budget ≈ 1 ms, `docs/architecture.md`) |
| NPU inference | < 10 ms (128 k NPU cycles ≈ 0.32 ms at 400 MHz, FVP) |
| Preprocessing on board | int8 windows equal `host/pipeline_ref.py` bit for bit (`make -C firmware/tests test`, FVP self-test) |
| Replay accuracy | With the ×20.48 encoding the reference equals the offline int8 predictions on 1080/1080 trials (`results/replay_accuracy.md`); mean 74.4 % (shipped seed 0), A03 99.2 %, A08 94.2 %. Board argmax must match `host/reference/expected_board_predictions.json` |
| Motion gate | Red LED within 100 ms of shaking the board; no false gate while still |
| Uptime | 30 min demo loop without reset |

## Steps

Clone and fetch the model bundle (needs `gh` with read access to the org):

```bash
git clone https://github.com/ArmExecuTorchEEG/Edge_Deploy && cd Edge_Deploy
./tools/fetch_model.sh                       # default v0.5-distill from Reproduce_Baselines; checks sha256, writes firmware/model/params.json
# ./tools/fetch_model.sh --local ../Reproduce_Baselines/artifacts/v4   # same bundle from a local checkout (no sha256 check)
python3 tools/gen_params_h.py --check        # firmware/src/params_gen.h matches params.json
```

The AI layer (`firmware/ai_layer/model_pte.c`, the `.pte` as a C array) is committed, so a build needs only the CMSIS toolchain: VS Code + Keil Studio Pack, CMSIS-Toolbox 2.15, ExecuTorch 1.5.1. From the command line, activate it first in `firmware/` with `source ~/.vcpkg/vcpkg-init && vcpkg-shell activate` (the init script defines `vcpkg-shell`, not `vcpkg`).

### A0. Prove the pipeline on the FVP (no board)

The `SelfTest` build replays an embedded A03 stream (5 trials, 13,381 frames) through the real parser, preprocessing, Ethos-U85 and JSON path:

```bash
cbuild firmware/edge_eeg.csolution.yml --context edge_eeg.SelfTest+SSE-320-U85 --packs
cd firmware && docker run --rm -i --init --user $(id -u) --mount "type=bind,src=$HOME/,dst=$HOME/" --workdir "$PWD" \
  --env "ARMLM_CACHED_LICENSES_LOCATION=$HOME/.armlm" cmsis-fvp:11.32.23 stdbuf -oL -eL \
  FVP_Corstone_SSE-320 -f board/Corstone-320/fvp_config.txt --simlimit 600 --stat \
  -a out/edge_eeg/SSE-320-U85/SelfTest/edge_eeg.axf
```

Expected: `Test_result: PASS`, 0 of 28,160 int8 elements differ. Recorded run: [`results/fvp_selftest.md`](results/fvp_selftest.md). On macOS the FVP runs in Docker (colima or Docker Desktop) with a Keil MDK Community licence.

### A. Build and flash (board)

```bash
cbuild firmware/edge_eeg.csolution.yml --active DevKit-E8 --packs
```
or in VS Code: CMSIS view on `firmware/edge_eeg.csolution.yml` > Manage Solution > target-type **DevKit-E8** > Build.

1. Tools: Alif SETOOLS ≥ 1.110, SEGGER J-Link ≥ 8.42 (`firmware/board/DevKit-E8/README.md`).
2. Board: USB-C on **PRG USB**, jumpers JP5 1-2, JP7 3-4. SW4 → **SEUART**.
3. First time only: Terminal > Run Task > *Alif: Install M55_HP debug stubs*.
4. SW4 → **UART4**. CMSIS view > Run.
5. Serial console at **921600** shows `READY n_ch=22 model=tinyeegnet_int8 pte_bytes=11456 fs=250 fs_model=125 win=256 baud=921600`, then `# imu …`. Close the console before starting the bridge.

### B. Demo run (replay)

```bash
pip install -r host/requirements.txt
python host/export_replay.py --subject all          # once; needs the Reproduce_Baselines env and ~/mne_data
python host/demo_ui.py --listen 5555                 # terminal 1 (--headless for text only)
python host/bridge.py --source replay --subject 3 --session 2 --board /dev/tty.usbmodemXXXX   # terminal 2
python host/bridge.py --source replay --subject 3 --session 2 --board sim                    # no board: in-process reference
```
Default trials 24–143 (0-based), 10 s lead-in, `--free-run` for free-run mode. The full judged run, with fallbacks, is [`docs/demo_script.md`](docs/demo_script.md).

### C. Verify numbers

```bash
python host/measure.py --listen 5555 --trials 120 --out results/board_metrics.md   # during a bridge run
python host/replay_accuracy.py                       # host-side table -> results/replay_accuracy.md
make -C firmware/tests test                          # firmware C++ vs pipeline_ref, host compiler
python -m pytest host/tests -q                       # host tests (Reproduce_Baselines/.venv has every dependency)
```

## Layout

```
firmware/
  edge_eeg.csolution.yml, edge_eeg.cproject.yml   targets SSE-320-U85 (FVP) and DevKit-E8; build types Debug and SelfTest
  src/        main, app (cue bookkeeping, gap fill, free run), uart_rx (frame parser), preprocess (CAR, SOS, decimate, EA, quantize),
              infer (ExecuTorch), imu, motion_gate + motion_config.h, leds, json_out, clock, selftest,
              params_gen.h (generated by tools/gen_params_h.py from model/params.json)
  board/      DevKit-E8/ and Corstone-320/ board layers (UART4 stdio at 921600, I3C IMU, LEDs, NPU)
  ai_layer/   generated: the .pte as a C array (model_pte.c)
  model/      fetched: params.json (committed), .pte, test_vectors.npz, float checkpoint (ignored)
  tests/      preprocess_test.cpp + Makefile; generated/ written by host/make_vectors.py
host/
  bridge.py         replay -> UART frames at 250 Hz, republish board output over UDP 5555
  demo_ui.py        live window (or --headless): EEG trace, class, confidence, latency, gate
  measure.py        metrics from the UDP feed or a bridge --log JSONL
  replay_accuracy.py, pipeline_ref.py (bit-exact reference), int8_oracle.py, make_vectors.py, export_replay.py, common.py, listen.py
  drivers/          replay.py, sim_board.py (--board sim)
  reference/        offline_int8_predictions.json, expected_board_predictions.json
  tests/            pytest
  data/             A0{s}_session2.npz replay streams (git-ignored)
tools/      fetch_model.sh, gen_params_h.py
results/    fvp_selftest.md (+ .log), replay_accuracy.md, board_metrics.md (once measured)
docs/       architecture.md, demo_script.md, banner.png, media/ (pipeline SVGs)
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
| Functionality & reliability (20) | 30-min uptime, board argmax = expected int8 output, bit-exact windows |
| Performance & efficiency (20) | `board_metrics.md`: NPU ms, arena, `.pte` KB |
| Effective hardware use (15) | All ML on Ethos-U85; CPU does only DSP and I/O |
| Multimodal & sensor integration (15) | EEG stream + on-board IMU motion gate, shown by moving the board |
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

MIT. Starter code © Arm Limited, Apache-2.0 (SPDX headers in those files); ExecuTorch runtime license in `firmware/src/LICENSE-ExecuTorch`.

## Status

| Step | Done |
|---|---|
| Builds for DevKit-E8 | x |
| FVP self-test PASS (DSP + NPU chain, 5 trials) | x |
| Replay mode end-to-end on board | |
| Streamed predictions match offline int8 | |
| IMU gate | |
| Demo UI | |
| Metrics recorded | |
| Demo rehearsed twice, video recorded | |
| Tag `v1.0-submission` | |

### Known untested (no board yet)

- UART4 at 921600 in interrupt mode, and whether the PRG USB serial bridge carries 921600.
- ICM-42670-P bring-up over I3C (`firmware/src/imu.cpp`: RSTDAA, SETDASA, WHO_AM_I, ±4 g 100 Hz; each transfer bounded to 5 ms; on failure the board prints `# imu init failed: ...` and streams with `"imu":0`) and its readings.
- RGB LED pins and polarity (driven high = on, as in the pack's Blinky).
- Ethos-U PMU counters on the E8.
- All real timing: `lat_us`, cycles at 400 MHz.
