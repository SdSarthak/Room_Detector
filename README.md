# Room Detector — ESP32 CSI indoor room classification

Identify which room a device is in from WiFi **Channel State Information (CSI)** alone — no GPS,
no beacons, no extra infrastructure beyond one ESP32 and an existing WiFi network.

Every WiFi packet an ESP32 receives carries a per-subcarrier estimate of the radio channel.
Walls, furniture and geometry shape that channel in a way that is stable within a room and
different between rooms. This project turns that raw CSI stream into a labelled fingerprint
dataset, trains a classifier on it, and runs live room prediction off the serial port.

```
ESP32 (CSI firmware) --serial--> collect --> data/<room>/*.csv
                                                    |
                                    parse -> clean -> window -> features
                                                    |
                                          train --> models/*.joblib
                                                    |
                                  predict <--serial-- live CSI stream
```

## Status

Working end to end on simulated captures and on the real example capture shipped with
ESP32-CSI-Tool. **No trained model or dataset is included** — CSI fingerprints are specific to one
physical building, so a model is only meaningful if you record it yourself. See
[Getting data](#getting-data).

---

## Install

```bash
git clone --recurse-submodules https://github.com/SdSarthak/Room_Detector.git
cd Room_Detector

# if you already cloned without --recurse-submodules:
git submodule update --init --recursive

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                  # optional, gives you the `room-detector` command
```

Requires Python 3.9+. `ESP32-CSI-Tool/` is a submodule pinned to
[StevenMHernandez/ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) — it is the
firmware, and it is not modified by this project.

Copy `.env.example` to `.env` and set at least your serial port, or pass
`--config config.example.yaml`, or override on the command line. All three are optional; the
defaults work.

---

## Try it without hardware

```bash
python examples/generate_demo_data.py     # writes simulated captures to data/
python -m room_detector train
python -m room_detector predict --file data/kitchen/capture_00.csv
```

The simulator gives each room a distinct multipath profile, so the rooms are separable *by
construction* — near-perfect accuracy here means the pipeline is wired up correctly, and says
nothing about how well it would do in your house. Use it to exercise the code, not to estimate
real accuracy.

---

## Real hardware

### 1. Flash the firmware

Follow [`ESP32-CSI-Tool/README.md`](ESP32-CSI-Tool/README.md). In short, with ESP-IDF installed:

```bash
cd ESP32-CSI-Tool/active_sta      # receiver; use active_ap on the other board
idf.py menuconfig                 # set WiFi SSID/password, enable "Should Collect CSI"
idf.py build flash
```

You need **two** boards: one transmitting (`active_ap`) and one receiving (`active_sta`), or one
receiver in `passive` mode listening to existing WiFi traffic. Only the receiver is connected to
your computer. Keep the transmitter in a fixed position — moving it invalidates every fingerprint
you have recorded.

### 2. Record one capture per room

```bash
python -m room_detector collect --room kitchen --port COM3 --seconds 120
python -m room_detector collect --room bedroom --port COM3 --seconds 120
python -m room_detector collect --room hallway --port COM3 --seconds 120
```

Record **at least two separate captures per room**, ideally on different days, walking around
normally. Cross-validation groups by capture file, so a single capture per room cannot be
validated honestly.

If you would rather use the tool's own serial scripts, anything that produces the raw lines works:

```bash
idf.py monitor | python -m room_detector collect --room kitchen --stdin
```

### 3. Train and evaluate

```bash
python -m room_detector train
```

```
36 windows x 474 features from 6 captures across 3 rooms
  bedroom                12 windows
  hallway                12 windows
  kitchen                12 windows

5-fold cross-validation (grouped by capture) over 36 windows
accuracy: 0.9167
confusion matrix (rows = true, cols = predicted):
           bedroo hallwa kitche
  bedroom      11      1      0
  hallway       0     12      0
  kitchen       1      1     10
```

*(Illustrative output shape — your numbers depend entirely on your building.)*

### 4. Predict live

```bash
python -m room_detector predict --port COM3
```

```
[ 4501823.114] room=kitchen          confidence=0.883 (window vote of 64 packets)
[ 4501824.771] room=kitchen          confidence=0.907 (window vote of 64 packets)
```

---

## Commands

| Command | What it does |
| --- | --- |
| `collect --room NAME` | Record a labelled capture from serial (`--port`) or stdin (`--stdin`). `--seconds` / `--packets` bound the run. |
| `inspect PATH` | Summarise a capture: packet count, subcarriers, rate, RSSI, how many training windows it yields. |
| `train` | Build the dataset, cross-validate, fit, and save the model. |
| `evaluate` | Cross-validate only; writes nothing. |
| `predict` | Classify a live serial stream (`--port`), stdin (`--stdin`), or a recorded file (`--file`). |

Global flags: `--config FILE.yaml`, `--data-dir DIR`, `--model-path PATH`.
Run `python -m room_detector <command> --help` for the rest.

---

## Getting data

There is no public CSI dataset that will work here — a fingerprint is only valid for the building
it was recorded in, with the transmitter in the position it was recorded from. Your options:

1. **Record your own** (the real path): steps above. Budget ~2 minutes per room per capture.
2. **Simulate** (`examples/generate_demo_data.py`): exercises the pipeline, teaches you nothing
   about your building.
3. **The upstream example capture**: `ESP32-CSI-Tool/python_utils/example_csi.csv`, 13 packets from
   real hardware. Enough to test the parser, far too small to train on.

Captures are plain CSV and stay out of git (`data/` and `*.csv` are ignored).

---

## How it works

### Parsing (`room_detector/csi.py`)

The firmware prints one line per received packet: 25 scalar fields then a bracketed array of
signed bytes interleaving the **imaginary and real** part of each subcarrier.

```
CSI_DATA,AP,3C:71:BF:6D:2A:78,-73,11,1,...,384,[101 -48 5 0 0 0 ... ]
```

Two things make this less trivial than `line.split(",")`:

- The trailing `len` field is unreliable. The upstream example capture reports both `128` and
  `384` on packets that all carry 64 subcarriers, because an LLTF-only build truncates the buffer
  but not the reported length. The parser always sizes a packet from the array it actually got.
- Serial output interleaves boot logs, non-UTF-8 bytes and half-written lines cut off by a reset.
  Parsing skips those instead of crashing a two-minute recording.

Amplitude is `sqrt(imag² + real²)` and phase is `atan2(imag, real)`, matching
`ESP32-CSI-Tool/python_utils/parse_csi.py` exactly — there is a test asserting that.

### Cleaning (`room_detector/preprocessing.py`)

- **Subcarrier selection** — of the 64 bins in a 20 MHz LLTF symbol, only 52 carry data; guard
  bands and DC are constant zero. Bins that are dead across the whole dataset are dropped too.
- **Hampel filter** — replaces impulsive outliers (AGC steps, collisions) with the local median,
  using a MAD-based robust scale.
- **Phase sanitization** — raw CSI phase is dominated by a linear ramp from sampling time offset
  and carrier frequency offset, which changes packet to packet *even when nothing moves*. Fitting
  and subtracting that ramp is what makes phase usable at all; without it phase features are noise.
- **Smoothing / normalisation** — optional moving average and per-packet L2/z-score/min-max scaling.

### Features (`room_detector/features.py`)

A single packet is far too noisy to classify. Packets are grouped into sliding windows (64 packets,
step 32 by default) and each window becomes one sample:

| Block | Per subcarrier | Meaning |
| --- | --- | --- |
| Amplitude | mean, std, median, min, max, IQR | the static multipath signature of the location |
| Motion | mean absolute packet-to-packet change | how much the channel fluctuates there |
| Phase | mean, std (after sanitization) | multipath structure amplitude alone misses |
| RSSI | mean, std, min, max, noise floor, SNR | coarse distance to the transmitter |

With 52 surviving subcarriers that is 474 features per window.

### Model (`room_detector/model.py`)

`StandardScaler` + one of random forest (default), extra trees, SVM, k-NN or logistic regression.

Cross-validation uses **`GroupKFold` grouped by capture file**. This matters more than the choice
of classifier: consecutive windows overlap and come from the same few seconds of radio conditions,
so a plain shuffled split puts near-duplicate samples on both sides and reports accuracy far above
what the model achieves on a genuinely new capture.

The saved bundle contains the estimator *and* the preprocessing decisions (which subcarriers
survived, which window size was used), because a live stream has to be reduced to exactly the
feature layout the model was trained on.

### Live prediction (`room_detector/stream.py`)

A ring buffer of the last `window_size` packets is re-classified every `window_step` packets, then
smoothed by majority vote over the last few windows so a single bad window does not flip the
reported room. `min_confidence` suppresses low-confidence output entirely.

---

## Library use

```python
from room_detector import Config, build_dataset, RoomClassifier

config = Config(data_dir="data")
dataset = build_dataset(config)
print(dataset.summary())

model = RoomClassifier(config)
print(model.evaluate(dataset.X, dataset.y, dataset.groups))
model.fit_dataset(dataset).save("models/room_classifier.joblib")
```

```python
from room_detector import RoomClassifier
from room_detector.stream import RealtimeRoomDetector, serial_lines

model = RoomClassifier.load("models/room_classifier.joblib")
detector = RealtimeRoomDetector(model)
for prediction in detector.process(serial_lines("COM3", 921600)):
    print(prediction.room, prediction.confidence)
```

---

## Tests

```bash
python -m pytest
```

137 tests, no hardware and no downloads required. They cover the parser against the real upstream
capture, the cleaning maths (a pure phase ramp must sanitize to zero; a spike must be replaced by
the local median), feature/name alignment, dataset grouping, model round-tripping, and every CLI
command including its failure paths.

---

## Tuning

| Symptom | Try |
| --- | --- |
| Rooms confused with each other | Longer `window_size`; record more captures per room; check the transmitter has not moved |
| Accuracy collapses days later | Re-record — furniture and door positions changed the channel |
| Live output flickers | Raise `vote_window` and `min_confidence` |
| Very few training windows | Longer captures, or lower `window_step` |
| `collect` writes 0 packets | Firmware not built with `CONFIG_SHOULD_COLLECT_CSI`, or no transmitter active |

---

## Limitations

Honest ones, not marketing:

- **Room level only.** This tells you *which room*, not where in it. No triangulation, no
  coordinates, no multi-floor support.
- **A fingerprint is not portable.** A model trained in one flat is worthless in another, and
  degrades in the same flat as furniture moves. Expect to re-record periodically.
- **The transmitter must stay put.** Move the AP and every fingerprint is invalidated.
- **2.4 GHz only**, inherited from the ESP32 CSI API.
- **One receiver.** Multiple receivers would improve accuracy; combining them is not implemented.
- **Accuracy is environment-specific.** This repo ships no real-world benchmark, and any number
  quoted without naming the building it was measured in is meaningless.

---

## Layout

```
room_detector/
  csi.py            parse the firmware's serial/CSV output
  preprocessing.py  subcarrier selection, outlier filtering, phase sanitization
  features.py       sliding windows -> fingerprint vectors
  dataset.py        data/<room>/*.csv -> X, y, groups
  model.py          classifier pipeline, grouped cross-validation, persistence
  stream.py         serial/stdin/file sources, live windowed prediction
  simulate.py       synthetic captures for testing without hardware
  config.py         all settings, from env / YAML / CLI
  cli.py            collect | inspect | train | evaluate | predict
tests/              137 tests
examples/           demo data generator
ESP32-CSI-Tool/     firmware submodule (upstream, unmodified)
```

## License

MIT. `ESP32-CSI-Tool/` is licensed separately by its authors — see
[`ESP32-CSI-Tool/LICENSE`](ESP32-CSI-Tool/LICENSE).

## Acknowledgments

Firmware and CSI extraction by [Steven M. Hernandez](https://github.com/StevenMHernandez) —
[ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool).
