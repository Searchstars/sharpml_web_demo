# sharpml_demo

This repository now contains a Rust-first SHARP MCU packaging pipeline alongside the original browser prototype.

## Rust workspace

- `crates/sharp-mcu-core`
  Pure Rust preprocessing, ONNX Runtime inference, depth projection, slice extraction, backdrop generation, manifest export, and optional debug PLY export.
- `crates/sharp-mcu-cli`
  Command-line entry point for generating an MCU slice package from a source image.
- `crates/sharp-mcu-preview`
  Preview-only desktop app that reads `manifest.json` plus exported PNG layers, uploads them to GPU textures once, and simulates the virtual gyro drag interaction.

## Model source

The generator now uses a platform-specific default model source:

- Apple hosts (`macOS` while running this CLI): `pearsonkyle/Sharp-coreml / sharp.mlpackage`
- Other hosts: `pearsonkyle/Sharp-onnx / sharp_fp16.onnx`

The reason for the split is practical:

- Apple native CoreML loading avoids the ONNX Runtime CoreML execution provider's subgraph-capture and first-run compile path.
- Non-Apple hosts still use standard ONNX Runtime.
- ONNX Runtime does not provide a generic OpenCL execution provider, so AMD GPU support in this workspace is exposed as `migraphx`, not `opencl`.

If `--model` is omitted, the CLI downloads the default model for the current host through the `hf-hub` Rust crate and reuses the local Hugging Face cache.
In this workspace, the default cache location is `./caches/hf-hub` so model downloads stay inside the repo tree instead of `~/.cache`.
Download source selection follows this priority:

- `--hf-endpoint https://hf-mirror.com`
- `HF_ENDPOINT=https://hf-mirror.com`
- fallback to the official `https://huggingface.co`

Download transport selection is separate:

- `--download-backend hf-hub`
  Pure Rust, no extra system dependency, but single-stream download.
- `--download-backend aria2`
  External `aria2c` multi-connection download, better for multi-GB weights.

Inference backend selection is also configurable:

- `--inference-backend coreml-native`
  Default on Apple hosts. Loads the downloaded `.mlpackage/.mlmodel/.mlmodelc` directly through a small Swift/CoreML bridge and caches the compiled model under `--coreml-cache-dir`.
- `--inference-backend cpu`
  Default on non-Apple hosts. Matches the upstream `inference_onnx.py` behavior and uses plain ONNX Runtime CPU execution for universal compatibility.
- `--inference-backend cuda`
  Explicit NVIDIA GPU path for ONNX models. Intended for Linux/Windows builds with CUDA 12/13 and cuDNN available.
- `--inference-backend migraphx`
  Explicit AMD GPU path for ONNX models. This is the ONNX Runtime AMD execution provider we expose here; there is no generic OpenCL EP.
  In practice, this usually means supplying an ONNX Runtime build/provider library with MIGraphX enabled on the target machine; it is not as plug-and-play as the CUDA prebuilt path.
- `--inference-backend auto`
  Uses `coreml-native` for native CoreML assets and otherwise follows the normal ONNX Runtime path.
- `--inference-backend coreml-neural-engine`
  ONNX Runtime CoreML EP mode. Kept as an explicit advanced option for single-file ONNX on Apple.
- `--inference-backend coreml-gpu`
  ONNX Runtime CoreML EP mode with `CPUAndGPU`.
- `--inference-backend coreml`
  ONNX Runtime CoreML EP mode using all compatible Apple accelerators.

## Generate a package

```bash
cargo run -p sharp-mcu-cli -- generate \
  --input /path/to/photo.jpg \
  --output-dir ./outputs/rust-package
```

Using a mirror explicitly:

```bash
cargo run -p sharp-mcu-cli -- generate \
  --input /path/to/photo.jpg \
  --output-dir ./outputs/rust-package \
  --hf-endpoint https://hf-mirror.com
```

Using the default Apple-native CoreML path on macOS:

```bash
cargo run -p sharp-mcu-cli -- generate \
  --input /path/to/photo.jpg \
  --output-dir ./outputs/rust-package \
  --hf-endpoint https://hf-mirror.com
```

If you want to force ONNX Runtime CPU behavior instead:

```bash
cargo run -p sharp-mcu-cli -- generate \
  --input /path/to/photo.jpg \
  --output-dir ./outputs/rust-package \
  --inference-backend cpu
```

Using multi-connection `aria2`:

```bash
brew install aria2

cargo run -p sharp-mcu-cli -- generate \
  --input /path/to/photo.jpg \
  --output-dir ./outputs/rust-package \
  --hf-endpoint https://hf-mirror.com \
  --download-backend aria2 \
  --aria2-connections 16
```

Using an environment variable:

```bash
HF_ENDPOINT=https://hf-mirror.com \
cargo run -p sharp-mcu-cli -- generate \
  --input /path/to/photo.jpg \
  --output-dir ./outputs/rust-package
```

Useful options:

- `--model /path/to/sharp.mlpackage`
- `--model /path/to/sharp_fp16.onnx`
- `--hf-endpoint https://hf-mirror.com`
- `--download-backend aria2`
- `--aria2-connections 16`
- `--model-file sharp.mlpackage`
- `--model-file sharp_fp16.onnx`
- `--model-data-file sharp.onnx.data`
- `--inference-backend cuda`
- `--inference-backend migraphx`
- `--gpu-device-id 0`
- `--cuda-root-dir /usr/local/cuda/lib64`
- `--cudnn-root-dir /usr/lib/x86_64-linux-gnu`
- `--inference-backend coreml-native`
- `--inference-backend coreml-neural-engine`
- `--coreml-cache-dir ./caches/coreml`
- `--ort-intra-threads 4`
- `--ort-inter-threads 2`
- `--cache-dir ./caches/hf-hub`
- `--layers 10`
- `--layer-count 10`
- `--preview-max-edge 2560`
- `--preview-target-pixels 3686400`
- `--disparity-factor 1.0`
- `--depth-scale 1.0`

`--model-data-file` is only needed for legacy external-data ONNX models. The default Apple CoreML path does not use it, and the default non-Apple ONNX path is the single-file `sharp_fp16.onnx`.

## Preview a package

```bash
cargo run -p sharp-mcu-preview -- \
  --manifest ./outputs/rust-package/manifest.json
```

Drag inside the bottom-right device pad to simulate gyro input. The preview consumes only exported PNG layers and motion parameters from the manifest.
The current preview path decodes each PNG once, uploads it to the GPU, and then only updates quad transforms each frame.
Keyboard: `[` / `]` adjusts tilt, `-` / `=` adjusts parallax, `0` or `Esc` resets preview motion.

## Package contents

The generator writes:

- `manifest.json`
- `backdrop.png`
- `layer_00.png ... layer_N.png`
- `debug.ply` when debug export is enabled

`manifest.json` records:

- source image dimensions
- model identity and local model path
- model endpoint
- model format
- model download backend
- requested inference backend and thread settings
- CoreML compiled-model cache directory
- model cache directory
- package dimensions and layer count
- slicing parameters and coverage stats
- motion tuning
- per-layer draw order, near weight, file name, and transform coefficients
