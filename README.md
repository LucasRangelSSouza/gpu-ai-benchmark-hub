# GPU and AI ecosystem benchmark, October 2026

**Eleven repositories, twenty long-form Kaggle articles and one public dataset from a measurement campaign on rented and owned GPUs.** The question behind all of it: *for the open-source AI tools a small company might build on (language model serving, vision, speech, image and video generation, 3D, browser automation), what runs on which hardware, how fast, and at what cost per unit of work?*

Eight GPU models (RTX 3090, RTX 4090, RTX 5090, L40S, A100, H100 NVL, RTX PRO 6000 and an ARM GB10 desk machine) on a marketplace, two days, **US$ 50.79 in total**. Everything that is reported was measured; what failed is kept, classified and priced; estimates are labelled as estimates.

## How the work is organized

- **Repositories** hold the code that ran, the curated data, reproducible analysis (figures and tables regenerate with `make all`), tests that pin the published conclusions to the data, and a README that states why the test matters, the headline numbers, and the limits.
- **Articles** (Kaggle notebooks, English) tell each story in depth with the same data attached: methods, figures, what could be wrong, and how much it cost.
- **Dataset**: [`lucasrangelss/gpu-ai-benchmark-campaign-2026`](https://www.kaggle.com/datasets/lucasrangelss/gpu-ai-benchmark-campaign-2026): curated CSVs (no host locations, no instance identifiers, no model outputs).

## Repositories

| Repository | What is in it |
|---|---|
| [vast-gpu-benchmark-harness](https://github.com/LucasRangelSSouza/vast-gpu-benchmark-harness) | Orchestration, safety nets, cost ledger and the pricing model |
| [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) | vLLM, SGLang and llama.cpp under load; KV cache; long context; start logs |
| [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab) | Quantization ladder with paired tests, speculative decoding, bandwidth identity, offload |
| [vision-detection-tracking-bench](https://github.com/LucasRangelSSouza/vision-detection-tracking-bench) | YOLO11, SAM 2, Grounding DINO, Mask R-CNN, Detectron2 on 8 GPUs |
| [multi-camera-rtsp-capacity](https://github.com/LucasRangelSSouza/multi-camera-rtsp-capacity) | How many RTSP cameras one GPU sustains: process-per-camera against a batch service |
| [image-video-generation-bench](https://github.com/LucasRangelSSouza/image-video-generation-bench) | SDXL-Turbo, FLUX.3 Klein, LTX-Video, FFmpeg NVENC; includes the generated artifacts |
| [voice-stt-tts-bench](https://github.com/LucasRangelSSouza/voice-stt-tts-bench) | F5-TTS, faster-whisper, OpenVoice, PersonaPlex: real-time factors and cold starts |
| [image-to-3d-and-digital-twin](https://github.com/LucasRangelSSouza/image-to-3d-and-digital-twin) | Ten attempts at an image-to-3D tool, COLMAP note, 3D viewer build |
| [web-automation-lab](https://github.com/LucasRangelSSouza/web-automation-lab) | Playwright, Crawl4AI, Obscura and a bot-detection page in a sandbox |
| [agent-tooling-token-savings](https://github.com/LucasRangelSSouza/agent-tooling-token-savings) | Context compression and tracing smoke tests, with a clear statement of what is not claimed |
| [app-platforms-smoke-tests](https://github.com/LucasRangelSSouza/app-platforms-smoke-tests) | Taipy, Litpage, Open WebUI and MediaMTX install-and-answer checks |

## Articles

| # | Article | What it shows | Code |
|---|---|---|---|
| 1 | [What a GPU benchmark campaign really costs](https://www.kaggle.com/code/lucasrangelss/what-a-gpu-benchmark-campaign-really-costs) | US$ 50.79 reconciled against US$ 27.25 of hourly rental: where the money and the mistakes went | [vast-gpu-benchmark-harness](https://github.com/LucasRangelSSouza/vast-gpu-benchmark-harness) |
| 2 | [What fits and what fails across seven GPUs](https://www.kaggle.com/code/lucasrangelss/what-fits-and-what-fails-across-seven-gpus) | 126 cells (model x GPU): memory predicts most failures, software explains the rest | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench), [vast-gpu-benchmark-harness](https://github.com/LucasRangelSSouza/vast-gpu-benchmark-harness) |
| 3 | [Cost per token and cost per user for a 27B model](https://www.kaggle.com/code/lucasrangelss/cost-per-token-and-cost-per-user-for-a-27b-model) | Cost per million tokens and per user for a 27B model on each card, with an SLA | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) |
| 4 | [The quantization ladder for a 27B model](https://www.kaggle.com/code/lucasrangelss/the-quantization-ladder-for-a-27b-model) | AWQ, NVFP4, FP8 and BF16 on the same 538 items: no detectable loss, and NVFP4 is no smaller than FP8 | [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab) |
| 5 | [Measuring refusal behavior in 27B builds](https://www.kaggle.com/code/lucasrangelss/measuring-refusal-behavior-in-27b-builds) | How to measure the behavior of unrestricted builds with a fixed judge (aggregates only) | [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab) |
| 6 | [How much speculative decoding really speeds up](https://www.kaggle.com/code/lucasrangelss/how-much-speculative-decoding-really-speeds-up) | What a drafter really buys: single-user speed up, aggregate under load not always | [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab), [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) |
| 7 | [Long context and prefix caching for agents](https://www.kaggle.com/code/lucasrangelss/long-context-and-prefix-caching-for-agents) | Cold prefill against prefix-cache hits for agent workloads | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) |
| 8 | [A 35-minute soak test on one RTX 5090](https://www.kaggle.com/code/lucasrangelss/a-35-minute-soak-test-on-one-rtx-5090) | A 35-minute stability test: 4,051 requests, no failure, and what the warm-up hides | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) |
| 9 | [vLLM against SGLang on one RTX 5090](https://www.kaggle.com/code/lucasrangelss/vllm-against-sglang-on-one-rtx-5090) | Two engines tie until 16 users; at 32 a batch cap makes one of them queue | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) |
| 10 | [A 27B model on a 24 GB card in three ways](https://www.kaggle.com/code/lucasrangelss/a-27b-model-on-a-24-gb-card-in-three-ways) | Quantize, shrink or offload: three routes for a 27B model on 24 GB, with measured caches | [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab), [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench) |
| 11 | [NVFP4 on cards without FP4 hardware](https://www.kaggle.com/code/lucasrangelss/nvfp4-on-cards-without-fp4-hardware) | NVFP4 needs memory, not Blackwell: the 5090 fails, the Ada L40S runs | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench), [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab) |
| 12 | [What runs on an ARM GB10 desk machine](https://www.kaggle.com/code/lucasrangelss/what-runs-on-an-arm-gb10-desk-machine) | A 121 GB ARM desk machine: huge cache, slow decode, two software gaps | [llm-serving-engines-bench](https://github.com/LucasRangelSSouza/llm-serving-engines-bench), [vision-detection-tracking-bench](https://github.com/LucasRangelSSouza/vision-detection-tracking-bench) |
| 13 | [YOLO11 on eight GPUs and a CPU-bound surprise](https://www.kaggle.com/code/lucasrangelss/yolo11-on-eight-gpus-and-a-cpu-bound-surprise) | Small detectors are limited by the host, not the GPU | [vision-detection-tracking-bench](https://github.com/LucasRangelSSouza/vision-detection-tracking-bench) |
| 14 | [SAM 2 Grounding DINO and Mask R-CNN latency](https://www.kaggle.com/code/lucasrangelss/sam-2-grounding-dino-and-mask-r-cnn-latency) | Segmentation latency across cards, an unexplained 16x outlier and Detectron2 against its stand-in | [vision-detection-tracking-bench](https://github.com/LucasRangelSSouza/vision-detection-tracking-bench) |
| 15 | [How many cameras can one RTX 5090 serve](https://www.kaggle.com/code/lucasrangelss/how-many-cameras-can-one-rtx-5090-serve) | The system, not the model, sets how many cameras one GPU sustains | [multi-camera-rtsp-capacity](https://github.com/LucasRangelSSouza/multi-camera-rtsp-capacity), [vision-detection-tracking-bench](https://github.com/LucasRangelSSouza/vision-detection-tracking-bench) |
| 16 | [Image and video generation cost per output](https://www.kaggle.com/code/lucasrangelss/image-and-video-generation-cost-per-output) | Cost per image and per second of video: batching, load time and hardware support | [image-video-generation-bench](https://github.com/LucasRangelSSouza/image-video-generation-bench) |
| 17 | [Speech workloads on rented GPUs](https://www.kaggle.com/code/lucasrangelss/speech-workloads-on-rented-gpus) | Real-time factors, cold starts and cost per audio hour for speech workloads | [voice-stt-tts-bench](https://github.com/LucasRangelSSouza/voice-stt-tts-bench) |
| 18 | [Ten attempts to run an image-to-3D tool](https://www.kaggle.com/code/lucasrangelss/ten-attempts-to-run-an-image-to-3d-tool) | Ten attempts at one image-to-3D tool: two finished, one platform limit | [image-to-3d-and-digital-twin](https://github.com/LucasRangelSSouza/image-to-3d-and-digital-twin) |
| 19 | [Eighty-three open tools in a disposable sandbox](https://www.kaggle.com/code/lucasrangelss/eighty-three-open-tools-in-a-disposable-sandbox) | 83 open tools in a sandbox: 69% of failures were my harness, retries never helped | [web-automation-lab](https://github.com/LucasRangelSSouza/web-automation-lab), [agent-tooling-token-savings](https://github.com/LucasRangelSSouza/agent-tooling-token-savings), [app-platforms-smoke-tests](https://github.com/LucasRangelSSouza/app-platforms-smoke-tests) |
| 20 | [From a benchmark to a price per unit](https://www.kaggle.com/code/lucasrangelss/from-a-benchmark-to-a-price-per-unit) | From a measured speed to a price: utilization ceilings, a price formula and what it leaves out | [vast-gpu-benchmark-harness](https://github.com/LucasRangelSSouza/vast-gpu-benchmark-harness) |

## Ten findings that survived the checking

1. A benchmark campaign's total spend was **1.86x** its hourly-rental estimate; the difference is what price lists do not show.
2. **Memory, not architecture, decides which model starts**: the 32 GB RTX 5090 cannot hold the 28 GiB FP8 or NVFP4 builds of a 27B model, while an Ada card with 45 GB runs NVFP4 through a software kernel.
3. This NVFP4 checkpoint is **no smaller than FP8**; only AWQ is really small. No precision was detectably worse on 538 paired items.
4. Every route that stays on the GPU reads weights at about a terabyte per second; **offloading to disk reads at about half a percent of that**.
5. Two serving engines tied up to 16 users; at 32 one queued because of an unreported **batch cap**: report achieved concurrency, not requested.
6. Small vision models are **host-bound**: a newer GPU can be slower for one image at a time.
7. The detector's speed overestimated how many cameras a GPU could watch by several times; **decoding and process design** set the limit.
8. A video model took about 160 times longer to load than to generate a second of video: price it as a **warm pool**.
9. Of 217 sandbox attempts, **69% of the failures were my own harness**, and no retry ever recovered a failed first attempt.
10. Pooling is worth more than a faster card: a queue model puts the utilization ceiling at about 5% for one slot and about 79% for 32.

## Rules the work followed

Only tests that were actually run count; installs without output were left out. Failures are data. Estimates are never presented as measurements. No secrets, host locations or instance identifiers are published. Model outputs of unrestricted builds are not published, only aggregate rates. Licences of tools and weights are listed in each repository's `LICENSE_NOTES.md`; weights are never redistributed.

Author: Lucas Rangel. Code under MIT; third-party tools keep their licences.
