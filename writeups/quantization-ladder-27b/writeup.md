# The quantization ladder for a 27B model

*Lucas Rangel. The full executable notebook with every number traced to its data file is [here](https://www.kaggle.com/code/lucasrangelss/the-quantization-ladder-for-a-27b-model); code, tests and data are in the [gpu-ai-benchmark-hub](https://github.com/LucasRangelSSouza/gpu-ai-benchmark-hub) repositories.*

## Summary

A 27 billion parameter language model can be stored and served in four precisions that matter to anyone renting GPUs: full BF16, FP8, NVFP4 and 4-bit AWQ. Each step down roughly halves the memory the weights occupy, and each step has a reputation for costing intelligence. I ran the same benchmark battery on all four on one RTX PRO 6000 with 96 GB, with identical prompts, greedy decoding and a fixed seed, and compared them item by item.

Four findings:

- No precision was measurably worse than the others on math, general knowledge or code. Accuracy ranged from 94.7 to 96.0 percent on 150 math problems, from 80.0 to 82.0 percent on 200 knowledge questions and from 91.5 to 93.3 percent on 164 coding problems. A paired test found no significant difference between any two precisions on any benchmark. The smallest p-value for any pair was 0.48.
- The absence of a difference has limits. With samples of this size, a true gap smaller than about 4 to 5 points cannot be detected. The honest claim is that quantizing cost less than a few points, not that it cost nothing.
- Memory is where the precisions differ sharply. The weights occupy 17.8 GiB in AWQ, 27.7 in NVFP4, 28.1 in FP8 and 51.1 in BF16, and the room left for the conversation cache falls from 654,396 tokens (AWQ) to 318,042 (BF16) on the same card.
- Answers differ in length and speed. Under the same 24-way concurrent load, a math answer took 9.0 seconds in AWQ and 15.6 seconds in BF16, the AWQ and NVFP4 builds wrote answers 32 to 44 percent longer than BF16 and FP8, and BF16 delivered only 55 to 58 percent of the correct answers per minute that FP8 did, for the same accuracy.

One confound runs through the whole ladder and I will repeat it where it matters: the AWQ build comes from a different author and a different starting checkpoint than the other three. The comparison across precisions is therefore a comparison of builds, and only BF16 against FP8 is a clean change of numeric precision on identical weights.

> **Code and data.** Everything behind this article is public: the code that ran, the curated data, the analysis that regenerates every figure, and tests that pin the conclusions to the data. Repositories: [quantization-and-offload-lab](https://github.com/LucasRangelSSouza/quantization-and-offload-lab). Index of the whole series: [gpu-ai-benchmark-hub](https://github.com/LucasRangelSSouza/gpu-ai-benchmark-hub).

## Why a ladder

Renting a GPU makes memory a price. A 24 gigabyte card costs a fraction of a 96 gigabyte one. If a 4-bit build gives the same answers as the BF16 original, you can serve the model on a cheaper card or fit more users on the same card. If it does not, you pay for the larger card or accept worse output. The question is practical, and the literature on it is full of benchmarks that use different prompts, decoding settings and versions of the same model, which makes the claims hard to compare.

I wanted one battery, one card, one engine, one day. The ladder has four rungs: AWQ 4-bit, NVFP4, FP8 and BF16. The purpose is to give a reader a calibrated sense of how big the quality differences are, relative to the noise in the measurement.

## The four builds

The ladder compares builds of a 27 billion parameter model in the Qwen family. The builds are not all derived from the same weights.

The BF16 build is the full-precision checkpoint from the Blackfrost project, at 51.1 GiB in memory. The FP8 configuration loads the same BF16 checkpoint and quantizes it to 8 bits on the fly when the server starts, so it shares every weight with the BF16 build and differs only in numeric precision. The NVFP4 build is a separately quantized checkpoint from the same project, with weights in a 4-bit floating point format that recent NVIDIA cards support. The AWQ build is a 4-bit integer quantization published by a different author, made from a different uncensored base checkpoint.

So the rungs are not a pure ladder:

- BF16 against FP8 isolates the effect of 8-bit numeric precision.
- BF16 against NVFP4 compares the original with a 4-bit float quantization from the same project, a fair test of 4-bit quantization of the same model.
- AWQ against everything else mixes quantization and a different model lineage. A difference there could come from either, and I cannot separate them with this data.

I label the confound here because the summary of such results tends to lose it. Anyone reading "AWQ scored 80 percent on the knowledge benchmark and BF16 scored 82" will want to know whether that gap belongs to the format or to the model. It could belong to either.

## The battery

Four benchmarks, one prompt per item, no examples in the prompt, temperature 0, seed 1, reasoning mode off.

- Math: the first 150 problems of the GSM8K test set, graded by the final numeric answer.
- Knowledge: 200 multiple-choice questions sampled from MMLU across subjects, graded by the chosen letter.
- Code: all 164 HumanEval problems, graded by running the model's code against the problem's unit tests in a separate subprocess.
- Structure: 24 prompts that ask for a JSON object with a fixed schema, graded by a validity check on the parsed output.

All four builds ran on the same RTX PRO 6000 host with 24 requests in flight at a time, through the same vLLM server version, one after the other. The benchmark questions are public datasets. The dataset I publish keeps only the per-item correct or incorrect flags, with no question or answer text.

## Accuracy with error bars

An accuracy measured on 150 items carries uncertainty. A 95 percent Wilson interval around 95 percent on 150 items runs from about 90 to 98. A single number without that interval invites over-reading.

![Figure 1](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig01.png)

| benchmark | precision | correct | n | acc | ci_low | ci_high |
|---|---|---|---|---|---|---|
| gsm8k | AWQ 4-bit | 142 | 150 | 0.947 | 0.898 | 0.973 |
| gsm8k | NVFP4 | 144 | 150 | 0.96 | 0.915 | 0.982 |
| gsm8k | FP8 (online from BF16) | 144 | 150 | 0.96 | 0.915 | 0.982 |
| gsm8k | BF16 | 143 | 150 | 0.953 | 0.907 | 0.977 |
| mmlu | AWQ 4-bit | 160 | 200 | 0.8 | 0.739 | 0.85 |
| mmlu | NVFP4 | 164 | 200 | 0.82 | 0.761 | 0.867 |
| mmlu | FP8 (online from BF16) | 163 | 200 | 0.815 | 0.755 | 0.863 |
| mmlu | BF16 | 164 | 200 | 0.82 | 0.761 | 0.867 |
| humaneval | AWQ 4-bit | 153 | 164 | 0.933 | 0.884 | 0.962 |
| humaneval | NVFP4 | 152 | 164 | 0.927 | 0.876 | 0.958 |
| humaneval | FP8 (online from BF16) | 150 | 164 | 0.915 | 0.862 | 0.948 |
| humaneval | BF16 | 151 | 164 | 0.921 | 0.869 | 0.953 |

Every interval overlaps every other one. On math, the lowest point estimate is AWQ at 94.7 percent (interval 89.8 to 97.3) and the highest are NVFP4 and FP8 at 96.0 (91.5 to 98.2). On knowledge, AWQ scored 80.0 percent (73.9 to 85.0) and NVFP4 and BF16 each 82.0 (76.1 to 86.7). On code, the order inverts: AWQ scored the highest at 93.3 percent (88.4 to 96.2) and FP8 the lowest at 91.5 (86.2 to 94.8). The ordering changes by benchmark, which is what noise looks like: if quantization systematically hurt accuracy, the lower-precision builds would sit lower on all three.

The JSON task needs no figure. Every build passed all 24 prompts.

## Paired comparisons

Overlapping intervals are a weak test, because the builds answered the same questions and mostly agreed. The better question is on which items they disagreed, and whether the disagreements lean one way. That is a paired comparison.

For each pair of builds and each benchmark, I count the items that build A got right and build B wrong, and the reverse, and test whether the split differs from 50 percent using an exact binomial test (McNemar's exact version).

| benchmark | A | B | A_only_right | B_only_right | diff_points | ci95_low | ci95_high | p_exact |
|---|---|---|---|---|---|---|---|---|
| gsm8k | AWQ 4-bit | BF16 | 3 | 4 | -0.7 | -4.1 | 2.8 | 1 |
| gsm8k | NVFP4 | BF16 | 4 | 3 | 0.7 | -2.8 | 4.1 | 1 |
| gsm8k | FP8 (online from BF16) | BF16 | 1 | 0 | 0.7 | -0.6 | 2 | 1 |
| gsm8k | AWQ 4-bit | NVFP4 | 3 | 5 | -1.3 | -5 | 2.4 | 0.727 |
| mmlu | AWQ 4-bit | BF16 | 7 | 11 | -2 | -6.1 | 2.1 | 0.481 |
| mmlu | NVFP4 | BF16 | 5 | 5 | 0 | -3.1 | 3.1 | 1 |
| mmlu | FP8 (online from BF16) | BF16 | 1 | 2 | -0.5 | -2.2 | 1.2 | 1 |
| mmlu | AWQ 4-bit | NVFP4 | 6 | 10 | -2 | -5.9 | 1.9 | 0.454 |
| humaneval | AWQ 4-bit | BF16 | 8 | 6 | 1.2 | -3.2 | 5.7 | 0.791 |
| humaneval | NVFP4 | BF16 | 3 | 2 | 0.6 | -2.1 | 3.3 | 1 |
| humaneval | FP8 (online from BF16) | BF16 | 0 | 1 | -0.6 | -1.8 | 0.6 | 1 |
| humaneval | AWQ 4-bit | NVFP4 | 8 | 7 | 0.6 | -4 | 5.2 | 1 |

![Figure 2](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig02.png)

The exact tests give p-values from 0.48 to 1.0 across the 12 comparisons. The smallest, 0.481, is AWQ against BF16 on knowledge, where BF16 got 11 items right that AWQ missed and AWQ got 7 right that BF16 missed. On math, AWQ and BF16 disagreed on only 7 of 150 items (3 against 4), and NVFP4 and BF16 on 7 as well (4 against 3). On code, AWQ and BF16 split 8 to 6, and FP8 against BF16 differed on a single problem.

Two observations deserve attention.

First, the disagreements are few. For FP8 against BF16, the two builds gave the same correct or incorrect verdict on all but 1 of 150 math items, all but 3 of 200 knowledge items and all but 1 of 164 coding items. That is the signature of a mild change of numeric precision: the model gives the same answers almost every time, and when the verdicts differ, it is one item in a hundred.

Second, the paired interval is much narrower than the unpaired one. The 95 percent interval on the AWQ minus BF16 difference is about plus or minus 4 points on knowledge, against about plus or minus 8 for two independent samples. Pairing buys a cut in the width of roughly half, because the builds make correlated mistakes. The residual width is the limit of what these 150 to 200 items can detect.

## What difference could this battery detect?

A null result is only informative if you know what effect it could have seen. A rough power calculation says how large a difference would have to be to show up reliably.

![Figure 3](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig03.png)

With about 5 percent of items on which the two builds disagree, which is what the data above shows, the smallest gap that a paired test would detect with 80 percent power is about 5.1 points at 150 items, 4.4 at 200 and 2.2 at 800 items. The multiplier of 2.8 is the standard sum of the 5 percent significance and 80 percent power factors.

So the statement I can defend is this. If any of these builds were worse than BF16 by more than about 5 points on these benchmarks, I would probably have seen it. A gap of one or two points is below what I could resolve. Published evaluations that report differences of a point or two between quantization formats on a few hundred items are mostly reporting noise, and the same caution applies to mine.

## Run-to-run noise as a second yardstick

I ran the AWQ build twice on this harness, once on an RTX 3090 earlier in the campaign and once here on the PRO 6000. The two runs scored 95.3 and 94.7 percent on math and 94.5 and 93.3 percent on code, with the knowledge score identical at 80.0. The same model, the same prompts and the same decoding settings moved by up to 1.2 points because of different hardware kernels and floating point order. That is a useful scale: differences between builds of the size of one point are indistinguishable from the difference between two runs of one build.

## How much smaller is each build, really?

The names suggest a tidy ladder: 16 bits, 8 bits, 4 bits and 4 bits. The measured memory says the rungs are not equally spaced.

| precision | weights_gib | gib_vs_bf16 | bits_per_weight_if_27B_params |
|---|---|---|---|
| AWQ 4-bit | 17.8 | 0.35 | 5.66 |
| NVFP4 | 27.69 | 0.54 | 8.81 |
| FP8 (online from BF16) | 28.06 | 0.55 | 8.93 |
| BF16 | 51.1 | 1 | 16.26 |

![Figure 4](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig04.png)

Relative to the BF16 build at 51.1 GiB, FP8 occupies 55 percent, which is what you expect from 8 bits against 16 plus the scales. The AWQ build occupies 35 percent, above the 25 percent that a pure 4-bit model would take, and the NVFP4 build occupies 54 percent, which is almost the size of the 8-bit build.

If the model has about 27 billion parameters, the measured memory corresponds to an average of 5.7 bits per weight for AWQ, 8.8 for NVFP4, 8.9 for FP8 and 16.3 for BF16. The 16.3 bits for BF16 suggests the true parameter count is a little above 27 billion, or that a small part of the memory is not weights, so I read these averages as approximate. The pattern survives that doubt: the AWQ build is much closer to 4 bits than the NVFP4 build is.

Why would a 4-bit build be as large as an 8-bit one? The most likely reason is that the NVFP4 checkpoint quantizes only some of the layers and leaves others, such as attention projections, embeddings, the output head and perhaps a vision tower, in 16-bit. I did not inspect the checkpoint's layer list, so this is an inference from sizes. With the sizes alone, and a block-scaled 4-bit format costing about 4.5 bits per weight, the arithmetic implies that roughly 63 percent of the weights are in 4-bit form if the rest are in 16-bit. The practical consequence is large: this particular NVFP4 build does not save memory over FP8. Whatever advantage it has must come from speed on cards with hardware paths for 4-bit floating point, not from the footprint.

A buyer who sees "4-bit" on a model card and plans a 24 gigabyte card around it should check the size on disk. The AWQ build fit on a 24 gigabyte card with room to spare. The NVFP4 build did not fit on a 32 gigabyte card.

## Where the builds disagree

Aggregate accuracy hides which questions are hard. Looking at the items, most mistakes are shared by all four builds, and the disagreements come from a small set of borderline questions.

| index  | items Unnamed: 1_level_1 | all four correct Unnamed: 2_level_1 | all four wrong Unnamed: 3_level_1 | builds disagree Unnamed: 4_level_1 |
|---|---|---|---|---|
| gsm8k | 150 | 136 | 3 | 11 |
| mmlu | 200 | 150 | 25 | 25 |
| humaneval | 164 | 144 | 3 | 17 |

![Figure 5](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig05.png)

On math, 136 of 150 items were solved by all four builds and 3 were missed by all four; only 11 items (7 percent) split the builds. On knowledge, 150 of 200 were correct everywhere and 25 were wrong everywhere, leaving 25 items (12.5 percent) where builds differed. On code, 144 of 164 were solved by all four, 3 by none, and 17 split the builds.

The knowledge benchmark is the most sensitive because multiple choice with four options gives a 25 percent chance of guessing right. A borderline question can flip with tiny numeric changes. On the splits, the builds were not uniformly ordered: for example on math, among items where exactly one build was wrong, 8 items had one build failing and 1 item had only one build succeeding. A consistent ranking of precisions by quality would show up as the lowest-precision build failing the shared borderline items. It does not.

The 25 items that every build missed on the knowledge benchmark are worth a note. They are either hard, ambiguous or mislabeled, and they set a ceiling of 87.5 percent on this sample that no precision could break. Looking at the benchmark's actual score range of 80 to 82 percent against that ceiling puts the differences between builds in perspective: of the 175 items that at least one build got right, the weakest build got 160 and the strongest 164.

## How to run your own ladder

The method is cheap, and the mistakes are common, so here is the protocol I would follow again.

Fix everything but the thing you vary. Same engine and version, same prompts, same decoding settings, same seed, same host for every rung, one after the other. If your builds come from different sources, write the confound down before you run, because afterwards you will want to forget it.

Use greedy decoding for accuracy benchmarks. With a nonzero temperature, repeated runs of one build move by more than the differences you are looking for, and you would need several repetitions per build to see anything.

Record per-item results. A score is a count, and a count cannot be paired. The per-item flags let you run a paired test and tell shared mistakes from flips.

Size the benchmark to the effect you care about. If you only care about gaps larger than 5 points, 150 items is enough. If you care about 1 point, you need on the order of 4,000 items with a similar disagreement rate, and a fixed set of questions that you re-run rather than a fresh sample each time.

Run one build twice. The gap between two runs of the same configuration is your noise floor, and it costs one extra half hour of rental.

Report the interval next to the point estimate. A table with 94.7 and 95.3 side by side invites a conclusion the data cannot support.

## Questions I expect

Does 4-bit quantization lose accuracy? On this battery, with the detection limit stated above, I could not see a loss. I did not test harder or longer tasks, where the effect could be larger.

Why does the NVFP4 build not shrink more? Its memory is close to FP8's, and my reading is that a large share of its weights stay in 16-bit. If the footprint is what you need, check the size on disk of the specific checkpoint.

Why did AWQ and NVFP4 write longer answers? They derive from different checkpoints than the FP8 and BF16 builds, and answer length is a property of the checkpoint's fine-tuning. The two builds that share weights, FP8 and BF16, wrote math answers within 3 percent of each other in length (309 and 317 tokens).

Which build should I pick for a coding assistant? The one that fits your memory and gives acceptable latency, and then a test on your own code. HumanEval scores of 91.5 to 93.3 percent say nothing about your repository.

Can I use these results for other models? The method transfers. The numbers do not. A smaller model may be more sensitive to quantization. I did not test one, because the 9 billion model ran only in BF16.

## Memory: where the precisions truly differ

Quality is a wash; memory is not.

![Figure 6](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig06.png)

| precision | weights_gib | kv_cache_tokens | concurrent_8k_requests |
|---|---|---|---|
| AWQ 4-bit | 17.8 | 654396 | 79.9 |
| NVFP4 | 27.69 | 555128 | 67.8 |
| FP8 (online from BF16) | 28.06 | 553200 | 67.5 |
| BF16 | 51.1 | 318042 | 38.8 |

The server reports how many tokens of conversation history fit in the cache after loading the weights. On the 96 gigabyte card with the engine's default memory fraction, AWQ leaves room for 654,396 tokens, NVFP4 for 555,128, FP8 for 553,200 and BF16 for 318,042. In terms of simultaneous 8,192-token conversations, that is about 80 for AWQ, 68 each for NVFP4 and FP8 and 39 for BF16. BF16 halves the number of long conversations a card can hold compared with AWQ.

The effect is larger on smaller cards, because the weights take a bigger share. On a 32 gigabyte RTX 5090, the 17.8 gigabyte AWQ build leaves about 10 gigabytes of headroom under the 1.15 memory rule, and nothing larger loads at all. The ladder's memory column is therefore the argument for 4-bit formats, and the capability columns say you do not give up much to get there.

## Speed and answer length

The same server ran each build under 24 concurrent requests, and each item records its latency and, for math, the number of tokens generated.

| index  | seconds Unnamed: 1_level_1 | tokens Unnamed: 2_level_1 | tokens_per_second_per_request Unnamed: 3_level_1 |
|---|---|---|---|
| AWQ 4-bit | 9 | 444.9 | 49.5 |
| NVFP4 | 11.9 | 419.5 | 35.2 |
| FP8 (online from BF16) | 9.3 | 308.7 | 33.3 |
| BF16 | 15.6 | 317.1 | 20.4 |

| index  | AWQ 4-bit Unnamed: 1_level_1 | NVFP4 Unnamed: 2_level_1 | FP8 (online from BF16) Unnamed: 3_level_1 | BF16 Unnamed: 4_level_1 |
|---|---|---|---|---|
| gsm8k | 9 | 11.9 | 9.3 | 15.6 |
| humaneval | 4.6 | 5.7 | 5.9 | 10.2 |
| json | 8.8 | 10.4 | 10.7 | 11.8 |
| mmlu | 0.9 | 0.6 | 1.1 | 1.8 |

![Figure 7](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig07.png)

The AWQ build wrote math answers of 445 tokens on average and the NVFP4 build 420, against 309 for FP8 and 317 for BF16. The longer answers are consistent with the fact that AWQ and NVFP4 builds come from different checkpoints than the FP8 and BF16 pair: a lineage change can alter verbosity without altering the final answer. They are not consistent with a pure effect of 4-bit numeric precision, because the NVFP4 build and the AWQ build are far apart in method yet close in length.

Mean latency per math item under the 24-way load was 9.0 seconds for AWQ, 9.3 for FP8, 11.9 for NVFP4 and 15.6 for BF16. BF16 took 1.7 times as long as FP8 despite generating almost the same number of tokens, which is what a weight-streaming bound predicts: BF16 weights are 1.8 times larger than FP8 weights, and each generated token reads all of them. AWQ is as fast as FP8 despite generating 44 percent more tokens, so per token it is the fastest by a wide margin.

For a single user the pattern holds in the serving data. On the same card the AWQ build decoded 76.7 tokens per second and the FP8 build 45.4. Multiplying each speed by the size of its weights gives 1,365 and 1,274 GiB per second, which is within 7 percent between the two builds and 76 to 82 percent of the 1,792 GB per second memory bandwidth that the manufacturer publishes for this card, a figure I did not measure. In other words, in single-user decoding, speed follows bytes of weights read. The NVFP4 and BF16 single-user results (143 and 79 tokens per second) used speculative decoding and cannot be compared this way. That technique has its own article in the series.

## Correct answers per minute: speed and quality together

Accuracy says how often a build is right and latency says how long one answer takes. Neither tells you how much correct work a card produces. A build that is 1 point more accurate but twice as slow is not better for a service that pays by the hour. The ladder runs make the combined measure possible, because all four builds ran with the same harness, the same 24 concurrent requests and the same engine settings, and, as the server logs confirm, with speculative decoding off for every build. This is the cleanest speed comparison in the whole campaign, since nothing but the build changes.

| index  | tokens Unnamed: 1_level_1 | correct Unnamed: 2_level_1 | wall_s Unnamed: 3_level_1 | tokens_per_second Unnamed: 4_level_1 | correct_per_minute Unnamed: 5_level_1 |
|---|---|---|---|---|---|
| AWQ 4-bit | 66,730 | 142 | 67 | 996 | 127.2 |
| NVFP4 | 62,923 | 144 | 92.5 | 680.2 | 93.4 |
| FP8 (online from BF16) | 46,299 | 144 | 66.7 | 694.1 | 129.5 |
| BF16 | 47,570 | 143 | 119.6 | 397.7 | 71.7 |

| index  | correct Unnamed: 1_level_1 | wall_s Unnamed: 2_level_1 | correct_per_minute Unnamed: 3_level_1 |
|---|---|---|---|
| AWQ 4-bit | 153 | 69.6 | 131.9 |
| NVFP4 | 152 | 91 | 100.2 |
| FP8 (online from BF16) | 150 | 94.1 | 95.6 |
| BF16 | 151 | 162.3 | 55.8 |

![Figure 8](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig08.png)

On the math run the AWQ build generated 996 output tokens per second in aggregate, FP8 694, NVFP4 680 and BF16 398. The ratio between the fastest and slowest is 2.5. It tracks the size of the weights, with one exception: NVFP4 is no faster than FP8 here despite its marketing as a 4-bit format. That agrees with the footprint finding above, where the NVFP4 build is the same size as the FP8 one. Without speculative decoding, on this card and at this concurrency, NVFP4 bought no speed over FP8.

The right-hand panel converts speed into useful work. On math, AWQ and FP8 produced about 127 and 130 correct answers per minute, NVFP4 93 and BF16 72. On code, AWQ produced 132 correct solutions per minute, NVFP4 100, FP8 96 and BF16 56. On math, BF16 delivered 56 percent of AWQ's correct answers per minute and 55 percent of FP8's. On code it delivered 42 percent of AWQ's and 58 percent of FP8's. The paired tests cannot distinguish the quality of any of them.

Two caveats limit how far to take these figures. First, the harness uses 24 concurrent requests regardless of the build, which suits the faster builds and understates the largest-batch throughput the slower builds could reach if I raised the concurrency. Second, the AWQ build writes the longest answers, so its tokens-per-second lead is partly a lead in verbose output, and its correct-answers-per-minute figure is the fairer one. On that measure it is level with FP8 on math and well ahead on code.

If you translate this to money, the BF16 build needs about 1.8 times as many GPU-minutes per correct answer as FP8 on math and 1.7 times on code, on identical weights and with the same accuracy. At the 1.47 dollars per hour the PRO 6000 cost in this session, that is the difference between about 1.9 and 3.4 cents per hundred correct math answers. The sums are small because the battery is small, and the ratio is what carries over to a service that answers millions of questions.

| index  | math_usd_per_100_correct Unnamed: 1_level_1 | code_usd_per_100_correct Unnamed: 2_level_1 |
|---|---|---|
| AWQ 4-bit | 0.0193 | 0.0186 |
| NVFP4 | 0.0262 | 0.0245 |
| FP8 (online from BF16) | 0.0189 | 0.0256 |
| BF16 | 0.0342 | 0.0439 |

## What the cache numbers imply for smaller cards

The server reports its cache capacity in tokens, and the weights in gibibytes. Two numbers per build on one card are enough to back out how much memory one cached token costs, which then lets me estimate what a smaller card would have left over. This is an extrapolation and I label it as one.

| pair | mib_per_token |
|---|---|
| AWQ 4-bit vs BF16 | 0.1014 |
| AWQ 4-bit vs NVFP4 | 0.102 |
| NVFP4 vs BF16 | 0.1011 |
| FP8 (online from BF16) vs BF16 | 0.1003 |

| index  | weights_gib Unnamed: 1_level_1 | cache_pool_gib Unnamed: 2_level_1 | implied_overhead_gib Unnamed: 3_level_1 |
|---|---|---|---|
| AWQ 4-bit | 17.8 | 64.7 | 3.9 |
| NVFP4 | 27.69 | 54.9 | 3.8 |
| FP8 (online from BF16) | 28.06 | 54.7 | 3.6 |
| BF16 | 51.1 | 31.4 | 3.9 |

| index  | AWQ 4-bit Unnamed: 1_level_1 | NVFP4 Unnamed: 2_level_1 | FP8 (online from BF16) Unnamed: 3_level_1 | BF16 Unnamed: 4_level_1 |
|---|---|---|---|---|
| RTX 3090 / 4090 (24) | 0 | 0 | 0 | 0 |
| RTX 5090 (32) | 73 | 0 | 0 | 0 |
| L40S (48) | 218 | 118 | 115 | 0 |
| PRO 6000 (96) | 655 | 555 | 552 | 319 |

![Figure 9](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig09.png)

The four pairwise estimates agree closely, at about 0.10 MiB (roughly 104 KiB) of cache per token. That agreement is a good sign for the method, because the pairs mix different weight sizes and different cache sizes. Multiplying the reported cache capacity by this cost gives the size of the cache pool in each run, between 31 and 65 GiB, and subtracting weights and pool from 90 percent of the card's memory gives an implied fixed overhead of about 3 to 4 GiB for activations and graph capture.

Carrying the same per-token cost and overhead to smaller cards gives mixed results. A 32 gigabyte card running the AWQ build comes out at about 73 thousand tokens of cache, enough for a handful of simultaneous short conversations and not for long agent sessions. The 48 gigabyte card with FP8 or NVFP4 comes out near 115 to 118 thousand tokens, and BF16 does not fit. Only the 96 gigabyte card reaches half a million tokens for the 4-bit builds. For the 24 gigabyte cards the extrapolation says zero, yet those cards did serve the AWQ build with short context windows. So the memory fraction in those runs must have been higher than the 90 percent I assumed, or the overhead lower, and the extrapolation is too pessimistic at the small end. Treat the table as a lower bound and as a way to rank options.

I stress that these are estimates built from one pair of runs on one card, with the assumption that the engine's memory fraction and overhead carry over. The real figure will differ with the engine version, the context-length setting and CUDA graph options. Use the table to rank options, and measure before you commit.

## A buyer's scenario

To close the loop, take a service that must answer 10,000 questions of the GSM8K type an hour at the accuracy this battery shows. The correct-answers-per-minute figures translate into the number of cards, and the price of the card into dollars.

| precision | correct_answers_per_card_hour | cards_for_10k_per_hour | usd_per_hour_96GB_card | usd_per_10k_answers |
|---|---|---|---|---|
| AWQ 4-bit | 7630 | 1.31 | 1.471 | 1.928 |
| NVFP4 | 5604 | 1.78 | 1.471 | 2.625 |
| FP8 (online from BF16) | 7772 | 1.29 | 1.471 | 1.893 |
| BF16 | 4304 | 2.32 | 1.471 | 3.417 |

At 24 concurrent requests, one PRO 6000 running the FP8 build answers about 7,770 math questions correctly per hour, the AWQ build 7,630, NVFP4 5,600 and BF16 4,300. To reach 10,000 correct answers an hour you need 1.3 cards of FP8, 1.3 of AWQ, 1.8 of NVFP4 and 2.3 of BF16, which at 1.471 dollars an hour is 1.89, 1.93, 2.62 and 3.42 dollars. The problem set is a toy and the absolute amounts are meaningless. The relative amounts carry over: the BF16 build costs 1.8 times as much per correct answer as the FP8 build that shares its weights.

## Throughput at scale

Under concurrency the picture changes, because large batches amortize the weight reads.

| index  | BF16 + spec Unnamed: 1_level_1 | FP8 (no spec) Unnamed: 2_level_1 | NVFP4 + spec Unnamed: 3_level_1 | AWQ Unnamed: 4_level_1 |
|---|---|---|---|---|
| 1 | 85 | 42 | 129 | 75 |
| 2 | 134 |  | 195 |  |
| 4 | 280 | 140 | 390 | 234 |
| 8 | 402 | 278 | 605 | 399 |
| 16 | 608 | 419 | 930 | 693 |
| 32 | 505 | 694 | 1,364 |  |
| 64 |  |  | 1,390 |  |
| 100 |  |  | 1,349 |  |
| 128 |  |  | 1,341 |  |

![Figure 10](https://raw.githubusercontent.com/LucasRangelSSouza/gpu-ai-benchmark-hub/main/writeups/quantization-ladder-27b/figures/fig10.png)

At 16 users the AWQ build reached 693 tokens per second and FP8 without speculative decoding 419. At 32 users FP8 reached 694, NVFP4 with speculative decoding 1,364 and BF16 with speculative decoding 505, while the AWQ configuration had not been tested past 16 users on this card. The serving configurations differ in more than precision, since two use speculative decoding and the AWQ and FP8 runs do not, so I read this figure as a measurement of four deployable configurations and not as a controlled comparison of formats.

## Takeaways by role

If you pay for the GPU, the useful figure is correct answers per minute, and the 4-bit AWQ and 8-bit FP8 builds led it on this card at 127 to 132 per minute on both math and code, well ahead of BF16.

If you write the service, the useful figure is the cache capacity, because it sets how many long conversations a card holds. The AWQ build held 654,396 tokens and the BF16 build 318,042 on the same card.

If you evaluate models, the useful figure is the detectable gap. At 150 to 200 items per benchmark the battery cannot see differences below about 4 to 5 points, so a claim that one precision is better than another needs a larger sample and paired item-level data.

If you carry the decision to a buyer, the honest sentence is that I found no capability loss from quantization larger than the noise, that the size of the saving depends on the specific checkpoint, and that behavior, which this article does not cover, was the thing that moved.

## Behavior is a different story

The ladder also ran a behavior battery: a set of prompts that probe where each build draws the line on requests it should refuse, judged by one fixed judge. The result is not "no difference". The builds differ a great deal in how often they comply with boundary-testing requests, and that difference is much larger than any difference in capability. I treat it in a separate article on abliteration, because it is a property of the checkpoint lineage and not of the numeric format, and because the evidence is aggregate counts only. The relevant point for this article is the contrast: capability moved within noise across the ladder while behavior did not. If you choose a quantization format by capability, you may be choosing a behavior without knowing.

## A decision guide

Given the evidence, here is how I would choose.

If memory is the constraint, use the 4-bit format your hardware supports. AWQ ran on every card I measured and left the most room for cache. NVFP4 is no smaller than FP8 in this checkpoint (27.69 against 28.06 GiB loaded), so it saves no memory over FP8, and the 5090 failed to hold it for that reason. The 4-bit float kernels ran on the PRO 6000 and the GB10, and through a software kernel on an Ada card, the L40S. Expect no loss of capability larger than a few points.

If you have the memory for FP8 and your card supports it natively (compute capability 8.9 or higher), FP8 is the safe middle: the same weights as BF16, indistinguishable answers on this battery, and a bit more than half the memory.

Use BF16 only when you must, such as when you intend to fine-tune or when you need to compare against a reference. It costs a card's worth of memory, halves the long-conversation capacity and was the slowest in every latency measure.

Whichever format you pick, run a few hundred of your own items before you commit. The reason is not that quantization is dangerous. It is that the battery here cannot resolve differences of one or two points, and your task may be more sensitive than grade-school math.

## Limitations

The sample sizes are 150, 200 and 164 items, and the JSON set has 24. The detectable gap is 4 to 5 points or more. The builds differ in lineage, and the AWQ comparison cannot isolate format. The battery is zero-shot at temperature 0, so it says nothing about sampling behavior at higher temperatures, long-context accuracy or multilingual quality. The Portuguese free-text generations that the harness also produced are not scored in this article. The server was vLLM at one version with default memory settings. One host produced all four rows, which controls for hardware and confounds nothing, but it also means I have a single repetition per build except for the AWQ rerun. The latency and token counts are under a particular concurrency level, and the throughput table mixes configurations that differ in speculative decoding.

## How much it cost

The ladder ran in one rental of an RTX PRO 6000 that lasted about 42 minutes and cost 1.03 dollars at 1.47 dollars per hour, according to the lease ledger. The same session also ran the behavior battery and an engine comparison, so the ladder's own share is smaller. The campaign as a whole cost US$ 50.79.

## Reproducing this analysis

Attach the dataset `gpu-ai-benchmark-campaign-2026`. The per-item file `ladder_items.csv` holds a correct or incorrect flag, the latency and, for math, the token count for each of the 538 items per build. The repository `quantization-and-offload-lab` publishes the harness, `intel.py`, and the launch script for the ladder.

## Glossary

AWQ: activation-aware weight quantization to 4-bit integers. NVFP4: a 4-bit floating point format with block scaling, native to Blackwell tensor cores. FP8: 8-bit floating point, native from compute capability 8.9. BF16: 16-bit brain floating point, the usual full-precision format for training and serving. KV cache: stored attention state for earlier tokens. Wilson interval: a confidence interval for a proportion that behaves well near 0 and 1. McNemar's test: a paired test for two classifiers evaluated on the same items. Speculative decoding: a draft model proposes tokens that the large model verifies in parallel.

Lucas Rangel
