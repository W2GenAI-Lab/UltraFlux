# UltraFlux: Data-Model Co-Design for High-quality Native 4K Text-to-Image Generation across Diverse Aspect Ratios


<p align="center"> 
<a href="https://huggingface.co/Owen777/UltraFlux-v1"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=green"></a>
<!-- <a href="https://huggingface.co/datasets/bytedance-research/UNO-1M"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Dataset&color=yellow"></a> -->
</p>

UltraFlux is a diffusion transformer that extends Flux backbones to native 4K synthesis with consistent quality across a wide range of aspect ratios. The project unifies data, architecture, objectives, and optimization so that positional encoding, VAE compression, and loss design reinforce each other rather than compete. This README outlines the motivation, core ideas, and resources released with the paper.

![UltraFlux samples](fig/git_image_full.jpg)

## Why UltraFlux?
- **4K positional robustness.** Resonance 2D RoPE with YaRN keeps training-window awareness while remaining band-aware and aspect-ratio aware, avoiding ghosting and aliasing when extrapolating to extreme resolutions.
- **Detail-preserving compression.** A lightweight, non-adversarial post-training routine sharpens Flux VAE reconstructions at 4K without sacrificing throughput, resolving the usual trade-off between speed and micro-detail.
- **4K-aware objectives.** The SNR-Aware Huber Wavelet Training Objective emphasizes high-frequency fidelity in the latent space so gradients stay balanced across timesteps and frequency bands.
- **Aesthetic-aware scheduling.** Stage-wise Aesthetic Curriculum Learning (SACL) routes high-aesthetic supervision toward high-noise steps, sculpting the model prior where it matters most for vivid detail and alignment.

## MultiAspect-4K-1M Dataset
- **Scale and coverage.** 1M native and near-4K images with controlled aspect-ratio sampling to ensure both ultra-wide and portrait regimes are equally represented.
- **Content balance.** A dual-channel collection pipeline debiases landscape-heavy sources toward human-centric content.
- **Rich metadata.** Every sample includes bilingual captions, subject tags, CLIP/VLM-based quality and aesthetic scores, and classical IQA metrics, enabling targeted subset sampling for specific training stages.

## Model & Training Recipe
1. **Backbone.** Flux-style DiT trained directly on MultiAspect-4K-1M with token-efficient blocks and Resonance 2D RoPE + YaRN for AR-aware positional encoding.
2. **Objective.** SNR-Aware Huber Wavelet loss aligns gradient magnitudes with 4K statistics, reinforcing high-frequency fidelity under strong VAE compression.
3. **Curriculum.** SACL injects high-aesthetic data primarily into high-noise timesteps so the model’s prior captures human-desired structure early in the trajectory.
4. **VAE Post-training.** A simple, non-adversarial fine-tuning pass boosts 4K reconstruction quality while keeping inference cost low.

## Results
UltraFlux surpasses recent native-4K and training-free scaling baselines on standard 4K benchmarks spanning:
- Pixel fidelity at 4096×4096 and higher
- Aesthetic preference scores
- Text-image alignment metrics across diverse aspect ratios

## Resources
We will release the full stack upon publication:
- MultiAspect-4K-1M dataset with metadata loaders
- Training pipelines, checkpoints, and inference scripts
- Evaluation code covering fidelity, aesthetic, and alignment metrics

## 🚀 Updates
For the purpose of fostering research and the open-source community, we plan to open-source the entire project, encompassing training, inference, weights, etc. Thank you for your patience and support! 🌟
- [x] Release GitHub repo.
- [x] Release inference code (`inf_ultraflux.py`)
- [x] Release model checkpoints.
- [ ] Release arXiv paper.
- [ ] Release training code.

- [ ] Release HuggingFace Space demo.
- [ ] Release dataset (MultiAspect-4K-1M).

Stay tuned for links and usage instructions. For updates, please watch this repository or open an issue.
