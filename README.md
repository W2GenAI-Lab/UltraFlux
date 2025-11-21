# UltraFlux: Data-Model Co-Design for High-quality Native 4K Text-to-Image Generation across Diverse Aspect Ratios


<p align="center"> 
<a href="https://huggingface.co/Owen777/UltraFlux-v1"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=green"></a>
<a href="https://huggingface.co/Owen777/UltraFlux-v1-TechReport/resolve/main/Techreport_UltraFlux.pdf"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=TechReport&color=yellow"></a>
</p>

**UltraFlux is a diffusion transformer that extends Flux backbones to native 4K synthesis with consistent quality across a wide range of aspect ratios. The project unifies data, architecture, objectives, and optimization so that positional encoding, VAE compression, and loss design reinforce each other rather than compete.**

![UltraFlux samples](fig/git_image_full.jpg)

<details>
<summary>Show UltraFlux generation examples</summary>

<table>
  <tr>
    <td align="center"><img src="fig/ultra_flux_01.jpeg" alt="UltraFlux sample 01 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_02.jpeg" alt="UltraFlux sample 02 (4096×4096)" width="300"></td>
  </tr>
  <tr>
    <td align="center"><img src="fig/ultra_flux_03.jpeg" alt="UltraFlux sample 03 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_04.jpeg" alt="UltraFlux sample 04 (4096×4096)" width="300"></td>
  </tr>
  <tr>
    <td align="center"><img src="fig/ultra_flux_05.jpeg" alt="UltraFlux sample 05 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_06.jpeg" alt="UltraFlux sample 06 (4096×4096)" width="300"></td>
  </tr>
  <tr>
    <td align="center"><img src="fig/ultra_flux_07.jpeg" alt="UltraFlux sample 07 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_08.jpeg" alt="UltraFlux sample 08 (4096×4096)" width="300"></td>
  </tr>
  <tr>
    <td align="center"><img src="fig/ultra_flux_09.jpeg" alt="UltraFlux sample 09 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_10.jpeg" alt="UltraFlux sample 10 (4096×4096)" width="300"></td>
  </tr>
  <tr>
    <td align="center"><img src="fig/ultra_flux_11.jpeg" alt="UltraFlux sample 11 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_12.jpeg" alt="UltraFlux sample 12 (4096×4096)" width="300"></td>
  </tr>
  <tr>
    <td align="center"><img src="fig/ultra_flux_13.jpeg" alt="UltraFlux sample 13 (4096×4096)" width="300"></td>
    <td align="center"><img src="fig/ultra_flux_14.jpeg" alt="UltraFlux sample 14 (4096×4096)" width="300"></td>
  </tr>
</table>
<p align="center"><em>Each sample is rendered at 4096×4096 resolution.</em></p>

</details>

## 👥 Authors

> [**Tian Ye**](https://owen718.github.io/)<sup>1</sup>\*‡,[**Song Fei**](https://feisong123.github.io)<sup>1</sup>\*, [**Lei Zhu**](https://sites.google.com/site/indexlzhu/home)<sup>1,2</sup>†
>
> <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)  
> <sup>2</sup>The Hong Kong University of Science and Technology  
>
> \*Equal Contribution, ‡Project Leader, †Corresponding Author

---



## 📰 News ✨✨

**[2025.11.21]** – We released the **UltraFlux-v1.1** transformer checkpoint. It is fine-tuned on a carefully curated set of high-aesthetic synthetic images to further improve visual aesthetics and composition quality. You can now enable it easily by uncommenting the corresponding lines in `inf_ultraflux.py`!

**[2025.11.20]** – We released the **UltraFlux-v1** checkpoint, inference code, and the accompanying tech report.

---

## Inference Quickstart
- The script `inf_ultraflux.py` downloads the latest `Owen777/UltraFlux-v1` weights (transformer + VAE) and runs a set of curated prompts.
- Ensure PyTorch, `diffusers`, and CUDA are available, then run:

```bash
python inf_ultraflux.py
```

- Generated images are saved into `results/ultra_flux_*.jpeg` at 4096×4096 resolution; edit the prompt list or pipeline arguments inside the script to customize inference.

## Why UltraFlux?
- **4K positional robustness.** Resonance 2D RoPE with YaRN keeps training-window awareness while remaining band-aware and aspect-ratio aware to avoiding ghosting.
- **Detail-preserving compression.** A lightweight, non-adversarial post-training routine sharpens Flux VAE reconstructions at 4K without sacrificing throughput, resolving the usual trade-off between speed and micro-detail.
- **4K-aware objectives.** The SNR-Aware Huber Wavelet Training Objective emphasizes high-frequency fidelity in the latent space so gradients stay balanced across timesteps and frequency bands.
- **Aesthetic-aware scheduling.** Stage-wise Aesthetic Curriculum Learning (SACL) routes high-aesthetic supervision toward high-noise steps, sculpting the model prior where it matters most for vivid detail and alignment.

## MultiAspect-4K-1M Dataset
- **Scale and coverage.** 1M native and near-4K images with controlled aspect-ratio sampling to ensure both wide and portrait regimes are equally represented.
- **Content balance.** A dual-channel collection pipeline debiases landscape-heavy sources toward human-centric content.
- **Rich metadata.** Every sample includes bilingual captions, subject tags, CLIP/VLM-based quality and aesthetic scores, and classical IQA metrics, enabling targeted subset sampling for specific training stages.

## Model & Training Recipe
1. **Backbone.** Flux-style DiT trained directly on MultiAspect-4K-1M with token-efficient blocks and Resonance 2D RoPE + YaRN for AR-aware positional encoding.
2. **Objective.** SNR-Aware Huber Wavelet loss aligns gradient magnitudes with 4K statistics, reinforcing high-frequency fidelity under strong VAE compression.
3. **Curriculum.** SACL injects high-aesthetic data primarily into high-noise timesteps so the model’s prior captures human-desired structure early in the trajectory.
4. **VAE Post-training.** A simple, non-adversarial fine-tuning pass boosts 4K reconstruction quality while keeping inference cost low.

## Results
UltraFlux surpasses recent native-4K and training-free scaling baselines on standard 4K benchmarks spanning:
- Image fidelity at 4096×4096 and higher
- Aesthetic preference scores
- Text-image alignment metrics across diverse aspect ratios

## Resources
We will release the full stack upon publication:
- MultiAspect-4K-1M dataset with metadata loaders
- Training pipelines
- Evaluation code covering fidelity, aesthetic, and alignment metrics

## 🚀 Updates
For the purpose of fostering research and the open-source community, we plan to open-source the entire project, encompassing training, inference, weights, etc. Thank you for your patience and support! 🌟
- [x] Release GitHub repo.
- [x] Release inference code (`inf_ultraflux.py`).
- [ ] Release training code.
- [x] Release model checkpoints.
- [ ] Release arXiv paper.
- [ ] Release HuggingFace Space demo.
- [x] Release dataset (MultiAspect-4K-1M).

Stay tuned for links and usage instructions. For updates, please watch this repository or open an issue.
