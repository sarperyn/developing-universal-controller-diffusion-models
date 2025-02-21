# Developing a Universal Controller for 3D Non-linear Attributes on the h-space of Diffusion Models

> **Status**: This project was conducted at the Georgia Tech Rehg Lab under the supervision of Ozgur Kara (PhD student of Prof. James M. Rehg). Despite extensive exploration and experimentation, it ended without conclusive success. We share our research notes, code, and intermediate findings here for transparency and future reference.

## Overview

Recent work on Diffusion Models (DDPMs) reveals that they contain a **semantic latent space**, referred to as **h-space**. By exploiting this space, we can manipulate image attributes (such as rotation, distance, and lighting) without retraining the entire diffusion model. Instead, we introduce a lightweight component—called a **Delta-h Generator**—to shift the latent representation, thus steering the output image along a chosen attribute (e.g., object rotation in 3D).

This project explores how to:

1. **Identify and learn** local directions in h-space corresponding to 3D attributes.
2. **Demonstrate** that even non-linear, continuous attributes (like azimuth, elevation, or zoom) can be effectively controlled via small modifications in the h-vectors.
3. **Generalize** the approach to multiple objects and attributes, aiming for a **universal controller** that can manipulate diverse 3D attributes with minimal computation.

---

## Motivation

- **Semantic h-space**: Prior work (e.g., [1]) shows that diffusion models naturally encode semantic information in h-space. Initial studies focused on discrete attributes (e.g., gender, presence of eyeglasses), but many real-world tasks require controlling **continuous, non-linear** attributes like rotation or zoom.
- **Efficiency**: Rather than fine-tuning the entire diffusion model, we train only a small network (the Delta-h Generator). This lowers computational cost and speeds up iterative experimentation.
- **3D Representation**: Controlling viewpoint, camera distance, lighting, and object location is crucial for **novel view synthesis**, **3D editing**, and other vision tasks.

---

## Key Ideas

1. **h-space Inversion**  
   We leverage “asymmetric reverse processes” ([1], “ASYRP”) to extract latent vectors (h-vectors) that correspond to semantic attributes. We analyze these vectors for multiple images with varied viewpoints or attributes, visualizing them via PCA or UMAP to identify distinct trajectories in the latent space.

2. **Delta-h Generator**  
   - A lightweight module that learns to output a small shift **Δh** given a current h-vector and a desired attribute change (e.g., “rotate by 20°”).  
   - By **adding** this Δh to the original h-vector and running the diffusion model’s forward or reverse steps, we obtain a manipulated image that reflects the desired change.

3. **Loss Functions**  
   - **Reconstruction loss** between the generated image and a target image (e.g., an image of the same object from the desired new viewpoint).  
   - **Identity preservation** terms that ensure the object’s appearance remains consistent while only the target attribute changes.  
   - Metrics (L1, L2, SSIM, LPIPS, PSNR) measure how close the manipulated image is to both the source and target images.

4. **Partial Sampling Procedure**  
   - **T → t_edit**: Perform an asymmetric sampling (inversion) to obtain h-vectors.  
   - **t_edit → t_boost**: Inject the Δh shift.  
   - **t_boost → 0**: Use DDIM sampling for quality boosting of the final image.

---

## Methodology

1. **Reconstructing the Original Image**  
   - For a given image, run partial diffusion steps backward (DDIM inversion) to find its latent representation h.

2. **Applying Delta-h**  
   - Feed the original h-vector and a desired attribute embedding (e.g., “rotate +θ”) into the Delta-h Generator.  
   - The network predicts a small shift **Δh**.

3. **Forward Generation**  
   - Add Δh to the original h-vector to get a modified h-vector.  
   - Run forward diffusion steps (optionally with partial noise injection and sampling) to produce the edited image.

4. **Training the Delta-h Generator**  
   - Pairs of images (x₁, x₂) depict the same object with a known attribute difference.  
   - Invert both x₁ and x₂ to obtain h₁ and h₂, compute the difference.  
   - During training, minimize the discrepancy between the generated image (via h₁ + Δh) and x₂, while preserving overall appearance.

---

## Datasets

1. **Fixed-Elevation Chair Dataset**
   - 32 chairs (16 excluded for poor reconstruction).  
   - 360° azimuth rotation, 1° increments, fixed distance and elevation.  
   - Images are 128×128, no background.

2. **Varying-Elevation Chair Dataset**
   - 300 chairs (123 excluded).  
   - 4 full-rotations, each with a different elevation.  
   - 480 images per object (3° increments in azimuth), fixed distance, no background.

**Note**: Objects with consistently **distorted reconstructions** were excluded from training (16 from the fixed-elevation set, 123 from the varying-elevation set).

---

## Experiments

### Single-Object Rotation (Fixed Elevation)

- Train Delta-h Generator on a single object’s views (e.g., 360° rotation).  
- Observe if the model successfully maps small Δh changes to consistent, incremental rotations.

### Multi-Object Rotation (Fixed Elevation)

- Expand to multiple chairs with fixed elevation.  
- Evaluate how well the learned Δh generalizes across different shapes and appearances.

### Multi-Object Rotation (Varying Elevation)

- Include objects with multiple elevations.  
- Assess the disentanglement of **azimuth** vs. **elevation** in the latent space.

### Metrics

- **Quantitative**: L1, L2, LPIPS, SSIM, PSNR comparing (modified_img vs. source_img) and (modified_img vs. target_img).  
- **Qualitative**: Visual inspection to confirm smooth rotation and identity preservation.

---

## Discussion and Next Steps

- **Evaluation Protocol**: We aim to establish rigorous quantitative measures for how faithfully the attribute changes occur while preserving identity.
- **Generalization**:
  - Train on multiple objects; test on unseen ones.  
  - Add embeddings for continuous angle changes and potentially other attributes (e.g., zoom or lighting).
- **Stable Diffusion**:  
  - Transfer these ideas to text-to-image diffusion models (e.g., Latent Diffusion Models [9]) and see if textual prompts like “rotate by 45 degrees” can drive the same Delta-h approach.
- **Disentangling Multiple Attributes**:  
  - Use orthogonalization (e.g., Gram–Schmidt) to separate different latent directions (rotation vs. appearance changes).  
  - Could enable more robust, multi-attribute editing in a single unified framework.

---

## Findings

- [**Experiment Results**](https://drive.google.com/file/d/1tbSvsnILHzBn2w2na6Bdq19bfTGIjpr8/view?usp=sharing)
- [**Additional Presentation**](https://drive.google.com/file/d/1C5ePbwt4Sr6kHfRi1LWi4hzq7vSoqEUu/view?usp=sharing)

These slides detail some of our experiments, including intermediate and final outputs. Although the final outcomes were inconclusive, we hope these findings may guide future research.

---

## References

1. Kwon, M., Jeong, J., & Uh, Y. (2022). [Diffusion models already have a semantic latent space](https://arxiv.org/abs/2210.10960). *arXiv preprint arXiv:2210.10960*.
2. Lu, Z., et al. (2023). [Hierarchical Diffusion Autoencoders and Disentangled Image Manipulation](https://arxiv.org/abs/2304.11829). *arXiv preprint arXiv:2304.11829*.
3. Park, Y.-H., et al. (2023). [Unsupervised discovery of semantic latent directions in diffusion models](https://arxiv.org/abs/2302.12469). *arXiv preprint arXiv:2302.12469*.
4. Jiménez, Á. B. (2023). [Mixture of diffusers for scene composition and high resolution image generation](https://arxiv.org/abs/2302.02412). *arXiv preprint arXiv:2302.02412*.
5. Couairon, G., et al. (2022). [Diffedit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427). *arXiv preprint arXiv:2210.11427*.
6. Meng, C., et al. (2021). [SDEdit: Guided image synthesis and editing with stochastic differential equations](https://arxiv.org/abs/2108.01073). *arXiv preprint arXiv:2108.01073*.
7. Ryu, D., & Ye, J. C. (2022). [Pyramidal denoising diffusion probabilistic models](https://arxiv.org/abs/2208.01864). *arXiv preprint arXiv:2208.01864*.
8. Jeong, J., Kwon, M., & Uh, Y. (2023). [Training-free Style Transfer Emerges from h-space in Diffusion Models](https://arxiv.org/abs/2303.15403). *arXiv preprint arXiv:2303.15403*.
9. Rombach, R., et al. (2022). [High-resolution image synthesis with latent diffusion models](https://arxiv.org/abs/2112.10752). *CVPR*.
10. Vahdat, A., Kreis, K., & Kautz, J. (2021). [Score-based generative modeling in latent space](https://arxiv.org/abs/2111.XXXX). *NeurIPS, 34*.
11. Kim, G., Kwon, T., & Ye, J. C. (2022). [DiffusionCLIP: Text-guided diffusion models for robust image manipulation](https://arxiv.org/abs/2110.02711). *CVPR*.
