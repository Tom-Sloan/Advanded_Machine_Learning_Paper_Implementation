# Carleton University COMP 5801 - Advanded Machine Learning Papers Implementation

Bolded papers are completed.
Based on the provided files, here's a summary for the README:

# Machine Learning Papers Implementation

This repository contains implementations of various machine learning papers, organized into different categories:

## Project Structure

-   **src/**: Contains the core src code
    -   `*.py`: Python scripts for training and evaluation of models
    -   `models/`: Model definitions, only for papers that use custom models
    -   `trained_models/`: Directory for storing trained models, used for evaluation
    -   `training/`: Training utilities
    -   `data/`: Dataset management and utilities
-   **data/**: Contains datasets including Stonefly images
-   **Attention Is all you need/**: Implementation and presentation materials for the Transformer architecture. This was used for my in-class presentation.

## Usage

The run_paper.py script has implementations for all the papers in the repository, along with with python arguments to run the scripts.

### Examples

#### Training Vision Transformer

```bash
python src/train_vit.py --mode train --data_dir ./Data/Stoneflies --model_path ./trained_models/vit_stonefly/best_model.pt --epochs 10 --batch_size 32 --learning_rate 0.001 --augment
```

#### Running Grad-CAM Visualization

```bash
python src/Grad-cam.py
```

## Dependencies

-   PyTorch
-   torchvision
-   scikit-learn
-   numpy
-   PIL
-   matplotlib
-   tqdm
-   tiktoken

## Project Status

The repository is actively maintained with ongoing implementations of various papers.

## Datasets

The data if from various places, openwebtext, and the shakespeare folders from https://github.com/karpathy/nanoGPT. The Stoneflies data from https://web.engr.oregonstate.edu/~tgd/bugid/stonefly9/.

## Zero-shot Learning

1. C. H. Lampert, et al. “Attribute-based classification for zero-shot visual object categorization” T-PAMI, 2014.
2. Y. Xian, et al. “Latent embeddings for zero-shot classification,” CVPR, 2016.
3. E. Kodirov, et al. “Semantic Autoencoder for Zero-Shot Learning”, CVPR, 2017.
4. Y Xian, et al. "Feature Generating Networks for Zero-Shot Learning", CVPR 2018.

## Few-shot Learning

1. J. Snell, K. Swersky, and R. Zemel, "Prototypical networks for few-shot learning," in NIPS, 2017.
   Vinyals, O., Blundell, C., Lillicrap, T., Kavukcuoglu, K., & Wierstra, D. "Matching Networks for One Shot Learning", NIPS 2016.
2. Chelsea Finn, et. al. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, ICML 2017
3. F. Sung, Y. Yang, and L. Zhang, "Learning to Compare : Relation Network for Few-Shot Learning" CVPR 2018.

## GANs

1. **Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." CVPR, 2017.**
2. Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks" ICCV, 2017.
3. Karras, Tero, et al. "A style-based generator architecture for generative adversarial networks" CVPR 2019.
4. Tero Karras, et al. "Progressive Growing of GANs for Improved Quality, Stability, and Variation", ICLR, 2018.
5. Choi, Yunjey, et al. "Stargan: Unified generative adversarial networks for multi-domain image-to-image translation" CVPR 2018.

## Deep Clustering

1. Xie, Junyuan, et al. "Unsupervised deep embedding for clustering analysis." ICML 2016.
   Yang, Bo, et al. "Towards k-means-friendly spaces: Simultaneous deep learning and clustering" ICML 2017.
2. M. Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv 2022.
3. Shaham, Uri, et al. "Spectralnet: Spectral clustering using deep neural networks." ICLR, 2018

## Transfer Learning and Domain Adaptation

1. Z. Li and D. Hoiem, "Learning without Forgetting", 2016.
2. B. Fernando, et.al., "Unsupervised Visual Domain Adaptation Using Subspace Alignment", ICCV 2013.
3. Judy Hoffman, et.al. "CyCADA: Cycle-Consistent Adversarial Domain Adaptation", ICML 2018

## Self-supervised Learning

1. T Chen, et. al. "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
2. K. He, et. al. "Momentum Contrast for Unsupervised Visual Representation Learning", CVPR 2020
3. J. Grill, et. al. "Bootstrap Your Own Latent A New Approach to Self-Supervised Learning", NeurIPS 2020
4. M. Caron, et. al. "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments", NeurIPS 2020

## Interpretability of ML

1. Marco Tulio Ribeiro, et. al. Why Should I Trust You?: Explaining the Predictions of Any Classifier, KDD 2016
2. Scott Lundberg, Su-In Lee, A Unified Approach to Interpreting Model Predictions NeurIPS 2017
3. **R. R. Selvaraju, et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." ICCV 2017**
4. Been Kim, et. al. Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV), ICML 2018.
5. Oscar Li, et. al. Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions, AAAI 2018
6. C. Chen, et. al. "This Looks Like That: Deep Learning for Interpretable Image Recognition", NeurIPS 2019
7. Marco Ribeiro, et. al. Anchors: High-Precision Model-Agnostic Explanations, AAAI 2018

## Transformers, Language models

1. **Kenton, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." ACL 2019.**
2. Radford, Alec, et al. "Improving language understanding by generative pre-training." (2018).
3. X. L. Li and P. Liang. "Prefix-tuning: Optimizing continuous prompts for generation." ACL, 2021.
4. Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models" ICLR 2022.
5. Kojima, Takeshi, et al. "Large language models are zero-shot reasoners." Neurips, 2022.

## Vision Language models

1. **A. Dosovitskiy, et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2020.**
2. A. Radford, et al. "Learning transferable visual models from natural language supervision." ICML 2021.
3. **J, Ho, A. Jain, and P. Abbeel. "Denoising diffusion probabilistic models." NeurIPS 2020.**
4. T. Brooks, A. Holynski, A. A. Efros "Instructpix2pix: Learning to follow image editing instructions" CVPR 2023.
