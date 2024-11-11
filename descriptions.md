## Zero-shot Learning

1. C. H. Lampert, et al. “Attribute-based classification for zero-shot visual object categorization” T-PAMI, 2014.

    - Introduced an attribute-based approach to zero-shot learning, enabling recognition of unseen object classes by leveraging semantic attributes.
    - Proposed methods to predict unseen classes by associating low-level visual features with high-level semantic attributes.
    - Demonstrated effectiveness on large-scale object recognition tasks, bridging the gap between visual data and semantic descriptions.

2. Y. Xian, et al. “Latent embeddings for zero-shot classification,” CVPR, 2016.

    - Presented a method that learns latent embeddings to align visual features with semantic representations for zero-shot classification.
    - Proposed a bilinear compatibility framework to model the relationship between images and class embeddings.
    - Achieved improved performance by learning more discriminative latent spaces for unseen class prediction.

3. E. Kodirov, et al. “Semantic Autoencoder for Zero-Shot Learning”, CVPR, 2017.

    - Introduced the Semantic Autoencoder (SAE) model that maps visual features to semantic space and reconstructs them back for zero-shot learning.
    - Enforced a reconstruction constraint to ensure the learned embeddings preserve semantic consistency.
    - Demonstrated that SAE improves alignment between visual and semantic domains, enhancing zero-shot classification accuracy.

4. Y. Xian, et al. "Feature Generating Networks for Zero-Shot Learning", CVPR 2018.
    - Proposed using Generative Adversarial Networks (GANs) to synthesize visual features for unseen classes.
    - Converted zero-shot learning into a supervised problem by generating synthetic examples for unseen classes.
    - Showed significant improvements in zero-shot learning performance by augmenting training data with generated features.

## Few-shot Learning

1. J. Snell, K. Swersky, and R. Zemel, "Prototypical Networks for Few-Shot Learning," NIPS 2017.

    - Introduced Prototypical Networks, which learn a metric space where classification is performed by computing distances to prototype representations of each class.
    - Simplified the few-shot learning process by averaging embeddings to form class prototypes.
    - Achieved strong performance on few-shot classification benchmarks with a simple and efficient framework.

2. Vinyals, O., et al. "Matching Networks for One Shot Learning", NIPS 2016.

    - Proposed Matching Networks that utilize attention and memory mechanisms for one-shot learning.
    - Employed a differentiable nearest-neighbor classifier to compare query examples with a small support set.
    - Demonstrated that leveraging metric learning and embedding functions improves one-shot classification.

3. Chelsea Finn, et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks," ICML 2017.

    - Presented Model-Agnostic Meta-Learning (MAML), an algorithm that enables models to adapt quickly to new tasks with minimal training data.
    - Designed to be compatible with any model trained with gradient descent, making it widely applicable.
    - Showed that MAML can achieve rapid learning on a variety of tasks, outperforming traditional meta-learning approaches.

4. F. Sung, et al. "Learning to Compare: Relation Network for Few-Shot Learning," CVPR 2018.
    - Introduced Relation Networks that learn a deep distance metric for comparing query images with support examples.
    - Modeled relationships between images using a neural network to compute similarity scores.
    - Improved few-shot classification accuracy by focusing on learning relations rather than absolute representations.

## GANs

1. **Isola, Phillip, et al. "Image-to-Image Translation with Conditional Adversarial Networks," CVPR 2017.**

    - Introduced Pix2Pix framework for supervised image-to-image translation using conditional GANs
    - Demonstrated effectiveness on multiple tasks including edges→photos, labels→scenes, and maps→aerial photos
    - Combined adversarial loss with L1 loss to produce both realistic and accurate translations
    - Proposed PatchGAN discriminator that penalizes structure at the scale of image patches

2. **Ho, Jonathan, et al. "Denoising Diffusion Probabilistic Models," NeurIPS 2020.**

    - Introduced DDPM, a new approach to generative modeling that combines the best aspects of VAEs and GANs
    - Proposed a diffusion process that gradually adds noise to data and learns to reverse this process
    - Achieved state-of-the-art image generation quality through simple training objective
    - Demonstrated superior sample diversity compared to GANs while maintaining high quality
    - Introduced diffusion models for high-quality image synthesis, offering an alternative to GANs.
    - Modeled the data distribution by iteratively denoising samples from a Gaussian distribution.
    - Achieved state-of-the-art results in image generation with better diversity and fidelity.

3. Zhu, Jun-Yan, et al. "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks," ICCV 2017.

    - Proposed CycleGAN, enabling image-to-image translation without paired training data.
    - Introduced cycle-consistency loss to enforce that the translation from one domain to another is invertible.
    - Achieved impressive results in style transfer and domain adaptation tasks using unpaired datasets.

4. Karras, Tero, et al. "A Style-Based Generator Architecture for Generative Adversarial Networks," CVPR 2019.

    - Developed StyleGAN, introducing a new generator architecture that controls image synthesis through style vectors at each convolution layer.
    - Enabled explicit control over features at different levels, such as overall style and fine details.
    - Produced high-resolution, photorealistic images with unprecedented quality and variation.

5. Tero Karras, et al. "Progressive Growing of GANs for Improved Quality, Stability, and Variation," ICLR 2018.

    - Introduced a training methodology where both the generator and discriminator are progressively grown, starting from low resolution.
    - Improved training stability and reduced training time for high-resolution image generation.
    - Achieved state-of-the-art results in generating high-quality images with increased diversity.

6. Choi, Yunjey, et al. "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation," CVPR 2018.
    - Proposed StarGAN, capable of performing image-to-image translation across multiple domains using a single model.
    - Incorporated domain labels to guide the translation process, enabling flexible and scalable multi-domain translation.
    - Simplified the architecture by avoiding the need for multiple models for each domain pair.

## Deep Clustering

1. Xie, Junyuan, et al. "Unsupervised Deep Embedding for Clustering Analysis," ICML 2016.

    - Introduced Deep Embedded Clustering (DEC), which jointly learns feature representations and cluster assignments.
    - Utilized deep autoencoders to transform data into a lower-dimensional space suitable for clustering.
    - Iteratively refined clusters by minimizing a KL divergence loss, improving clustering performance over traditional methods.

    Yang, Bo, et al. "Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering," ICML 2017.

    - Proposed a framework that integrates deep learning with K-means clustering.
    - Learned transformations that make data more amenable to clustering algorithms.
    - Demonstrated improved clustering accuracy by jointly optimizing representation learning and clustering objectives.

2. M. Grootendorst, "BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure," arXiv 2022.

    - Introduced BERTopic, a topic modeling technique leveraging BERT embeddings and class-based TF-IDF.
    - Combined transformer-based embeddings with traditional clustering to extract coherent topics from text data.
    - Improved topic interpretability and coherence over classical topic modeling methods like LDA.

3. Shaham, Uri, et al. "SpectralNet: Spectral Clustering Using Deep Neural Networks," ICLR 2018.
    - Presented SpectralNet, which approximates spectral clustering using neural networks for scalability.
    - Learned spectral embeddings without explicitly computing pairwise similarities or eigenvectors.
    - Enabled spectral clustering to be applied to large datasets with high efficiency.

## Transfer Learning and Domain Adaptation

1. Z. Li and D. Hoiem, "Learning without Forgetting," ECCV 2016.

    - Introduced a method to retain knowledge of old tasks while learning new ones without accessing old data.
    - Employed knowledge distillation to prevent catastrophic forgetting in neural networks.
    - Allowed models to incrementally learn new classes while preserving performance on previously learned classes.

2. B. Fernando, et al., "Unsupervised Visual Domain Adaptation Using Subspace Alignment," ICCV 2013.

    - Proposed Subspace Alignment (SA) for unsupervised domain adaptation by aligning source and target subspaces.
    - Reduced domain discrepancy by mapping source data into the target domain's feature space.
    - Demonstrated effectiveness in cross-domain object recognition tasks.

3. Judy Hoffman, et al., "CyCADA: Cycle-Consistent Adversarial Domain Adaptation," ICML 2018.
    - Introduced CyCADA, combining cycle-consistent GANs with domain adaptation techniques.
    - Aligned both pixel-level and feature-level representations across domains.
    - Improved performance in adapting models trained on synthetic data to real-world data.

## Self-supervised Learning

1. T. Chen, et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020.

    - Presented SimCLR, a framework that uses contrastive learning with data augmentation to learn visual representations.
    - Showed that larger batch sizes and stronger augmentations significantly improve performance.
    - Achieved results comparable to supervised learning on image classification tasks without using labeled data.

2. K. He, et al., "Momentum Contrast for Unsupervised Visual Representation Learning," CVPR 2020.

    - Introduced MoCo, employing a dynamic dictionary and a momentum encoder for contrastive learning.
    - Addressed the memory constraints of contrastive learning by maintaining a queue of representations.
    - Outperformed previous methods in unsupervised feature learning on various benchmarks.

3. J. Grill, et al., "Bootstrap Your Own Latent (BYOL): A New Approach to Self-Supervised Learning," NeurIPS 2020.

    - Proposed BYOL, a self-supervised method that learns representations by predicting target network outputs without negative samples.
    - Used two neural networks (online and target) that interact and learn from each other.
    - Demonstrated that high-quality representations can be learned without contrastive losses.

4. M. Caron, et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments," NeurIPS 2020.
    - Introduced SwAV, combining clustering with contrastive learning to learn unsupervised image representations.
    - Employed online clustering to assign pseudo-labels for contrastive learning.
    - Achieved competitive performance on image classification tasks without supervised labels.

## Interpretability of ML

1. Marco Tulio Ribeiro, et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier," KDD 2016.

    - Introduced LIME (Local Interpretable Model-agnostic Explanations), explaining individual predictions of any black-box model.
    - Provided a method to understand complex models by approximating them locally with interpretable models.
    - Enhanced transparency and trust in machine learning models by offering clear, understandable explanations.

2. Scott Lundberg and Su-In Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.

    - Presented SHAP (SHapley Additive exPlanations), unifying several interpretability methods under game-theoretic principles.
    - Provided consistent and accurate feature attribution for any machine learning model.
    - Enabled detailed insights into model behavior, improving interpretability and debugging.

3. **R. R. Selvaraju, et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV 2017**

    - Developed Grad-CAM, a technique to produce visual explanations for decisions made by CNNs.
    - Generated class-specific heatmaps highlighting important regions in input images.
    - Improved interpretability of deep models in tasks like image classification and captioning.

4. Been Kim, et al., "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)," ICML 2018.

    - Introduced TCAV, a method to interpret neural networks using human-friendly concepts.
    - Quantified the influence of high-level concepts on model predictions without requiring access to the training data.
    - Enabled users to test models for biases and ensure they align with human reasoning.

5. Oscar Li, et al., "Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions," AAAI 2018.

    - Proposed a neural network that makes predictions based on learned prototypes, akin to case-based reasoning.
    - Enhanced interpretability by showing which prototypes most influenced a given prediction.
    - Bridged the gap between deep learning and interpretable, example-based reasoning.

6. C. Chen, et al., "This Looks Like That: Deep Learning for Interpretable Image Recognition," NeurIPS 2019.

    - Developed a prototype-based network that associates parts of an input image with learned prototypes.
    - Provided visual explanations by highlighting image regions similar to prototypes.
    - Maintained high classification accuracy while offering transparent decision-making processes.

7. Marco Ribeiro, et al., "Anchors: High-Precision Model-Agnostic Explanations," AAAI 2018.
    - Introduced Anchors, providing high-precision rules that explain individual model predictions.
    - Generated model-agnostic explanations that are both interpretable and faithful to the original model.
    - Improved user trust by offering concise and understandable decision rules.

## Transformers, Language Models

1. **Devlin, Jacob, et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL 2019.**

    - Introduced BERT, a pre-trained deep bidirectional Transformer model for NLP tasks.
    - Utilized masked language modeling and next sentence prediction for effective pre-training.
    - Achieved state-of-the-art results on multiple NLP benchmarks through fine-tuning.

2. Radford, Alec, et al., "Improving Language Understanding by Generative Pre-Training," 2018.

    - Presented GPT, showcasing the power of unsupervised pre-training for language understanding.
    - Demonstrated that fine-tuning a generatively pre-trained model improves performance on various tasks.
    - Pioneered transformer-based models in NLP, influencing future language model architectures.

3. X. L. Li and P. Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation," ACL 2021.

    - Proposed prefix-tuning, a lightweight method to adapt pre-trained language models by optimizing continuous prompts.
    - Reduced the number of trainable parameters needed for fine-tuning large models.
    - Enabled efficient adaptation to downstream tasks without full model retraining.

4. **Hu, Edward J., et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.**

    - Introduced LoRA, a technique for fine-tuning language models by injecting trainable rank-decomposition matrices.
    - Significantly reduced memory requirements and computational costs during adaptation.
    - Maintained performance while only updating a small fraction of the model's parameters.

5. Kojima, Takeshi, et al., "Large Language Models are Zero-Shot Reasoners," NeurIPS 2022.

    - Showed that large language models can perform reasoning tasks without task-specific training.
    - Demonstrated that appropriate prompting enables models to solve problems in a zero-shot setting.
    - Highlighted the emergent reasoning capabilities of large-scale pre-trained models.

6. **Lewis, Patrick, et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020.**

    - Introduced RAG, combining retrieval and generation for knowledge-intensive tasks
    - Used dense passage retrieval with neural generation
    - Enabled end-to-end training of retrieval and generation components
    - Demonstrated strong performance on question answering and fact verification

## Vision Language Models

1. **A. Dosovitskiy, et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021.**

    - Introduced the Vision Transformer (ViT), applying Transformer architectures directly to image recognition.
    - Demonstrated that Transformers can outperform convolutional networks when trained on large datasets.
    - Opened new avenues for using Transformer models in computer vision tasks.

2. A. Radford, et al., "Learning Transferable Visual Models from Natural Language Supervision," ICML 2021.

    - Presented CLIP, a model trained on image-text pairs using contrastive learning.
    - Achieved strong zero-shot performance on various visual classification tasks.
    - Bridged vision and language domains, enabling models to understand and relate visual content to natural language.

3. T. Brooks, A. Holynski, and A. A. Efros, "InstructPix2Pix: Learning to Follow Image Editing Instructions," CVPR 2023.
    - Developed InstructPix2Pix, enabling image editing based on textual instructions.
    - Combined language understanding with image-to-image translation for interactive image manipulation.
    - Demonstrated the ability to perform complex edits guided by natural language prompts.
