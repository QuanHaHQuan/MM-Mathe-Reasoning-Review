<h1 align="center">Multi-Modal Mathematical Reasoning Review</h1>

**Date:** 2026/12/17

---

## Table of Contents

- [1. Literature Review](#1-literature-review)
  - [1.1 G-LLaVA](#11-g-llava)
    - [Core Abstract & Motivation](#core-abstract--motivation)
    - [Methodology: Data Construction (Geo170K)](#methodology-data-construction-geo170k)
    - [Experimental Setup](#experimental-setup)
    - [Experimental Results & Analysis](#experimental-results--analysis)
    - [Conclusion](#conclusion)
  - [1.2 PRIMITIVE](#12-primitive)
    - [Core Abstract & Motivation](#core-abstract--motivation-1)
    - [Methodology: GeoGLIP & Feature Routing](#methodology-geoglip--feature-routing)
    - [Experimental Setup](#experimental-setup-1)
    - [Experimental Results & Analysis](#experimental-results--analysis-1)
    - [Conclusion](#conclusion-1)
- [2. Benchmark Datasets Reference](#2-benchmark-datasets-reference)
  - [MathVista](#mathvista)
  - [MathVerse](#mathverse)
  - [GeoQA / GeoQA+](#geoqa--geoqa)
  - [Geometry3K](#geometry3k)

---

# 1. Literature Review

## 1.1 G-LLaVA

| Attribute            | Details                                                      |
| :------------------- | :----------------------------------------------------------- |
| **Title**            | G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model |
| **Code**             | [https://github.com/pipilurj/G-LLaVA](https://github.com/pipilurj/G-LLaVA) |
| **Paper**            | [https://arxiv.org/pdf/2312.11370](https://arxiv.org/pdf/2312.11370) |
| **Institution**      | Noah's Ark Lab, HKU, HKUST                                   |
| **Venue**            | ICLR 2024                                                    |
| **Key Contribution** | Geo170K Dataset & G-LLaVA Model                              |

### Core Abstract & Motivation

**The Problem:** Current MLLMs (like GPT-4V) suffer from severe hallucinations in geometry tasks. They typically fail to perceive basic geometric elements (e.g., distinguishing if a point lies on a line or outside it) and struggle with topological relationships. This is largely due to the lack of large-scale, high-quality geometric image-text data.

**The Solution:** The authors construct a massive dataset, **Geo170K**, by leveraging text-only LLMs (ChatGPT) to synthesize visual descriptions and reasoning paths from existing sparse data (Geometry3K, GeoQA).

### Methodology: Data Construction (Geo170K)

![Overview of the data generation pipeline including Alignment and Instruction phases.](Papers/G-LLaVA/figure2_screenshot.png)

#### Phase 1: Geometric Cross-Modal Alignment Data (60K pairs)

*Objective: Teach the model to "see" geometric diagrams accurately.*

1.  **Geometric Image Caption Generation (Inverse Information Recovery):** Since existing datasets lack captions, the authors use ChatGPT to infer the image description based solely on the ground-truth QA pairs (Question + Answer → Caption).
2.  **Contrastive QA Pairs for Basic Elements:** They convert the formal logical forms (available in Geometry3K) into natural language descriptions. Then, they prompt ChatGPT to generate Boolean (Yes/No) questions to test the model's understanding of basic elements (e.g., "Is point D on line BC?").

#### Phase 2: Geometric Instruction Data (110K+ pairs)

*Objective: Teach the model "how to solve" problems.*

The authors utilize four specific strategies to augment the dataset:

1.  **Equation Solving (ES):** Replaces specific numbers in the problem with variables (e.g., length `3` → `x`) and prompts the LLM to derive the solution formula. This forces the model to learn abstract reasoning rather than memorizing numbers.
2.  **Value Scaling (VS):** Scales the length values in the diagram (e.g., x2) while keeping angles invariant to generate new samples. This improves the model's numerical robustness.
3.  **Re-Formulating Condition as Unknown (RCU):** Reverses the problem logic. Instead of "Given A and B, find C", it asks "Given A and C, find B". This reinforces the bidirectional understanding of geometric theorems.
4.  **Sentence Paraphrase (SP):** Paraphrases both the questions and answers into diverse sentence structures. This increases linguistic diversity and prevents the model from overfitting to specific phrasing patterns.

### Experimental Setup

**Datasets:**

* **Training Sources:** Generated from **GeoQA+** (Train set) and **Geometry3K** (Train set).
* **Evaluation Benchmarks:**
  * **MathVista (Minitest Split):** Specifically the Geometry Problem Solving (GPS) subset. Note: Overlapping samples from GeoQA+ were removed to prevent data leakage.
  * **GeoQA (Test Split):** For comparison against domain-specific baselines.

**Implementation Details:**

* **Architecture:** LLaVA framework.
  * *LLM Backbone:* LLaMA-2 (7B and 13B versions).
  * *Visual Encoder:* CLIP ViT-L/14.
  * *Resolution:* 336 x 336 pixels.
* **Training Hyperparameters:**
  * Optimizer: AdamW.
  * Learning Rate: `3e-5`.
  * Batch Size: 32 per GPU (Instruction Phase).
  * Epochs: 1 epoch for Alignment Phase, 2 epochs for Instruction Tuning Phase.
* **Augmentation:** Images are padded to squares with white background; random translation (max 0.25).

### Experimental Results & Analysis

**Result 1: Comparison with Generalist MLLMs (MathVista)**
On the MathVista (GPS split) benchmark, G-LLaVA demonstrates that specialized training on high-quality synthetic data allows smaller models to beat closed-source giants.

* **G-LLaVA-7B (53.4%)** outperforms **GPT-4V (50.5%)** and Gemini Pro (40.4%).
* This validates that "data quality > model size" in the geometry domain.

**Result 2: Comparison with Domain Solvers (GeoQA)**
Compared to previous specialist models (like Geoformer, UniMath), G-LLaVA sets a new SOTA.

* Note that G-LLaVA uses **Top-1 Accuracy** (stricter), while baselines often report Top-10.
* Even under this stricter metric, G-LLaVA-13B achieves **67.0%**, surpassing UniMath (50.0%).

**Result 3: Ablation Study (Alignment Phase)**
The authors conducted an ablation study to verify the importance of the "Alignment Phase" (Phase 1).

* Removing the alignment phase caused a performance drop (64.2% → 62.8%).
* This proves that *explicitly teaching the model to "read" the diagram* (via captions and contrastive QA) is essential before teaching it to solve problems.

### Conclusion

This paper presents G-LLaVA, a specialized MLLM for geometric reasoning. By addressing the data scarcity issue through a novel synthesis pipeline (Geo170K), the authors demonstrate that smaller, domain-specialized models (7B) can outperform large generalist models (GPT-4V) in geometry tasks. The key takeaway is the effectiveness of "Inverse Information Recovery" for visual alignment and the four instruction tuning strategies (ES, VS, RCU, SP) for reasoning.

---

## 1.2 PRIMITIVE

| Attribute            | Details                                                      |
| :------------------- | :----------------------------------------------------------- |
| **Title**            | Primitive Vision: Improving Diagram Understanding in MLLMs   |
| **Code**             | [https://github.com/AI4Math-ShanZhang/SVE-Math](https://github.com/AI4Math-ShanZhang/SVE-Math) |
| **Paper**            | [https://openreview.net/pdf?id=mgbFOJpKY4](https://openreview.net/pdf?id=mgbFOJpKY4) |
| **Institution**      | Australian Institute for Machine Learning, CSIRO, OSU        |
| **Venue**            | ICML 2025                                                    |
| **Key Contribution** | GeoGLIP Visual Encoder & Feature Router                      |

### Core Abstract & Motivation

**The Problem:** While MLLMs have advanced in math reasoning, they suffer from a "fine-grained visual perception bottleneck." A systematic evaluation reveals that GPT-4o exhibits a **70% grounding error rate** on geometric primitives (e.g., failing to identify junctions or boundaries correctly). Correcting these visual errors alone improves reasoning accuracy by 12%.

**The Solution:** Instead of scaling up instruction datasets (like G-LLaVA), the authors propose **PRIMITIVE**, a lightweight plug-and-play module. It introduces **GeoGLIP** (a geometry-grounded visual encoder) and a **Feature Router** to inject precise visual cues (Visual Soft Prompts) into the LLM, enabling it to "see" geometric details like lines and angles accurately.

### Methodology: GeoGLIP & Feature Routing

![The PRIMITIVE architecture.](Papers/GeoGLIP/figure2_screenshot.png)

#### Module 1: GeoGLIP (Geometric-Grounded Language-Image Pre-training)

* **Multi-Task Learning:** It is trained simultaneously on three tasks:
  * *Shape Grounding:* Detecting basic shapes (Triangle, Circle).
  * *Junction Detection:* Identifying intersection points of lines.
  * *Boundary Detection:* Pixel-level segmentation of shapes.
* **Cross-Resolution Mixture:** It fuses low-resolution semantic features with high-resolution spatial features to accurately localize fine-grained elements.

![Designs for the junction and boundary detectors](Papers/GeoGLIP/figure9_screenshot.png)

#### Module 2: Scalable Synthetic Data Generation

![The flow diagram depicts the process for generating synthetic math-specific datasets.](Papers/GeoGLIP/figure6_screenshot.png)

Instead of using expensive human annotation or LLM generation, the authors developed a **Python-based Synthetic Data Engine** (using Matplotlib).

* It programmatically generates random geometric diagrams with perfect ground-truth labels (Bounding Boxes, Junction Coordinates, Segmentation Maps).
* This allows for massive, low-cost pre-training of the GeoGLIP encoder.

#### Module 3: Feature Router & Connector

* **Dual Encoder Strategy:** The model uses both CLIP (for global semantics) and GeoGLIP (for geometric details).
* **Feature Router:** A dynamic "Soft Router" (MLP) weighs the hierarchical features from GeoGLIP. It decides how much "geometric detail" is needed and fuses it with CLIP features (via Channel-wise concatenation) before sending it to the LLM.

### Experimental Setup

**Datasets:**

* **Pre-training (GeoGLIP):** Trained exclusively on **Synthetic Data** (10k images) and FigureQA.
* **Instruction Tuning (MLLM):** Trained on Geo170K and MathV360K.
* **Evaluation Benchmarks:**
  * **MathVerse:** A challenging benchmark for multi-modal reasoning.
  * **MathVista:** A comprehensive benchmark (including FunctionQA, Geometry).
  * **GeoQA:** Domain-specific geometry QA.

**Implementation Details:**

* **Backbones:** Experiments conducted with LLaMA-2-7B, DeepSeek-Math-7B, and Qwen2.5-Math-7B.
* **Training Strategy:**
  * Stage 1: GeoGLIP Pre-training (Visual only).
  * Stage 2: Alignment (Train Projector).
  * Stage 3: Instruction Tuning (Train Projector + LLM).
* **Input Resolution:** 1000 x 1000 for GeoGLIP (high res), 448 x 448 for CLIP.

### Experimental Results & Analysis

**Result 1: Superiority on MathVerse (SOTA for 7B Models)**

* **PRIMITIVE-7B (26.4%)** outperforms the base LLaVA-1.5 (8.8%) and G-LLaVA (7.4%) by a large margin on MathVerse (TestMini).
* It even surpasses larger models or open-source competitors like LLaVA-NeXT-8B (21.2%).

**Result 2: Competitiveness on MathVista**

* **PRIMITIVE-Qwen2.5-7B (50.4%)** achieves performance on par with the closed-source **GPT-4V (50.5%)**.
* This proves that a smaller model with "better eyes" (GeoGLIP) can rival giant models.

**Result 3: Robustness Analysis (Ablation)**

* **Effect of GeoGLIP:** Removing GeoGLIP and using only CLIP drops the accuracy on GeoQA from 67.0% to 64.2%.
* **Effect of Feature Router:** Using a "Soft Router" (dynamic weights) yielded better results (67.0%) compared to a "Constant Router" (66.6%), proving the need for adaptive feature selection.

### Conclusion

This paper shifts the focus from "Data Scaling" to "**Visual Encoding Quality**." By proposing PRIMITIVE with a specialized GeoGLIP encoder, the authors address the root cause of geometric errors: the inability to perceive fine-grained primitives. The success of this method—achieving GPT-4V level performance with a 7B model—suggests that future MLLMs should incorporate domain-specific vision experts rather than relying solely on generalist encoders like CLIP.

---

# 2. Benchmark Datasets Reference

*This section introduces the mainstream datasets mentioned in the reviews above.*

## MathVista

* **Focus:** A comprehensive benchmark for visual mathematical reasoning.
* **Geometry Split (GPS):** The subset specifically focused on geometry problems.
* **Significance:** Often used to compare against generalist LLMs (GPT-4V, Gemini) to test "in-the-wild" reasoning capabilities.

## MathVerse

* **Focus:** A rigorous benchmark designed to evaluate whether MLLMs truly understand visual diagrams or just rely on textual shortcuts.
* **Structure:** It separates text-dominant, vision-dominant, and vision-only samples to test visual grounding capabilities strictly.
* **Significance:** Highlighted in the PRIMITIVE paper as a key metric for evaluating "fine-grained visual perception."

## GeoQA / GeoQA+

* **Focus:** Geometric Question Answering.
* **Scale:** GeoQA+ (expanded version) contains large-scale training data.
* **Use Case:** The standard benchmark for domain-specific geometry solvers.
* **Metrics:** Conventionally Top-10 accuracy, though recent papers (like G-LLaVA) are moving towards Top-1 accuracy.

## Geometry3K

* **Focus:** A dataset featuring human-labeled **formal logic forms** (e.g., `Line(A,B)`).
* **Significance:** Crucial for data generation tasks (as seen in G-LLaVA) because the structured logic allows for programmatic conversion into text descriptions or new questions.
