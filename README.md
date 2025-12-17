[TOC]
# Multi-Modal Mathematical Reasoning Review

**Last Updated:** 17/12/2026

---

## 1. Literature Review

### 1.1 G-LLaVA

| **Meta Information** |                                                              |
| :------------------- | :----------------------------------------------------------- |
| **Title**            | G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model |
| **Code**             | [GitHub Link](https://github.com/pipilurj/G-LLaVA)           |
| **Paper**            | [arXiv PDF](https://arxiv.org/pdf/2312.11370)                |
| **Institution**      | Noah's Ark Lab, HKU, HKUST                                   |
| **Venue**            | ICLR 2024                                                    |
| **Key Contribution** | Geo170K Dataset & G-LLaVA Model                              |

#### Core Abstract & Motivation

* **The Problem:** Current MLLMs (like GPT-4V) suffer from severe hallucinations in geometry tasks. They fail to perceive basic geometric elements (e.g., distinguishing if a point lies on a line) and struggle with topological relationships. This stems from a lack of large-scale, high-quality geometric image-text data.
* **The Solution:** The authors construct a massive dataset, **Geo170K**, by leveraging text-only LLMs (ChatGPT) to synthesize visual descriptions and reasoning paths from existing sparse data (Geometry3K, GeoQA).

#### Methodology: Data Construction (Geo170K)

![Overview of the data generation pipeline](Papers/G-LLaVA/figure2_screenshot.png)
*Figure 1: Overview of the data generation pipeline including Alignment and Instruction phases.*

**Phase 1: Geometric Cross-Modal Alignment Data (60K pairs)** *Objective: Teach the model to "see" geometric diagrams accurately.*

1.  **Geometric Image Caption Generation (Inverse Information Recovery):** Using ChatGPT to infer detailed image descriptions from existing Question-Answer pairs.
2.  **Contrastive QA Pairs:** Converting formal logical forms (from Geometry3K) into natural language, then prompting ChatGPT to generate True/False questions (e.g., "Is point D on line BC?") to test element recognition.

**Phase 2: Geometric Instruction Data (110K+ pairs)** *Objective: Teach the model "how to solve" problems.*

1.  **Equation Solving (ES):** Replacing specific numbers with variables (e.g., $3 \to x$) to teach symbolic derivation.
2.  **Value Scaling (VS):** Scaling lengths (while keeping angles fixed) to improve numerical robustness.
3.  **Re-Formulating Condition as Unknown (RCU):** Reverse questioning (Given Answer, find Condition) to reinforce bidirectional understanding.
4.  **Sentence Paraphrase (SP):** Paraphrasing questions/answers to increase linguistic diversity.

#### Experimental Setup

* **Datasets:** GeoQA+ (Train) and Geometry3K (Train).
* **Benchmarks:** MathVista (GPS Split) and GeoQA (Test Split).
* **Architecture:** LLaVA (LLaMA-2 + CLIP ViT-L/14, 336px resolution).
* **Training:** 1 epoch for Alignment, 2 epochs for Instruction Tuning.

#### Experimental Results & Analysis

* **Result 1 (MathVista):** G-LLaVA-7B (**53.4%**) outperforms GPT-4V (**50.5%**) on the Geometry split, validating that data quality > model size for domain tasks.
* **Result 2 (GeoQA):** G-LLaVA-13B achieves **67.0%** Top-1 Accuracy, surpassing previous SOTA (UniMath 50.0%).
* **Result 3 (Ablation):** Removing the Alignment Phase caused a significant performance drop (64.2% $\to$ 62.8%), proving the necessity of explicit visual perception training.

#### Conclusion

This paper shifts the focus from model scaling to data synthesis. By using "Inverse Information Recovery" and diverse augmentation strategies (ES, VS, RCU), G-LLaVA proves that smaller models can beat generalist giants in geometry if trained on high-quality, synthetic data.

---

### 1.2 PRIMITIVE

| **Meta Information** |                                                              |
| :------------------- | :----------------------------------------------------------- |
| **Title**            | Primitive Vision: Improving Diagram Understanding in MLLMs   |
| **Code**             | [GitHub Link](https://github.com/AI4Math-ShanZhang/SVE-Math) |
| **Paper**            | [OpenReview PDF](https://openreview.net/pdf?id=mgbFOJpKY4)   |
| **Institution**      | Australian Institute for Machine Learning, CSIRO, OSU        |
| **Venue**            | ICML 2025                                                    |
| **Key Contribution** | GeoGLIP Visual Encoder & Feature Router                      |

#### Core Abstract & Motivation

* **The Problem:** MLLMs face a "fine-grained visual perception bottleneck." Evaluations show GPT-4o has a **70% grounding error rate** on geometric primitives. Correcting these visual errors improves reasoning accuracy by 12%.
* **The Solution:** Instead of scaling instruction data, the authors propose **PRIMITIVE**, a plug-and-play module. It introduces **GeoGLIP** (a geometry-grounded visual encoder) and a **Feature Router** to inject precise visual cues (Visual Soft Prompts) into the LLM.

#### Methodology: GeoGLIP & Feature Routing

![The PRIMITIVE architecture](Papers/GeoGLIP/figure2_screenshot.png)
*Figure 2: The PRIMITIVE architecture featuring dual encoders (CLIP + GeoGLIP) and a Feature Router.*

**Module 1: GeoGLIP (Geometric-Grounded Language-Image Pre-training)**

* **Multi-Task Learning:** Trained simultaneously on:
  * *Shape Grounding:* Detecting basic shapes.
  * *Junction Detection:* Identifying intersection points.
  * *Boundary Detection:* Pixel-level segmentation.
* **Cross-Resolution Mixture:** Fuses low-res semantic features with high-res spatial features for precise localization.

![Designs for the junction and boundary detectors](Papers/GeoGLIP/figure9_screenshot.png)
*Figure 3: Detailed design of junction and boundary detectors.*

**Module 2: Scalable Synthetic Data Generation**

* **Synthetic Data Engine:** Using Python (Matplotlib) to generate random geometric diagrams with perfect ground-truth labels (Bounding Boxes, Coordinates). This solves the lack of annotated real-world data.

![Synthetic data generation process](Papers/GeoGLIP/figure6_screenshot.png)
*Figure 4: Flow diagram for generating synthetic math-specific datasets.*

**Module 3: Feature Router & Connector**

* **Dual Encoder Strategy:** Uses CLIP for global semantics and GeoGLIP for geometric details.
* **Feature Router:** A dynamic "Soft Router" (MLP) weighs the hierarchical features from GeoGLIP and fuses them with CLIP features via channel-wise concatenation.

#### Experimental Setup

* **Pre-training:** GeoGLIP trained exclusively on **Synthetic Data** (10k images) + FigureQA.
* **Instruction Tuning:** Trained on Geo170K and MathV360K.
* **Benchmarks:** MathVerse, MathVista, GeoQA.
* **Backbones:** LLaMA-2-7B, DeepSeek-Math-7B, Qwen2.5-Math-7B.

#### Experimental Results & Analysis

* **Result 1 (MathVerse):** **PRIMITIVE-7B (26.4%)** outperforms base LLaVA-1.5 (8.8%) and G-LLaVA (7.4%) significantly, setting a new SOTA for 7B models.
* **Result 2 (MathVista):** **PRIMITIVE-Qwen2.5-7B (50.4%)** achieves parity with closed-source **GPT-4V (50.5%)**.
* **Result 3 (Robustness):** Removing GeoGLIP drops GeoQA accuracy from 67.0% to 64.2%. Using a dynamic "Soft Router" (67.0%) is better than a static "Constant Router" (66.6%).

#### Conclusion

This paper argues for "**Visual Encoding Quality**" over data scaling. By equipping MLLMs with a specialized "eye" (GeoGLIP) trained on cheap synthetic data, PRIMITIVE solves the root cause of geometric errors—perception—allowing 7B models to rival GPT-4V.

---

## 2. Benchmark Datasets Reference

### MathVista

* **Focus:** A comprehensive benchmark for visual mathematical reasoning.
* **Geometry Split (GPS):** The subset specifically focused on geometry problems.
* **Significance:** The standard "in-the-wild" test for comparing against generalist LLMs (GPT-4V, Gemini).

### MathVerse

* **Focus:** A rigorous benchmark designed to evaluate *true* visual understanding vs. textual shortcuts.
* **Structure:** Separates samples into Text-Dominant, Vision-Dominant, and Vision-Only categories.
* **Significance:** Highlighted in the PRIMITIVE paper as a key metric for "fine-grained visual perception."

### GeoQA / GeoQA+

* **Focus:** Geometric Question Answering.
* **Scale:** GeoQA+ contains large-scale training data.
* **Use Case:** The standard benchmark for domain-specific geometry solvers.
* **Metrics:** Conventionally Top-10 accuracy; recent papers (G-LLaVA, PRIMITIVE) prefer Top-1 accuracy.

### Geometry3K

* **Focus:** A dataset featuring human-labeled **formal logic forms** (e.g., `Line(A,B)`).
* **Significance:** Crucial for data generation tasks (G-LLaVA) as the structured logic allows programmatic conversion into text.
