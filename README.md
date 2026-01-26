# DAGAF: A directed acyclic generative adversarial framework for joint structure learning and tabular data synthesis

Authors: Hristo Petkov, Calum MacLellan and Feng Dong

Paper: DAGAF: A directed acyclic generative adversarial framework for joint structure learning and tabular data synthesis (Applied Intelligence, 31 March, 2025)

DAGAF is a novel generative framework for simultaneous causal discovery and tabular data generation.

#### Why use DAGAF?

* Provides a unified framework for causal structure learning and realistic tabular data generation with preserved causality.
* Integrates ANM, LiNGAM, and PNL models through a multi-objective loss to enable robust causal structure learning under diverse data assumptions.
* DAGAF uses a two-step iterative approach that combines causal knowledge acquisition with high-quality data generation. The causal relationships identified in the first step are transferred and leveraged in the second step to facilitate causal-based tabular data generation.
* Validated on synthetic, benchmark, and real-world datasets, DAGAF significantly outperforms state-of-the-art methods in DAG learning while enabling high-quality, realistic data generation.

#### Target Audience
The primary audience for hands-on use of DAGAF are researchers and sophisticated practitioners in Causal Structure Learning, Probabilistic Machine Learning and AI. It is recommended to use the framework as a sequence of steps towards achieving a more accurate approximation of the generative process of data. In other words, users should focus on utilizaing the framework for their own novel research, which may include the following: 1) exploration of different Generative Models; 2) application of different Structural Causal Models; 3) integration of different data modes (e.g. time-series data, mixed data, image, video or sound data) and 4) experimentation with various architectures and hyper-parameters. We hope this framework will bridge the gap between the current state of the causal structure learning field and future contributions.  

## Introduction to DAGAF

We present a novel framework capable of modeling causality resembling the underlying causal mechanisms of the input data and employing them to synthesize diverse, high-fidelity data samples. DAGAF learns multivariate causal structures by applying various functional causal models and determines through experimentation which one best describes the causality in a tabular dataset. Specifically, the framework supports the Post-Nonlinear (PNL) model along with its subsets, which include Linear non-Gaussian Acyclic Models (LiNGAM) and Additive Noise Models (ANM). Unlike other methods that assume data generation is limited to a single causal model, DAGAF satisfies multiple semi-parametric assumptions. 

Furthermore, supporting such a broad spectrum of identifiable models enables us to extensively compare our approach against the state-of-the-art in the field. We complete our study by investigating the quality of the discovered causality from a tabular data generation standpoint. We hypothesize that a precise approximation of the original causal mechanisms in a given probability distribution can be leveraged to produce realistic data samples. To prove our hypothesis, DAGAF incorporates an adversarial tabular data synthesis step, based on transfer learning, into our causal discovery framework. 

For a more detailed theoretical and technical analysis, please read our paper: [H. Petkov, C. MacLellan and F. Dong. DAGAF: A directed acyclic generative adversarial framework for joint structure learning and tabular data synthesis. Applied Intelligence, Springer, 31 March 2025.](https://link.springer.com/article/10.1007/s10489-025-06410-8) 

## Visual aid for understanding critical concepts

We provide users with helpful visualizations (TLDR version of our paper) of the main features of our framework, which include the following: 1) a diagram of our entire framework with different steps included and 2) transition between different forms (basic form -> only working with ANM and LiNGAM; extended form -> working with PNL)

## Insatallation

The easiest way to gain access to our work is to clone the github repo using the following:

```bash
git clone https://github.com/ItsyPetkov/DAGAF.git
cd DAGAF
```

## Setting up your environment

To run our code, users must first create a conda environment using our [environment_setup.yml](environment_setup.yml) file.

To achieve this just run the following:
```
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```
After your environment is configured and activated you are good to go.

## Examples
Here are some basic examples to get you started:

## Citing DAGAF
If you wish to use our framework, please cite the following paper:

```
@article{Petkov2025DAGAFAD,
  title={DAGAF: A directed acyclic generative adversarial framework for joint structure learning and tabular data synthesis},
  author={Hristo Petkov and Calum MacLellan and Feng Dong},
  journal={Applied Intelligence (Dordrecht, Netherlands)},
  year={2025},
  volume={55},
  url = {https://link.springer.com/article/10.1007/s10489-025-06410-8}
}
```

## License
DAGAF is MIT licensed, as found in the [LICENSE](LICENSE) file.
