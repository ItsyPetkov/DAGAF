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
The primary audience for hands-on use of DAGAF are researchers and sophisticated practitioners in Causal Structure Learning, Probabilistic Machine Learning and AI. It is recommended to use the framework as a sequence of steps towards achieving a more accurate approximation of the generative process of data. In other words, users should focus on utilizaing the framework for their own novel research, which may include the following: 1) 

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
DAGAF is MIT licensed, as found in the file.
