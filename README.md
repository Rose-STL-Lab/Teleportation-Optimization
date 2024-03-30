## Paper
Bo Zhao, Robert M. Gower, Robin Walters, Rose Yu. [Improving Convergence and Generalization Using Parameter Symmetries](https://arxiv.org/abs/2305.13404). *International Conference on Learning Representations (ICLR)*, 2024.

## Abstract
In many neural networks, different values of the parameters may result in the same loss value. Parameter space symmetries are loss-invariant transformations that change the model parameters. Teleportation applies such transformations to accelerate optimization. However, the exact mechanism behind this algorithm's success is not well understood. In this paper, we show that teleportation not only speeds up optimization in the short-term, but gives overall faster time to convergence. Additionally, teleporting to minima with different curvatures improves generalization, which suggests a connection between the curvature of the minimum and generalization ability. Finally, we show that integrating teleportation into a wide range of optimization algorithms and optimization-based meta-learning improves convergence. Our results showcase the versatility of teleportation and demonstrate the potential of incorporating symmetry in optimization.

## Requirements 
* [PyTorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/)
* [SciPy](https://scipy.org/install/)
* [Shapely](https://shapely.readthedocs.io/en/stable/)

## Initializing directories
```
python init_directory.py
```

## Reproducing experiments in the paper
Correlation between sharpness/curvature and validation loss (Table 1, Figure 9, Figure 10):

```
python correlation.py
```

How curvature influences the expected displacement of minima under distribution shifts (Figure 7):

```
python displacement_integration.py
```

Teleportation to change sharpness or curvature (Figure 4):

```
python teleport_sharpness_curvature.py
```

Integrating teleportation with various optimizers (Figure 5, Figure 13):

```
python teleport_optimization.py
```

Meta-learning (Figure 6):

```
cd learn-to-teleport
python multi_layer_regression.py
```
Figures are saved in directories `figures/` and `learn-to-teleport/figures/`.

## Cite
```
@article{zhao2024improving,
  title={Improving Convergence and Generalization Using Parameter Symmetries},
  author={Bo Zhao and Robert M. Gower and Robin Walters and Rose Yu},
  journal={International Conference on Learning Representations},
  year={2024}
}
``` 