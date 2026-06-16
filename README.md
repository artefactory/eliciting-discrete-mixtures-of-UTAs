<div align="center">

# Identifying two piecewise linear additive value functions from anonymous preference information


Vincent Auriau<sup>1, 2</sup>, Khaled Belahcène<sup>1</sup>, Emmanuel Malherbe<sup>2</sup>, Vincent Mousseau<sup>1</sup> and Marc Pirlot <sup>3</sup> <br>
<sup>1</sup> <sub>*MICS* - CentraleSupélec,</sub> <sup>2</sup> <sub> Artefact Research Center </sub> <sup>3</sup> <sub> Université de Mons </sub> <br>

</div>

<img align="right" src="./resources/illustration.png">

> *Eliciting a preference model involves asking a person, named decision-maker, a series of questions. We assume that these preferences can be represented by an additive value function. In this work, we query simultaneously  two decision-makers in the aim to elicit their respective value functions. For each query we receive two answers, without noise, but without knowing which answer corresponds to which decision-maker. We propose an elicitation procedure that identifies the two preference models when the marginal value functions are piecewise linear with known breakpoints. We also present experimental results that compare our active elicitation procedure with preference learning settings, showing the efficiency of our procedure.*


## Implementation of the elicitation procedure

An example with randomly drawn decision makers and their elicitation can be found in the notebook [./notebooks/elicitation_implementation.ipynb](./notebooks/elicitation_implementation.ipynb)

## Implementation of the MILO to compute the diameter of the UTA-compatible Space

The MILO implemented with Gurobi to compute $D^2(\Omega)$ can be used as follows:

```python
from python.distances import TwoUTASpaceDiameter

d2_omega = TwoUTASpaceDiameter(n_pieces=5)
d2_omega.fit_generic(
    X=X,
    Y=Y,
    relation_type="preference",
)
```
with $X, Y$ matrices from which each element $x_i$, $y_i$ are related, either $x_i$ is preferred to $y_i$ (in this case, use relation_type="preference") or $x_i$ and $y_i$ are indifferent (in this case use relation_type="indifference").