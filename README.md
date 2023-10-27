# ecnf
Flow matching with vector field parameterized by SE(3) GNN for the paper "SE(3) Equivariant Augmented Coupling Flows".

# Running experiments
```
python examples/dw4.py
python examples/lj13.py
python examples/qm9.py
```

We use wandb for logging - if you use wandb then adjust the config inside examples/config/{problem_name}.yaml appropriately.
Additionally there is a pandas/list logger if you don't use wandb. 
Feel free to contact us if you would like any help running this code.


# Useful links
 - [SE(3) Equivariant Augmented Coupling Flows repo](https://github.com/lollcat/se3-augmented-coupling-flows)
 - [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
 - [Diffrax: SDEs and ODEs in JAX](https://github.com/patrick-kidger/diffrax)



## Citation

```
@article{
midgley2023se3couplingflow,
title={SE(3) Equivariant Augmented Coupling Flows},
author={Laurence Illing Midgley and Vincent Stimper and Javier Antor{\'a}n and Emile Mathieu and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},,
journal={arXiv preprint arXiv:2308.10364}
year={2023},
}
```
