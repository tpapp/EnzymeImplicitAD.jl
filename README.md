# EnzymeImplicitAD.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/EnzymeImplicitAD.jl/workflows/CI/badge.svg)](https://github.com/tpapp/EnzymeImplicitAD.jl/actions?query=workflow%3ACI)
<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/EnzymeImplicitAD.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/EnzymeImplicitAD.jl/dev)
-->
<!-- Aqua badge, see test/runtests.jl -->
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package implements forward and reverse mode AD via Enzyme.jl for the implicit function

```math
y = f(x) \text{  defined by  } g(x, y) = 0
```

In code, these are implemented as `implicit_solve!(y, implicit_problem, x)` and `implicit_residuals!(r, implicit_probem, x, y)` and should be provided by the user.

It has the following purposes:

1. I want to understand automatic differentiation better.
2. I want to understand how Enzyme works.
3. Serve as an MWE for asking questions.
4. Eventually, contribute to [ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl), once I figure out how the details work.
