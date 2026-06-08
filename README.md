# EnzymeImplicitAD.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/EnzymeImplicitAD.jl/workflows/CI/badge.svg)](https://github.com/tpapp/EnzymeImplicitAD.jl/actions?query=workflow%3ACI)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package implements forward and reverse mode AD via Enzyme.jl for the implicit function

```math
y = f(x) \text{  defined by  } g(x, y) = 0
```

In code, these are implemented as `implicit_solve!(y, implicit_problem, x)` and `implicit_residuals!(r, implicit_problem, x, y)` and should be provided by the user.

It has the following purposes:

1. I want to understand automatic differentiation better.
2. I want to understand how Enzyme works.
3. Serve as an MWE for asking questions.
4. Eventually, contribute to [ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl), once I figure out how the details work.

## Q & A

### Are you following SemVer?

Technically yes, since the package will be kept at `v0.x`, so everything is assumed to be breaking 😉

### Can I use this in production code?

The code is intended to be usable, with unit tests, and if you commit the manifest it will be reproducible. That said, the project may be abandonned at any point without prior notice, and you should be ready to migrate to a new package when that happens.

### Will this package be registered/release eventually?

Not in this form. If I finish experimentation and arrive at a stable API, it will either be contributed into an existing package, or cleaned up significantly and then released under a different name.
