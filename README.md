# EnzymeImplicitAD.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/EnzymeImplicitAD.jl/dev)
[![build](https://github.com/tpapp/EnzymeImplicitAD.jl/workflows/CI/badge.svg)](https://github.com/tpapp/EnzymeImplicitAD.jl/actions?query=workflow%3ACI)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/tpapp/EnzymeImplicitAD.jl/graph/badge.svg?token=SUM2dileFk)](https://codecov.io/gh/tpapp/EnzymeImplicitAD.jl)


The purpose of this Julia package is to organize solution and automatic differentiation of *implicit problems*, described mathematically as

```math
F(x, y(x)) = 0
```

which define ``y`` as an [implicit function](https://en.wikipedia.org/wiki/Implicit_function) of ``x``.

We use the term “implicit problem” for an *implicit function associated with metadata*, such as problem dimensions, solution statistics.

The api of the package

1. helps *define problems* of this kind with a simple interface, implemented by the user,

2. implements functions for *checking and benchmarking* user implementations,

3. provides various *wrappers* that handle caching of values, recording or failed solutions, preallocation of buffers.

The package *does not export any symbols*, merely makes them `public`. It is recommended that the user *imports* the package with a short alias, such as in

```julia
import EnzymeImplicitAD as EI
```

## Motivation

I wrote this package because:

1. I want to understand automatic differentiation better.
2. I want to understand how Enzyme works.
3. Serve as an MWE for asking questions.
4. Eventually, potentially contribute to [ImplicitDifferentiation.jl](https://github.com/JuliaDecisionFocusedLearning/ImplicitDifferentiation.jl), once I figure out how the details work.

## Q & A

### Are you following SemVer?

Technically yes, since the package will be kept at `v0.x`, so everything is assumed to be breaking 😉

### Can I use this in production code?

The code is intended to be usable, with unit tests, and if you commit the manifest it will be reproducible. That said, the project may be abandonned at any point without prior notice, and you should be ready to migrate to a new package when that happens.

### Will this package be registered/released eventually?

Yes. If I finish experimentation and arrive at a stable API, it will either be contributed into an existing package, or cleaned up significantly and then released under a different name.
