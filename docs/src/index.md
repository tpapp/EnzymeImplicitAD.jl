# EnzymeImplicitAD

```@docs
EnzymeImplicitAD
```

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

```@example
import EnzymeImplicitAD as EI
```

## Problem interface

### Core interface

For each implicit problemA *problem* has to implement the methods for the following functions:

```@docs
EnzymeImplicitAD.get_dimensions
EnzymeImplicitAD.implicit_residuals!
EnzymeImplicitAD.implicit_solve_with_initial_guesses!
```

### Solver

```@docs
EnzymeImplicitAD.implicit_solve!
```

```@docs
EnzymeImplicitAD.get_solution_concept
EnzymeImplicitAD.ZeroResiduals
```

### Optional interface

```@docs
EnzymeImplicitAD.get_preferred_eltype
EnzymeImplicitAD.get_statistics
EnzymeImplicitAD.get_buffers
EnzymeImplicitAD.get_initial_guesses
```

## Threads and buffers

```@docs
EnzymeImplicitAD.independent_copy
```


## Wrappers

```@docs
EnzymeImplicitAD.keep_failures
EnzymeImplicitAD.innermost_parent
EnzymeImplicitAD.cache_results
EnzymeImplicitAD.preallocate_buffers
```

## Implementing AD

```@docs
EnzymeImplicitAD.calculate_∂y∂x
EnzymeImplicitAD.calculate_pushforward!
EnzymeImplicitAD.accumulate_pullback!
EnzymeImplicitAD.AD_CHUNKS
```

## Benchmarking and sanity checks

```@docs
EnzymeImplicitAD.benchmark_and_stresstest
EnzymeImplicitAD.Benchmarks
```

```@docs
EnzymeImplicitAD.sanity_checks
EnzymeImplicitAD.SanityChecks
```

## Utilities

```@docs
EnzymeImplicitAD.merge_disjoint
```
