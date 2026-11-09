# EnzymeImplicitAD

```@docs
EnzymeImplicitAD
```

## Solving implicit problems

The purpose of this package is to isolate an implicit problem in part of the code, and make it differentiable. The key entry point for that is

```@docs
EnzymeImplicitAD.implicit_solve!
```

which can be used in a problem-agnostic manner.

Problems can be queried for the preferred element type of buffers and their dimensions:

```@docs
EnzymeImplicitAD.get_preferred_eltype
EnzymeImplicitAD.get_dimensions
```

See also [Solution concepts](@ref).

## Implementing implicit problems

### Minimal conforming implementation

A *problem* has to implement at least the following methods for the following functions:

1. [`EnzymeImplicitAD.get_dimensions`](@ref),
2. [`EnzymeImplicitAD.implicit_residuals!`](@ref),
3. [`EnzymeImplicitAD.implicit_solve_with_initial_guesses!`](@ref).

An example implementation of a trivial problem (which of course does not require an implicit method):
```@example
import EnzymeImplicitAD as EI

"Solving x + y = 0"
struct TrivialProblem end

EI.get_dimensions(::TrivialProblem) = (; n_x = 1, n_y = 1, n_r = 1)
EI.implicit_residuals!(r, ::TrivialProblem, x, y) = (r[1] = only(x) + only(y); nothing)
EI.implicit_solve_with_initial_guesses!(y, ::TrivialProblem, x; initial_guesses) = (y[1] = -only(x); true)

EI.sanity_checks(TrivialProblem())
```

```@docs
EnzymeImplicitAD.implicit_residuals!
EnzymeImplicitAD.implicit_solve_with_initial_guesses!
```

### Solver


### Optional interface

```@docs
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

## Solution concepts

Currently, the interfact supports only the implicit function formulation. The following API is for future extensions, allowing least squares or fixed point formulations.

```@docs
EnzymeImplicitAD.get_solution_concept
EnzymeImplicitAD.ZeroResiduals
```
