# Benchmarking and sanity checks

## Benchmarking

The utilities below help the user benchmark solutions and ``∂y/∂x`` calculations.

```@docs
EnzymeImplicitAD.benchmark_and_stresstest
EnzymeImplicitAD.Benchmarks
```

## [Sanity checks](@id sanity_checks)

Sanity checks help diagnose problems and inconsistencies with implementing the [implicit problem api](@ref api):

```@example
import EnzymeImplicitAD as EI

struct BogusImplementation end

EI.sanity_checks(BogusImplementation())
```

```@docs
EnzymeImplicitAD.sanity_checks
EnzymeImplicitAD.SanityChecks
```
