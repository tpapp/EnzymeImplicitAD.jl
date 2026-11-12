# [The implicit problem API](@id api)

The purpose of this package is to isolate an implicit problem in part of the code, and make it differentiable. The table below summarizes the whole API that each problem *must* support. A convenient way to confirm this is with [`sanity_checks`](@ref).


| API function                                                    | fallback           | notes                                              |
|:----------------------------------------------------------------|:-------------------|:---------------------------------------------------|
| [`EnzymeImplicitAD.get_dimensions`](@ref)                       | **no**             | **must be implemented by user**                    |
| [`EnzymeImplicitAD.implicit_residuals!`](@ref)                  | **no**             | **must be implemented by user**                    |
| [`EnzymeImplicitAD.implicit_solve_with_initial_guesses!`](@ref) | **no**             | **must be implemented by user**                    |
| [`EnzymeImplicitAD.independent_copy`](@ref)                     | yes                | default only works for recursively immutable types |
| [`EnzymeImplicitAD.get_initial_guesses`](@ref)                  | yes, empty         | user should provide initial guesses if available   |
| [`EnzymeImplicitAD.get_buffers`](@ref)                          | yes, but allocates | see [`EnzymeImplicitAD.preallocate_buffers`](@ref) |
| [`EnzymeImplicitAD.innermost_parent`](@ref)                     | yes                | useful for [`wrappers`](@ref), default: `identity` |
| [`EnzymeImplicitAD.calculate_∂y∂x`](@ref)                       | yes, via Enzyme    | useful [`ad_hooks`](@ref)                          |
| [`EnzymeImplicitAD.calculate_pushforward!`](@ref)               | yes                | useful [`ad_hooks`](@ref)                          |
| [`EnzymeImplicitAD.accumulate_pullback!`](@ref)                 | yes                | useful [`ad_hooks`](@ref)                          |

The key entry point is

```@docs
EnzymeImplicitAD.implicit_solve!
```

which can be used in a problem-agnostic manner.

Problems can be queried for the preferred element type of buffers and their dimensions:

## Minimal conforming implementation

A *problem* has to implement at least the following methods:

1. [`EnzymeImplicitAD.get_dimensions`](@ref),
2. [`EnzymeImplicitAD.implicit_residuals!`](@ref),
3. [`EnzymeImplicitAD.implicit_solve_with_initial_guesses!`](@ref)

Importantly, this package is agnostic about the *solver*; you can use whichever method you want.

## Dimensions and float types

```@docs
EnzymeImplicitAD.get_preferred_eltype
EnzymeImplicitAD.get_dimensions
```

## Residuals and solvers

These describe the function ``F(x, y)=0`` and the solution method for `y`. Importantly, solutions can *fail*.

```@docs
EnzymeImplicitAD.implicit_solve_with_initial_guesses!
EnzymeImplicitAD.implicit_residuals!
```

## Initial guesses

A good initial guess is very useful when solving an implicit problem numerically. Implementing the method below allows solvers to use these guesses in a convenient manner also via [wrappers](@ref).

```@docs
EnzymeImplicitAD.get_initial_guesses
```

## Threads

Implementations are *not* required to be thread-safe. Rather, the caller should be able to create a thread-local copy.

```@docs
EnzymeImplicitAD.independent_copy
```

## Statistics

While problems are being solved, they may collect statistics about their solution. The user can query these with

```@docs
EnzymeImplicitAD.get_statistics
```
