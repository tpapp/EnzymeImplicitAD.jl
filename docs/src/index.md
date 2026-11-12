# EnzymeImplicitAD

```@docs
EnzymeImplicitAD
```

## Example

An example implementation of a univariate cube root (which, of course, Julia already knows how to do with `cbrt`).

```@example uni_cube_root
import EnzymeImplicitAD as EI

"Solving ``x - y^3 = 0`` for single-element vectors."
struct UniCubeRoot end

EI.get_dimensions(::UniCubeRoot) = (; n_x = 1, n_y = 1)

EI.implicit_residuals!(r, ::UniCubeRoot, x, y) = (r[1] = only(x) - only(y)^3; nothing)

EI.get_initial_guesses(P::UniCubeRoot, x) = [x]

function EI.implicit_solve_with_initial_guesses!(y, P::UniCubeRoot, x; initial_guesses)
    _x = only(x)
    _y = first(initial_guesses)[1]
    ϵ = √max(eps(_x), eps(typeof(_x))) # tolerance, up to the solution method
    for _ in 1:precision(_x)
        if abs(_x - _y^3) ≤ ϵ
            y[1] = _y
            return true
        end
        _y = (2*_y + _x / _y^2) / 3
    end
    false # failed to find root within given number of steps, will not happen here
end

EI.sanity_checks(UniCubeRoot())    # do some basic API checks
```

Note how we implement the method which uses *initial guesses*, which makes [`wrappers`](@ref) easy to use. However, someone *using* this implementation to solve problems of this kind would use

```@example uni_cube_root
P = UniCubeRoot()
(; n_y) = EI.get_dimensions(P)
y = zeros(n_y)
EI.implicit_solve!(y, P, [8.0])
y
```

which calculates the initial guesses (where applicable), and passes them on to the implementation above. This is the method through which AD operates.

## Where to go from here

Refer to the documentation of [the implicit problem api](@ref api) for *defining* problems. Note, however, that in most case only a few key methods are needed for defining a problem.

[Benchmarking and sanity checks](@ref) help the user debug and optimize problem implementations.

[Wrappers](@ref wrappers) add useful functionality to a user-defined problem, including caching solutions and residuals, collecting statistics, preallocating buffers.

In most cases, the user should not have to deal with the [nuts and bolts of automatic differentiation](@ref ad_hooks), but they can be customized.
