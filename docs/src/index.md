# EnzymeImplicitAD

```@docs
EnzymeImplicitAD
```

## Example

An example implementation of a trivial problem (which, of course, does not require an implicit method):

```@example trivial
import EnzymeImplicitAD as EI

"Solving ``x + y = 0``"
struct TrivialProblem end

EI.get_dimensions(::TrivialProblem) = (; n_x = 1, n_y = 1, n_r = 1)
EI.implicit_residuals!(r, ::TrivialProblem, x, y) = (r[1] = only(x) + only(y); nothing)
EI.implicit_solve_with_initial_guesses!(y, ::TrivialProblem, x; initial_guesses) = (y[1] = -only(x); true)

EI.sanity_checks(TrivialProblem())
```

Note how we implement the method which uses *initial guesses*, which makes [`wrappers`](@ref) easy to use. However, someone *solving* this problem would use

```@example trivial
P = TrivialProblem()
(; n_y) = EI.get_dimensions(P)
y = zeros(n_y)
EI.implicit_solve!(y, P, [1])
y
```

which calculates the initial guesses (where applicable), and passes them on to the implementation above. This is the method through which AD operates.

## Where to go from here

Refer to the documentation of [the implicit problem api](@ref api) for *defining* problems. Note, however, that in most case only a few key methods are needed for defining a problem.

[Benchmarking and sanity checks](@ref) help the usef debug and optimize problem implementations.

[Wrappers](@ref wrappers) add useful functionality to a user-defined problem, including caching solutions and residuals, collecting statistics, preallocating buffers.

In most cases, the user should not have to deal with the [nuts and bolts of automatic differentiation](@ref ad_hooks), but they can be customized.
