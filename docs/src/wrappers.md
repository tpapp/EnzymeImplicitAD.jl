# [Wrappers](@id wrappers)

Wrappers transform an implicit problem by adding a feature. They can be *chained*, which allows the user to add the features that the need, and of course implement their own wrappers.

To demonstrate the features of the wrappers, we define a trivial problem that rejects negative numbers.

```@example dontlikenegative
import EnzymeImplicitAD as EI

"Solving x + y = 0, failure for x < 0."
struct DontLikeNegative end

EI.get_dimensions(::DontLikeNegative) = (; n_x = 1, n_y = 1, n_r = 1)

EI.implicit_residuals!(r, ::DontLikeNegative, x, y) = (r[1] = only(x) + only(y); nothing)

function EI.implicit_solve_with_initial_guesses!(y, P::DontLikeNegative, x; initial_guesses)
    _x = only(x)
    if _x ≥ 0
        y[1] = -_x
        true
    else
        false
    end
end
```

The innermost problem can be accessed easily:

```@example dontlikenegative
P0 = DontLikeNegative()
P = EI.keep_failures(EI.cache_results(EI.preallocate_buffers(P0)))
P0 ≡ EI.innermost_parent(P)
```

Here, the cache helps us when calling the problem with the same arguments again and again:

```@example dontlikenegative
P = EI.cache_results(EI.keep_failures(DontLikeNegative()))
let xs = [-2.0, 3.0, 4.0], y = [0.0]
    for _ in 1:1000
        x = rand(xs)
        EI.implicit_solve!(y, P, [x])
    end
end
EI.get_statistics(P)
```

Note how the failure `[-2.0]` is recorded only once, as caching kicks in before calling the solver of the inner problem.

```@docs
EnzymeImplicitAD.innermost_parent
EnzymeImplicitAD.keep_failures
EnzymeImplicitAD.cache_results
EnzymeImplicitAD.preallocate_buffers
```
