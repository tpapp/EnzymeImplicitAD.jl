#####
##### the generic API
#####

"""
$(FUNCTIONNAME)(y, implicit_problem, x)

Solve the implicit problem ``g(x, y(x)) = 0`` at `x`, overwriting `y` with ``y(x)`` result.

Return `nothing`. See [`implicit_residuals!`](@ref), which implements ``g`` above.
```
"""
function implicit_solve! end

"""
$(FUNCTIONNAME)(r, implicit_problem, x, y)

Calculate the implicit residuals ``r = g(x, y)``, overwriting `r`.

Return `nothing`.

It is assumed that after
```julia
implicit_solve!(y, implicit_problem, x)
$(FUNCTIONNAME)(r, implicit_problem, x, y)
```
the residuals `r` are “approximately” zero, but this is not checked.
"""
function implicit_residuals! end
