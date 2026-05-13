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

"""
$(SIGNATURES)

Calculate the Jacobian `J = ∂g/∂y`, at `x` and `y`, which is assumed to be a valid
solution (not checked).
"""
function _calculate_∂g∂y(implicit_problem, x::AbstractVector, y::AbstractVector{T}) where T
    # allocate buffers (FIXME these could be reused)
    dy = similar(y)
    r = zero(y)
    dr = similar(y)
    # collect column by column
    make_zero!(dy)
    J = similar(y, T, axes(y, 1), axes(x, 1))
    for i in axes(x, 1)
        make_zero!(dr)          # FIXME do I need this?
        dy[i] = one(T)
        autodiff(Forward, implicit_residuals!, Duplicated(r, dr),
                 Const(implicit_problem), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
    J
end

@concrete terse struct ∂Y∂X
    implicit_problem
    x
    y
    ∂g∂y_factor
end

"""
$(SIGNATURES)

Return an object `∂y∂x` that acts like a Jacobian matrix when pre- or post-multiplied by
a conformable vector. It only has to support the five-argument `LinearAlgebra.mul!` and
does not have to be an actual matrix.

This function is required to be *type-stable*.
"""
function calculate_∂y∂x(implicit_problem, x, y)
    ∂Y∂X(implicit_problem, x, y, lu!(_calculate_∂g∂y(implicit_problem, x, y)))
end

"""
$(SIGNATURES)

Calculate `∂g/∂x ⋅ v` and put the result in the first argument, using forward mode in Enzyme.

`r` will be overwritten.
"""
function _inplace_∂g∂x_v!(Jv, v, implicit_problem, x, y, r = similar(y))
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, implicit_residuals!, Duplicated(r, Jv), Const(implicit_problem),
             Duplicated(x, v), Const(y))
    nothing
end

function mul!(Jv::AbstractVector, J::∂Y∂X, v::AbstractVector)
    (; implicit_problem, x, y, ∂g∂y_factor) = J
    _inplace_∂g∂x_v!(Jv, v, implicit_problem, x, y)
    ldiv!(∂g∂y_factor, Jv)
    Jv .*= -1
    Jv
end

"""
$(SIGNATURES)

Calculate `v ⋅ ∂g/∂x` and put the result in the first argument, using reverse mode in Enzyme.

NOTE: `r` and `v` are overwritten.
"""
function _inplace_v_∂g∂x!(vJ, v, implicit_problem, x, y, r = similar(y))
    make_zero!(vJ)              # FIXME: do I need this?
    autodiff(Reverse, implicit_residuals!, Duplicated(r, v), Const(implicit_problem),
             Duplicated(x, vJ), Const(y))
    nothing
end

function mul!(C::AbstractVector, v::AbstractVector, J::∂Y∂X, α::Real, β::Real)
    (; implicit_problem, x, y, ∂g∂y_factor) = J
    # math:
    #     v ⋅ ∂y/∂x = - (v / ∂g/∂y) ⋅ ∂g/∂x
    buffer1 = copy(v)
    buffer2 = similar(v)
    rdiv!(buffer1', ∂g∂y_factor)
    _inplace_v_∂g∂x!(buffer2, buffer1, implicit_problem, x, y)
    @. C = -α * buffer2 + β * C
    C
end
