"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

public implicit_solve!, implicit_residuals!

using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using LinearAlgebra: ldiv!, lu!, rdiv!

import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, FwdConfig, RevConfigWidth, overwritten,
    AugmentedReturn
using Enzyme: Forward, Reverse, autodiff, make_zero!

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

# """
# An implicit function ``y = f(x)`` defined by ``0 = g(x, y)``, with the call signature
# `f!(y, x)` and `g!(r, x, y)` where `r` is approximately zero for the solution (`f!` is
# not checked for this).
# """
# struct SquareImplicitFunction{F,G}
#     f!::F
#     g!::G
# end

# """
# $(SIGNATURES)

# In place form for the implicit solution, calculating the first argument from the second.
# """
# function (ℐ::SquareImplicitFunction)(y, x)
#     ℐ.f!(y, x)
#     nothing
# end

# """
# $(SIGNATURES)

# Functional form of the implicit solution, for convenience. Allocates a new vector.
# """
# function (ℐ::SquareImplicitFunction)(x)
#     y = similar(x)
#     ℐ(y, x)
#     y
# end

"""
$(SIGNATURES)

Calculate the Jacobian `J = ∂g/∂y`, at `x` and `y` and write it to the first argument `J`.

The following are assumed to be comformable buffers: `r`, `dr`, `x`, `y`, `dy`, and may
also be overwritten.
"""
function inplace_∂g∂y!(J, implicit_problem, r, dr, x, y, dy::AbstractVector{T}) where T
    make_zero!(dy)
    for i in axes(x, 1)
        make_zero!(dr)          # FIXME do I need this?
        dy[i] = one(T)
        autodiff(Forward, implicit_residuals!, Duplicated(r, dr),
                 Const(implicit_problem), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
    nothing
end

"""
$(SIGNATURES)

Calculate `∂g/∂x ⋅ v` and put the result in the first argument, using forward mode in Enzyme.

`r` will be overwritten.
"""
function inplace_∂g∂x_v!(Jv, v, implicit_problem, r, x, y)
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, implicit_residuals!, Duplicated(r, Jv), Const(implicit_problem),
             Duplicated(x, v), Const(y))
    nothing
end

"""
$(SIGNATURES)

Calculate `v ⋅ ∂g/∂x` and put the result in the first argument, using reverse mode in Enzyme.

NOTE: `r` and `v` are overwritten.
"""
function inplace_v_∂g∂x!(vJ, v, implicit_problem, r, x, y)
    make_zero!(vJ)              # FIXME: do I need this?
    autodiff(Reverse, implicit_residuals!, Duplicated(r, v), Const(implicit_problem),
             Duplicated(x, vJ), Const(y))
    nothing
end

function forward(config::FwdConfig, ::Const{typeof(implicit_solve!)}, ::Type{Const{Nothing}},
                 Dy::Union{Const,Duplicated}, ℐ::Const, Dx::Union{Const,Duplicated})
    implicit_problem = ℐ.val
    y = Dy.val
    x = Dx.val
    implicit_solve!(y, implicit_problem, x)
    if Dx isa Const || Dy isa Const
        if Dy isa Duplicated
            make_zero!(Dy.dval)
        end
        return nothing
    end
    J = similar(y, axes(y, 1), axes(x, 1))
    dx = Dx.dval
    dy = Dy.dval
    r = similar(y)              # FIXME do we need this? or could we use ...NoNeed?
    dr = similar(dy)
    # math:
    #     ∂g/∂x + ∂g/∂y ∂y/∂x = 0
    #     ∂y/∂x ⋅ v = - ∂g/∂y \ ∂g/∂x ⋅ v
    inplace_∂g∂y!(J, implicit_problem, r, dr, x, y, dy) # dy is used as a buffer
    inplace_∂g∂x_v!(dy, dx, implicit_problem, r, x, y)  # now dy = ∂g/∂x ⋅ dx
    ldiv!(lu!(J), dy)
    dy .*= -1
    nothing
end

function augmented_primal(config::RevConfigWidth{1},
                          ::Const{typeof(implicit_solve!)}, ::Type{<:Const},
                          Dy::Duplicated, ℐ::Const, Dx::Duplicated)
    x = Dx.val
    y = Dy.val
    implicit_solve!(y, ℐ.val, x)
    tape = (; y = overwritten(config)[2] ? copy(y) : nothing,
            x = overwritten(config)[3] ? copy(x) : nothing,)
    AugmentedReturn(nothing, nothing, tape) # FIXME do we need a shadow?
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(implicit_solve!)},
                 ::Type{Const{Nothing}}, tape,
                 Dy::Duplicated, ℐ::Const, Dx::Duplicated)
    implicit_problem = ℐ.val
    x = something(tape.x, Dx.val)
    y = something(tape.y, Dy.val)
    dy = Dy.dval
    dx = Dx.dval
    J = similar(y, axes(y, 1), axes(x, 1))
    r = similar(y)
    dr = similar(r)
    buffer = similar(dy)
    # math:
    #     dy ⋅ ∂y/∂x = - (dy / ∂g/∂y) ⋅ ∂g/∂x
    inplace_∂g∂y!(J, implicit_problem, r, dr, x, y, buffer)
    buffer .= dy
    rdiv!(buffer', lu!(J))
    inplace_v_∂g∂x!(r, buffer, implicit_problem, r, x, y) # reuse r
    dx .-= r                                # accumulate into shadow
    make_zero!(dy)                          # zero out y's shadow
    nothing, nothing, nothing
end


end # module
