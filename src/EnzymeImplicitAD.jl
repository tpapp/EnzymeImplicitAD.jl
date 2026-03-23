"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

export SquareImplicitFunction

using DocStringExtensions: SIGNATURES
using LinearAlgebra: ldiv!, lu!, rdiv!

import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, FwdConfig, RevConfigWidth, overwritten,
    AugmentedReturn
using Enzyme: Forward, Reverse, autodiff, make_zero!

"""
An implicit function ``y = f(x)`` defined by ``0 = g(x, y)``, with the call signature
`f!(y, x)` and `g!(r, x, y)` where `r` is approximately zero for the solution (`f!` is
not checked for this).
"""
struct SquareImplicitFunction{F,G}
    f!::F
    g!::G
end

"""
$(SIGNATURES)

In place form for the implicit solution, calculating the first argument from the second.
"""
function (ℐ::SquareImplicitFunction)(y, x)
    ℐ.f!(y, x)
    nothing
end

"""
$(SIGNATURES)

Functional form of the implicit solution, for convenience. Allocates a new vector.
"""
function (ℐ::SquareImplicitFunction)(x)
    y = similar(x)
    ℐ(y, x)
    y
end

"""
$(SIGNATURES)

Calculate the Jacobian `J = ∂g/∂y`, at `x` and `y`.

The following are buffers: `r`, `dr`, `x`, `y`, `dy`, and will be overwritten.
"""
function inplace_∂g∂y!(J, g!, r, dr, x, y, dy::AbstractVector{T}) where T
    make_zero!(dy)
    for i in axes(x, 1)
        make_zero!(dr)          # FIXME do I need this?
        dy[i] = one(T)
        autodiff(Forward, Const(g!), Duplicated(r, dr), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
end

"""
$(SIGNATURES)

Calculate `∂g/∂x ⋅ v` and put the result in the first argument, using forward mode in Enzyme.

`r` will be overwritten.
"""
function inplace_∂g∂x_v!(Jv, v, g!, r, x, y)
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, Const(g!), Duplicated(r, Jv), Duplicated(x, v), Const(y))
end

"""
$(SIGNATURES)

Calculate `v ⋅ ∂g/∂x` and put the result in the first argument, using reverse mode in Enzyme.

NOTE: `r` and `v` are overwritten.
"""
function inplace_v_∂g∂x!(vJ, v, g!, r, x, y)
    make_zero!(vJ)              # FIXME: do I need this?
    autodiff(Reverse, Const(g!), Duplicated(r, v), Duplicated(x, vJ), Const(y))
end

function forward(config::FwdConfig, Dℐ::Const{<:SquareImplicitFunction}, ::Type{Const{Nothing}},
                 Dy::Union{Const,Duplicated}, Dx::Union{Const,Duplicated})
    (; f!, g!) = Dℐ.val
    y = Dy.val
    x = Dx.val
    f!(y, x)
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
    inplace_∂g∂y!(J, g!, r, dr, x, y, dy) # dy is used as a buffer
    inplace_∂g∂x_v!(dy, dx, g!, r, x, y)  # now dy = ∂g/∂x ⋅ dx
    ldiv!(lu!(J), dy)
    dy .*= -1
    nothing
end

function augmented_primal(config::RevConfigWidth{1}, Dℐ::Const{<:SquareImplicitFunction}, RT,
                          Dy::Duplicated, Dx::Duplicated)
    (; f!) = Dℐ.val
    println("Using custom AUGMENTED PRIMAL")
    @show RT
    x = Dx.val
    y = Dy.val
    f!(y, x)
    tape = (; y = overwritten(config)[2] ? copy(y) : nothing,
            x = overwritten(config)[3] ? copy(x) : nothing,)
    AugmentedReturn(nothing, nothing, tape) # FIXME do we need a shadow?
end

function reverse(config::RevConfigWidth{1}, Dℐ::Const{<:SquareImplicitFunction}, ret, tape,
                 Dy::Duplicated, Dx::Duplicated)
    (; g!) = Dℐ.val
    println("Using custom REVERSE rule")
    @show ret
    x = something(tape.x, Dx.val)
    y = something(tape.y, Dy.val)
    dy = Dy.dval
    dx = Dx.dval
    J = similar(y, axes(y, 1), axes(x, 1))
    r = similar(y)
    dr = similar(r)
    buffer = similar(dy)
    # dy ⋅ ∂y/∂x = - (dy / ∂g/∂y) ⋅ ∂g/∂x
    inplace_∂g∂y!(J, g!, r, dr, x, y, buffer)
    buffer .= dy
    rdiv!(buffer', lu!(J))
    inplace_v_∂g∂x!(r, buffer, g!, r, x, y) # reuse r
    dx .+= r                                # accumulate into shadow
    nothing, nothing
end

end # module
