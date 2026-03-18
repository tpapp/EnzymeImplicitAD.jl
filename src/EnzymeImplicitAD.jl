"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

export SquareImplicitFunction

using DocStringExtensions: SIGNATURES
using LinearAlgebra: ldiv!, lu!

import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, FwdConfig
using Enzyme: Forward, autodiff, make_zero!

"""
An implicit function ``y = f(x)`` defined by ``0 = g(x, y)``, with the call signature
`f!(y, x)` and `g!(r, x, y)` where ``r`` is approximately zero for the solution.
"""
struct SquareImplicitFunction{F,G}
    f!::F
    g!::G
end

function (ℐ::SquareImplicitFunction)(y, x)
    ℐ.f!(y, x)
    nothing
end

function (ℐ::SquareImplicitFunction)(x)
    y = similar(x)
    ℐ(y, x)
    y
end

"""
$(SIGNATURES)
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
"""
function inplace_∂g∂x_v!(Jv, g!, r, x, v, y)
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, Const(g!), Duplicated(r, Jv), Duplicated(x, v), Const(y))
end

function forward(config::FwdConfig, ℐ::Const{<:SquareImplicitFunction}, RT::Type,
                 Dy::Union{Const,Duplicated}, Dx::Union{Const,Duplicated})
    (; f!, g!) = ℐ.val
    println("Using custom FORWARD rule")
    @show RT
    y = Dy.val
    x = Dx.val
    f!(y, x)
    if Dx isa Const && Dy isa Const
        return nothing
    elseif Dy isa Const
        error("how can this happen")
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
    inplace_∂g∂x_v!(dy, g!, r, x, dx, y)  # now dy = ∂g/∂x ⋅ dx
    ldiv!(lu!(J), dy)
    dy .*= -1
    nothing
end

end # module
