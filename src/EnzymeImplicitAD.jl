"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

export SquareImplicitFunction

import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, FwdConfig
using Enzyme: Forward, autodiff, make_zero!

"""
An implicit function ``y = f(x)`` defined by ``0 = g(x, y)``, with the call signature
`f!(y, x)` and `g!(r, x, y)` where ``r`` is approximately zero.
"""
struct SquareImplicitFunction{F,G}
    f!::F
    g!::G
end

function inplace_∂r∂y!(J, g!, r, dr, x, y, dy::AbstractVector{T}) where T
    make_zero!(dy)
    for i in axes(x, 1)
        dy[i] = one(T)
        autodiff(Forward, Const(g!), Duplicated(r, dr), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
end


# function forward(config::FwdConfig, I::Const{<:SquareImplicitFunction}, RT::Type,
#                  y::Union{Const,Duplicated}, x::Union{Const,Duplicated})
#     println("Using custom FORWARD rule")
#     @show RT
#     I(y.val, x.val)

# end

end # module
