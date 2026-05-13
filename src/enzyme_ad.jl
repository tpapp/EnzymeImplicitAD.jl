#####
##### AD implementation for Enzyme
#####

import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, FwdConfig, RevConfigWidth, overwritten,
    AugmentedReturn
using Enzyme: Forward, Reverse, autodiff, make_zero!
using LinearAlgebra: ldiv!, lu!, rdiv!

function forward(_config::FwdConfig, ::Const{typeof(implicit_solve!)}, ::Type{Const{Nothing}},
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
    ∂y∂x = calculate_∂y∂x(implicit_problem, x, y)
    mul!(Dy.dval, ∂y∂x, Dx.dval)
    nothing
end

function augmented_primal(config::RevConfigWidth{1},
                          ::Const{typeof(implicit_solve!)}, ::Type{<:Const},
                          Dy::Duplicated, ℐ::Const, Dx::Duplicated)
    x = Dx.val
    y = Dy.val
    implicit_solve!(y, ℐ.val, x)
    tape = (; y = overwritten(config)[2] ? copy(y) : nothing,
            x = overwritten(config)[3] ? copy(x) : nothing)
    AugmentedReturn(nothing, nothing, tape) # FIXME do we need a shadow?
end

function reverse(_config::RevConfigWidth{1}, ::Const{typeof(implicit_solve!)},
                 ::Type{Const{Nothing}}, tape,
                 Dy::Duplicated, ℐ::Const, Dx::Duplicated)
    implicit_problem = ℐ.val
    x = something(tape.x, Dx.val)
    y = something(tape.y, Dy.val)
    ∂y∂x = calculate_∂y∂x(implicit_problem, x, y)
    dy = Dy.dval
    dx = Dx.dval
    O = one(eltype(dx))
    mul!(dx, dy, ∂y∂x, O, O)
    make_zero!(dy)                          # zero out y's shadow
    nothing, nothing, nothing
end
