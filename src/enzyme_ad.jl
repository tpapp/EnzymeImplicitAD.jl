#####
##### AD implementation for Enzyme
#####

import Enzyme.EnzymeRules: augmented_primal, forward, reverse
using Enzyme.EnzymeRules: Const, Duplicated, FwdConfig, RevConfigWidth, overwritten,
    AugmentedReturn, needs_primal
using Enzyme: Forward, Reverse, autodiff, make_zero!
using LinearAlgebra: ldiv!, lu!, rdiv!

function forward(_config::FwdConfig{NP}, ::Const{typeof(implicit_solve!)},
                 ::Type{Const{Bool}}, Dy::Union{Const,Duplicated}, ℐ::Const,
                 Dx::Union{Const,Duplicated}) where NP
    implicit_problem = ℐ.val
    y = Dy.val
    x = Dx.val
    success = implicit_solve!(y, implicit_problem, x)
    if Dx isa Const || Dy isa Const
        if Dy isa Duplicated
            make_zero!(Dy.dval)
        end
    elseif !success
        throw(NonSolutionAD(implicit_problem, x))
    else
        ∂y∂x = calculate_∂y∂x(implicit_problem, x, y)
        calculate_pushforward!(Dy.dval, implicit_problem, x, y, ∂y∂x, Dx.dval)
    end
    NP ? success : nothing
end

function augmented_primal(config::RevConfigWidth{1},
                          ::Const{typeof(implicit_solve!)}, ::Type{<:Const},
                          Dy::Duplicated, ℐ::Const, Dx::Duplicated)
    x = Dx.val
    y = Dy.val
    success = implicit_solve!(y, ℐ.val, x)
    tape = (; y = overwritten(config)[2] ? copy(y) : nothing,
            x = overwritten(config)[3] ? copy(x) : nothing,
            success)
    AugmentedReturn(needs_primal(config) ? success : nothing, nothing, tape) # FIXME do we need a shadow?
end

function reverse(_config::RevConfigWidth{1}, ::Const{typeof(implicit_solve!)},
                 ::Type{Const{Bool}}, tape, Dy::Duplicated, ℐ::Const, Dx::Duplicated)
    implicit_problem = ℐ.val
    x = something(tape.x, Dx.val)
    y = something(tape.y, Dy.val)
    if !tape.success
        throw(NonSolutionAD(implicit_problem, x))
    end
    ∂y∂x = calculate_∂y∂x(implicit_problem, x, y)
    accumulate_pullback!(Dx.dval, implicit_problem, x, y, ∂y∂x, Dy.dval)
    make_zero!(Dy.dval)         # zero out y's shadow
    nothing, nothing, nothing
end
