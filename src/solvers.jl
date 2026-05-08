#####
##### solver integration, should be factored out at some point
#####

public square_implicit_problem, initial_guess

using ADTypes: AutoEnzyme
using ArgCheck: @argcheck
using Enzyme: Duplicated
using TrustRegionMethods

"""
Container for square implicit problems defined by `f(x, y(x)) = 0` where `x` and `y`
have the same dimension.

Solved by iterative methods, needs an initial guess.

This structure is not part
"""
struct SquareImplicitProblem{P,A}
    inner_problem::P
    solver_AD_backend::A
end

"""
$(SIGNATURES)

Wrap `problem` that implements [`implicit_residuals!`](@ref) and [`initial_guess`](@ref), implementing [`implicit_solve!`](@ref)
"""
function square_implicit_problem(problem;
                                 solver_AD_backend = AutoEnzyme(; function_annotation = Duplicated))
    SquareImplicitProblem(problem, solver_AD_backend)
end

function implicit_residuals!(r, problem::SquareImplicitProblem, x, y)
    implicit_residuals!(r, problem.inner_problem, x, y)
end

"""
$(SIGNATURES)

Provide an initial guess for the inner problem given `x`.
"""
function initial_guess(inner_problem, x)
    zero(x)
end

struct _SolverWrap{P,X}
    problem::P
    x::X
end

function (w::_SolverWrap)(y)
    (; problem, x) = w
    r = similar(x)
    implicit_residuals!(r, problem, x, y)
    r
end

function implicit_solve!(y, problem::SquareImplicitProblem, x)
    (; inner_problem, solver_AD_backend) = problem
    y0 = initial_guess(inner_problem, x)
    tol = √eps()
    root_problem = trust_region_problem(_SolverWrap(inner_problem, x), y0;
                                        AD_backend = solver_AD_backend)
    solution = trust_region_solver(root_problem; stopping_criterion = SolverStoppingCriterion(; residual_norm = tol))
    if maximum(abs, solution.residual) > tol
        @info "residuals" solution.residual
        error("residuals too big")
    end
    y .= solution.x
    nothing
end
