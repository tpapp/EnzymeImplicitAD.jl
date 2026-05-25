#####
##### solver integration, should be factored out at some point
#####

public square_implicit_problem, initial_guess

using ADTypes: AutoEnzyme
using ArgCheck: @argcheck
using Enzyme: Duplicated
import OhMyThreads
using TrustRegionMethods

"""
Container for square implicit problems defined by `f(x, y(x)) = 0` where `x` and `y`
have the same dimension.

Solved by iterative methods, needs an [`initial guess`](@ref).

This structure is not part of the API.
"""
@concrete terse struct SquareImplicitProblem{T}
    inner_problem
    solver_AD_backend
    buffers
end

"""
$(SIGNATURES)

Wrap `implicit_problem` that implements [`implicit_residuals!`](@ref) and [`initial_guess`](@ref), implementing [`implicit_solve!`](@ref)
"""
function square_implicit_problem(implicit_problem;
                                 solver_AD_backend = AutoEnzyme(; function_annotation = Duplicated),
                                 y_cache_size::Int = 100,
                                 ∂y∂x_cache_size::Int = 100
                                 )
    @argcheck is_square(implicit_problem)
    (; n_y) = get_dimensions(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    By = Vector{T}              # type of buffer
    B_y() = By(undef, n_y)      # make these buffers
    TB = @NamedTuple{buffer_y1::By,buffer_y2::By,buffer_y3::By}
    generate_buffers() = (buffer_y1 = B_y(), buffer_y2 = B_y(), buffer_y3 = B_y())
    buffers =  OhMyThreads.TaskLocalValue{TB}(generate_buffers)
    SquareImplicitProblem{T}(implicit_problem, solver_AD_backend, buffers)
end

get_dimensions(problem::SquareImplicitProblem) = get_dimensions(problem.inner_problem)

function get_preferred_eltype(problem::SquareImplicitProblem)
    get_preferred_eltype(problem.inner_problem)
end

is_square(::SquareImplicitProblem) = true

function implicit_residuals!(r, problem::SquareImplicitProblem, x, y)
    implicit_residuals!(r, problem.inner_problem, x, y)
end

task_local_buffers(problem::SquareImplicitProblem) = problem.buffers[]

"""
$(SIGNATURES)

Provide an initial guess for the inner problem given `x`.
"""
initial_guess(inner_problem, x) = zero(x)

"""
A callable for residuals evaluated at `x`.
"""
@concrete struct _SolverWrap
    problem
    x
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
    stopping_criterion = SolverStoppingCriterion(; residual_norm = tol)
    solution = trust_region_solver(root_problem; stopping_criterion)
    if maximum(abs, solution.residual) > tol
        @info "residuals" solution.residual
        error("residuals too big")
    end
    y .= solution.x
    nothing
end
