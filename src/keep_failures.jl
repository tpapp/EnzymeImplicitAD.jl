#####
##### wrapper to keep failed failures
#####

public keep_failures

struct KeepFailures{V,I}
    inner_problem::I
    failure_rate::OnlineMean{Int}
    failures::Vector{V}
end

for f in [:get_dimensions, :get_preferred_eltype, :get_solution_concept,
          :get_buffers, :get_∂y∂x_type]
    @eval ($f)(implicit_problem::KeepFailures) = ($f)(implicit_problem.inner_problem)
end

function implicit_residuals!(r, implicit_problem::KeepFailures, x, y)
    implicit_residuals!(r, implicit_problem.inner_problem, x, y)
end

function implicit_solve_with_initial_guesses!(y, implicit_problem::KeepFailures{V}, x;
                                              initial_guesses) where V
    (; inner_problem, failure_rate, failures) = implicit_problem
    success = implicit_solve_with_initial_guesses!(y, inner_problem, x; initial_guesses)
    update!(failure_rate, !success)
    success || push!(failures, _ensure_typed_copy(V, x))
    success
end

function independent_copy(implicit_problem::KeepFailures)
    (; inner_problem, failure_rate, failures) = implicit_problem
    KeepFailures(independent_copy(inner_problem), independent_copy(failure_rate),
                 copy(failures))
end

function get_initial_guesses(problem::KeepFailures, x)
    get_initial_guesses(problem.inner_problem, x)
end

function get_statistics(implicit_problem::KeepFailures)
    (; inner_problem, failure_rate, failures) = implicit_problem
    merge_disjoint(get_statistics(inner_problem),
                   (; failure_rate = get_mean(failure_rate), failures))
end

function Base.show(io::IO, implicit_problem::KeepFailures)
    (; inner_problem, failure_rate, failures) = implicit_problem
    println(io, "keeping failures for\n  ", inner_problem,
            "\n  failure rate: ", failure_rate)
end

"""
$(SIGNATURES)

Wrapper that keeps the `x`s where [`implicit_solve!`](@ref) failed (ie returned
`false`).

[`get_statistics`](@ref) for this wrapper adds the following properties:

- `failure_rate`: the average failure rate from all calls of `implicit_solve!`,

- `failures`: a vector of `x`s for which solution failed.
"""
function keep_failures(implicit_problem)
    KeepFailures(implicit_problem, online_mean(Int),
                 Vector{Vector{get_preferred_eltype(implicit_problem)}}())
end
