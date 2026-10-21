#####
##### preallocate buffers
#####

public preallocate_buffers

struct PreallocateBuffers
    inner_problem
    buffers
end

for f in [:get_dimensions, :get_preferred_eltype, :get_solution_concept,
          :get_∂y∂x_type, :get_statistics]
    @eval ($f)(implicit_problem::PreallocateBuffers) = ($f)(implicit_problem.inner_problem)
end

get_buffers(implicit_problem::PreallocateBuffers) = implicit_problem.buffers

function implicit_residuals!(r, implicit_problem::PreallocateBuffers, x, y)
    implicit_residuals!(r, implicit_problem.inner_problem, x, y)
end

function implicit_solve_with_initial_guesses!(y, implicit_problem::PreallocateBuffers, x;
                                              initial_guesses)
    implicit_solve_with_initial_guesses!(y, implicit_problem.inner_problem, x;
                                         initial_guesses)
end

function Base.show(io::IO, implicit_problem::PreallocateBuffers)
    (; inner_problem, failure_rate, failures) = implicit_problem
    println(io, "keeping failures for\n  ", inner_problem,
            "\n  failure rate: ", failure_rate)
end

"""
$(SIGNATURES)

A wrapper that preallocates buffers.
"""
function preallocate_buffers(implicit_problem)
    PreallocateBuffers(implicit_problem, get_buffers(implicit_problem))
end

function independent_copy(implicit_problem::PreallocateBuffers)
    preallocate_buffers(independent_copy(implicit_problem.inner_problem))
end
