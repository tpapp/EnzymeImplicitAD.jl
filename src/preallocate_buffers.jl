#####
##### preallocate buffers
#####

public preallocate_buffers

struct PreallocateBuffers{I,B} <: WrappedImplicitProblem
    inner_problem::I
    buffers::B
end

function implicit_solve_with_initial_guesses!(y, implicit_problem::PreallocateBuffers, x;
                                              initial_guesses)
    implicit_solve_with_initial_guesses!(y, implicit_problem.inner_problem, x;
                                         initial_guesses)
end

function Base.show(io::IO, implicit_problem::PreallocateBuffers)
    println(io, "preallocating buffers for\n  ", implicit_problem.inner_problem)
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

get_buffers(implicit_problem::PreallocateBuffers) = implicit_problem.buffers
