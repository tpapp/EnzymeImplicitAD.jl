#####
##### benchmarks
#####

public benchmark_and_stresstest, Benchmarks

import Random

####
#### timings
####

mutable struct Timings{Float64}
    count::Int
    average_time::Float64
    average_gc_time::Float64
end

Timings() = Timings(0, 0.0, 0.0)

struct _PrintSeconds{T<:Real}
    t::T
    sigdigits::Int
end

_PrintSeconds(s::Real; sigdigits::Int = 2) = _PrintSeconds(s, sigdigits)

function Base.show(io::IO, ps::_PrintSeconds)
    (; t, sigdigits) = ps
    if t ≤ 1e-8
        unit = "ns"
        t *= 1e9
    elseif t ≤ 1e-5
        unit = "μs"
        t *= 1e6
    elseif t ≤ 1e-2
        unit = "ms"
        t *= 1e3
    else
        unit = "s"
    end
    print(io, round(t; sigdigits), unit)
end

function Base.show(io::IO, timings::Timings)
    (; count, average_time, average_gc_time) = timings
    print(io, "$(count) evaluations, average time ", _PrintSeconds(average_time),
          ", average gc time ", _PrintSeconds(average_gc_time))
end

function update!(timings::Timings, timed, _x)
    timings.count += 1
    timings.average_time += (timed.time- timings.average_time) / timings.count
    timings.average_gc_time += (timed.gctime- timings.average_gc_time) / timings.count
    nothing
end

####
#### errors
####

"Structure for saving errors and nice printing."
Base.@kwdef struct ErrorAt
    x::Any
    error::Any
    backtrace::Any
end

function Base.show(io::IO, e::ErrorAt)
    (; x, error, backtrace) = e
    print(io, "error at x = ", x, "\n")
    showerror(io, error, backtrace)
end

####
#### benchmarks
####

"""
$(TYPEDEF)

**Fields are part of the API.**

$(FIELDS)
"""
Base.@kwdef struct Benchmarks
    "timings for [`implicit_solve!`](@ref)"
    implicit_solve_timings::Timings
    "vector of errors recorded for [`implicit_solve!`](@ref)"
    implicit_solve_errors::Vector{ErrorAt}
    "timings for [`calculate_∂y∂x`](@ref)"
    calculate_∂y∂x_timings::Timings
    "vector of errors recorded for [`calculate_∂y∂x`](@ref)"
    calculate_∂y∂x_errors::Vector{ErrorAt}
end

function Base.show(io::IO, benchmarks::Benchmarks)
    (; implicit_solve_timings, implicit_solve_errors,
     calculate_∂y∂x_timings, calculate_∂y∂x_errors) = benchmarks
    print(io, "implicit solve timings: ", implicit_solve_timings)
    print(io, "\n    ", length(implicit_solve_errors), " implicit solve errors")
    print(io, "\ncalculate ∂y∂x timings: ", calculate_∂y∂x_timings)
    print(io, "\n    ", length(calculate_∂y∂x_errors), " calculate ∂y∂x errors")
end

"""
$(SIGNATURES)
"""
function benchmark_and_stresstest(implicit_problem;
                                  count = 1000,
                                  max_errors = 20,
                                  worst_solver = max(20, div(count, 10, RoundUp)),
                                  worst_∂y∂x = max(20, div(count, 10, RoundUp)),
                                  rng = Random.default_rng())
    implicit_solve_errors = Vector{ErrorAt}()
    calculate_∂y∂x_errors = Vector{ErrorAt}()
    (; n_x, n_y, n_r) = get_dimensions(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    x = Vector{Float64}(undef, n_x)
    y = Vector{Float64}(undef, n_y)
    r = Vector{Float64}(undef, n_r)
    implicit_solve_timings = Timings()
    calculate_∂y∂x_timings = Timings()
    for j in 0:count            # j = 0 extra evaluation for the compiler
        @. x = randn()
        # solver
        try
            𝒯 = @timed implicit_solve!(y, implicit_problem, x)
            j > 0 && update!(implicit_solve_timings, 𝒯, x) # don't record j = 0
            # ∂y∂x
            try
                𝒯 = @timed calculate_∂y∂x(implicit_problem, x, y)
                j > 0 && update!(calculate_∂y∂x_timings, 𝒯, x) # don't record j = 0
            catch e
                if length(calculate_∂y∂x_errors) < max_errors
                    push!(calculate_∂y∂x_errors, ErrorAt(; x, error = e, backtrace = catch_backtrace()))
                else
                    break
                end
            end
        catch e
            if length(implicit_solve_errors) < max_errors
                push!(implicit_solve_errors, ErrorAt(; x, error = e, backtrace = catch_backtrace()))
            else
                break
            end
        end
    end
    Benchmarks(; implicit_solve_timings, implicit_solve_errors,
               calculate_∂y∂x_timings, calculate_∂y∂x_errors)
end
