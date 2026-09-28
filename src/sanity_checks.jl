#####
##### utilities for testing API conformance
#####

public API_sanity_checks

using LinearAlgebra: norm

####
#### sanity checks
####

"""
Implementation for `API_sanity_checks`, not part of the API.

Each field is either `missing` (test not performed), `nothing` (test passed), or an
error-backtrace pair.
"""
Base.@kwdef struct SanityChecks
    check_dimensions
    check_eltype
    check_initial_guess
    check_implicit_solve
    check_implicit_residuals
    check_task_local_buffers
    check_∂y∂x
    check_statistics
end

"""
$(SIGNATURES) → checks

Check that the interface implemented to `implicit_problem` conforms to the expected API.

Checks are not necessarily comprehensive, and may change without major version changes.
The user can access the property `checks.all_ok::Bool`, the rest of the fields can be
used for debugging but are not part of the API.

# Keyword arguments

- `residual_l2norm`: the Euclidean norm used to check the residual
"""
function API_sanity_checks(implicit_problem; residual_l2norm = √eps())
    # initialize sanity checks
    check_dimensions = missing
    check_eltype = missing
    check_initial_guess = missing
    check_implicit_solve = missing
    check_implicit_residuals = missing
    check_task_local_buffers = missing
    check_∂y∂x = missing
    check_statistics = missing
    local T, n_x, n_y, n_r, x, y

    # dimensions
    try
        (; n_x, n_y, n_r) = get_dimensions(implicit_problem)
        @argcheck n_x isa Int && n_x > 0
        @argcheck n_y isa Int && n_y > 0
        @argcheck n_r isa Int && n_r > 0
        solution_concept = get_solution_concept(implicit_problem) # test that it is defined
        if solution_concept ≡ ZeroResiduals()
            @argcheck n_y == n_r
        end
        check_dimensions = nothing
    catch e
        check_dimensions = (e, catch_backtrace())
        @goto done
    end

    # eltype
    try
        T = get_preferred_eltype(implicit_problem)
        @argcheck T <: AbstractFloat
        check_eltype = nothing
    catch e
        check_eltype = (e, catch_backtrace())
        @goto done
    end

    # initial guess
    try
        x = randn(T, n_x)
        y = fill(T(NaN), n_y)
        initial_guesses = get_initial_guesses(implicit_problem, x)
        for initial_guess in initial_guesses
            @argcheck initial_guess isa AbstractVector
            @argcheck all(isfinite, initial_guess)
        end
        check_initial_guess = nothing
    catch e
        check_initial_guess = (e, catch_backtrace())
        @goto done
    end

    # implicit solve
    try
        implicit_solve!(y, implicit_problem, x)
        @argcheck all(isfinite, y)
        check_implicit_solve = nothing
    catch e
        check_implicit_solve = (e, catch_backtrace())
        @goto done
    end

    # implicit residuals
    try
        r = fill(T(NaN), n_y)
        @argcheck implicit_residuals!(r, implicit_problem, x, y) ≡ nothing
        @argcheck norm(r, 2) ≤ residual_l2norm
        check_implicit_residuals = nothing
    catch e
        check_implicit_residuals = (e, catch_backtrace())
        @goto done
    end

    # task local buffers
    try
        buffers = task_local_buffers(implicit_problem)
        function _check_y_buffer(b, n)
            b[1] += one(T)      # check mutability
            @argcheck b isa AbstractVector
            @argcheck eltype(b) ≡ T
            @argcheck length(b) == n
        end
        _check_y_buffer(buffers.buffer_x, n_x)
        _check_y_buffer(buffers.buffer_y, n_y)
        _check_y_buffer(buffers.buffer_r, n_r)
        _check_y_buffer(buffers.buffer_r2, n_r)
        check_task_local_buffers = nothing
    catch e
        check_task_local_buffers = (e, catch_backtrace())
        @goto done
    end

    # ∂y∂x
    try
        ∂Y∂X = get_∂y∂x_type(implicit_problem)
        @argcheck isconcretetype(∂Y∂X)
        ∂y∂x = calculate_∂y∂x(implicit_problem, x, y)
        @argcheck ∂y∂x isa ∂Y∂X
        dx = similar(x)
        dy = similar(y)
        # pushforward
        dx .= one(T) / 2
        calculate_pushforward!(dy, implicit_problem, x, y, ∂y∂x, dx)
        @argcheck all(isfinite, dy)
        # pullback
        accumulate_pullback!(dx, implicit_problem, x, y, ∂y∂x, dy)
        @argcheck all(isfinite, dx)
        check_∂y∂x = nothing
    catch e
        check_∂y∂x = (e, catch_backtrace())
        @goto done
    end

    # statistics
    try
        @argcheck get_statistics(implicit_problem) isa NamedTuple
        check_statistics = nothing
    catch e
        check_statistics = (e, catch_backtrace())
        @goto done
    end
    # collate and return
    @label done
    SanityChecks(; check_dimensions, check_eltype, check_initial_guess, check_implicit_solve,
                 check_implicit_residuals, check_task_local_buffers, check_∂y∂x,
                 check_statistics)
end

function Base.getproperty(checks::SanityChecks, key::Symbol)
    if key ≡ :all_ok
        all(f -> getfield(checks, f) ≡ nothing,
            fieldnames(SanityChecks))
    else
        getfield(checks, key)
    end
end

function Base.show(io::IO, checks::SanityChecks)
    if checks.all_ok
        printstyled(io, "✔ all checks passed"; bold = true, color = :green)
    else
        printstyled(io, "✘ some checks failed"; bold = true, color = :red)
        for f in fieldnames(SanityChecks)
            e = getfield(checks, f)
            if e ≡ missing
                printstyled(io, "\n  ? ", string(f); color = :yellow)
            elseif e ≡ nothing
                printstyled(io, "\n  ✔ ", string(f); color = :green)
            else
                printstyled(io, "\n  ✘ ", string(f), " :\n"; color = :red)
                showerror(io, e...)
            end
        end
    end
end
