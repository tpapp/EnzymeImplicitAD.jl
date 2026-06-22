#####
##### utilities for testing API conformance
#####

public API_sanity_checks

"""
$(SIGNATURES) → checks

Check that the interface implemented to `implicit_problem` conforms to the expected API.

Checks are not necessarily comprehensive, and may change without major version changes.
The user can access the property `checks.all_ok::Bool`, the rest of the fields can be
used for debugging but are not part of the API.
"""
function API_sanity_checks(implicit_problem)
    # dimensions
    n_x = 0
    n_y = 0
    n_r = 0
    terminate = false
    @_sanity_check terminate check_dimensions begin
        dimensions = get_dimensions(implicit_problem)
        (; n_x, n_y, n_r) = dimensions
        @argcheck n_x isa Int && n_x > 0
        @argcheck n_y isa Int && n_y > 0
        @argcheck n_r isa Int && n_r > 0
        @argcheck (n_y == n_r) == is_square(implicit_problem)
    end
    # eltype
    T = Union{}
    @_sanity_check terminate check_eltype begin
        T = get_preferred_eltype(implicit_problem)
        @argcheck T <: AbstractFloat
    end
    # implicit solve
    x = randn(T, n_x)
    y = fill(T(NaN), n_y)
    @_sanity_check terminate check_implicit_solve begin
        implicit_solve!(y, implicit_problem, x)
        @argcheck all(isfinite, y)
    end
    # implicit residuals
    r = fill(T(NaN), n_y)
    @_sanity_check terminate check_implicit_residuals begin
        @argcheck implicit_residuals!(r, implicit_problem, x, y) ≡ nothing
        @argcheck sum(abs2, r) ≤ √eps(T) # FIXME this is hardcoded, API?
    end
    # task local buffers
    @_sanity_check terminate check_task_local_buffers begin
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
    end
    # ∂y∂x
    @_sanity_check terminate check_∂y∂x begin
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
    end
    # statistics
    @_sanity_check terminate check_statistics begin
        @argcheck get_statistics(implicit_problem) isa NamedTuple
    end
    # collate and return
    @label done
    SanityChecks(; check_dimensions, check_eltype, check_implicit_solve,
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
