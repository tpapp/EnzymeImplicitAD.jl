#####
##### the generic API
#####

"""
$(FUNCTIONNAME)(implicit_problem) → (; n_x, n_y, n_r)

Return the dimensions of the problem.
"""
function get_dimensions end

"""
$(SIGNATURES) → T

Return the preferred element type for a problem. This is used for buffers and interim
quantities, and should allow for enough precision even with input/output arrays that
have less.

The default is `Float64`.
"""
get_preferred_eltype(implicit_problem) = Float64

"""
$(SIGNATURES)

Return `true` iff the problem is square (`y`, `r` have the same dimensions).
"""
function is_square(implicit_problem)
    (; n_y, n_r) = get_dimensions(implicit_problem)
    n_y == n_r
end

"""
$(FUNCTIONNAME)(y, implicit_problem, x) → nothing

Solve the implicit problem ``g(x, y(x)) = 0`` at `x`, overwriting `y` with ``y(x)`` result.

Return `nothing`. See [`implicit_residuals!`](@ref), which implements ``g`` above.
```
"""
function implicit_solve! end

"""
$(FUNCTIONNAME)(r, implicit_problem, x, y) → nothing

Calculate the implicit residuals ``r = g(x, y)``, overwriting `r`.

Return `nothing`.

It is assumed that after
```julia
implicit_solve!(y, implicit_problem, x)
$(FUNCTIONNAME)(r, implicit_problem, x, y)
```
the residuals `r` are “approximately” zero, but this is not checked.
"""
function implicit_residuals! end

"""
$(SIGNATURES) → (; buffer_y1, buffer_y2, buffer_y3)

Return an object which contains the following buffers, which are accessible as
properties. Each is a vector, with lengths consistent with the corresponding dimension
in [`get_dimensions`](@ref).

- `buffer_r`, `buffer_r2`: has length `n_r`
- `buffer_x`: has length `n_x`
- `buffer_y`: has length `n_y`

The fallback method reallocates these for each use, implementations can provide shared
buffers but they are guaranteed to be task-local.

The element type of buffers should be consistent with [`get_preferred_eltype`](@ref).

$(BUFFER_DOCS)
"""
function task_local_buffers(implicit_problem)
    _make_buffers(get_preferred_eltype(implicit_problem); get_dimensions(implicit_problem)...)
end

@concrete struct ∂Y∂X
    ∂g∂y_factor
end

function Base.show(io::IO, ∂y∂x::∂Y∂X)
    n_x, n_y = size(∂y∂x.∂g∂y_factor)
    print(io, "∂Y∂X(« $(n_x) × $(n_y) »)")
end

"""
$(SIGNATURES)

The return type of [`calculate_∂y∂x`](@ref).

Should be a concrete type that depends only on `implicit_problem`, not `x` or `y`.

Used-defined methods should ensure consistency.
"""
function get_∂y∂x_type(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    L = typeof(lu!(ones(T, 1, 1))) # assumption: lu! is type stable, size does not matter
    ∂Y∂X{L}
end

"""
$(SIGNATURES)

Return an object `∂y∂x` that acts like a Jacobian matrix when pre- or post-multiplied by
a conformable vector, via the methods [`calculate_pushforward!`](@ref) and
[`accumulate_pullback!`](@ref).

The return type should depend only on `implicit_problem`, and should be consistent with
[`get_∂y∂x_type`]((@ref).

The implementation is free to ignore `y`, eg if it can obtain a solution from `x`.
"""
function calculate_∂y∂x(implicit_problem, x, y)
    (; buffer_y, buffer_r, buffer_r2) = task_local_buffers(implicit_problem)
    ∂g∂y = _calculate_∂g∂y(implicit_problem, x, y, buffer_y, buffer_r, buffer_r2)
    ∂Y∂X(lu!(∂g∂y))
end

"""
$(SIGNATURES)

Calculate the pushforward `dy = ∂y∂x ⋅ dx` into `dy`.

A fallback is provided using Enzyme, but an `implicit_problem` can define its own method.
"""
function calculate_pushforward!(dy, implicit_problem, x, y, ∂y∂x::∂Y∂X, dx)
    @assert is_square(implicit_problem)
    (; buffer_r) = task_local_buffers(implicit_problem)
    _inplace_∂g∂x_v!(dy, dx, implicit_problem, x, y, buffer_r)
    ldiv!(∂y∂x.∂g∂y_factor, dy)
    dy .*= -1
    nothing
end

"""
$(SIGNATURES)

Accumulate the pullback `dy ⋅ ∂y∂x` into `dx`.

A default is implemented using Enzyme, but an `implicit_problem` can define its own method.
"""
function accumulate_pullback!(dx, implicit_problem, x, y, ∂y∂x::∂Y∂X, dy)
    @assert is_square(implicit_problem)
    (; ∂g∂y_factor) = ∂y∂x
    # math:
    #     dy ⋅ ∂y/∂x = - (dy' / ∂g/∂y) ⋅ ∂g/∂x
    (; buffer_x, buffer_y, buffer_r) = task_local_buffers(implicit_problem)
    buffer_y .= dy                # buffer_y1 == dy
    rdiv!(buffer_y', ∂g∂y_factor) # a == dy' / ∂g∂y
    _inplace_v_∂g∂x!(buffer_x, buffer_y, implicit_problem, x, y,
                     buffer_r)  # b = (dy' / ∂g/∂y) ⋅ ∂g/∂x
    dx .-= buffer_x
    nothing
end

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
    check_implicit_solve
    check_implicit_residuals
    check_task_local_buffers
    check_∂y∂x
end

"""
$(SIGNATURES)

Helper macro for implementing [`API_sanity_checks`](@ref).

Initializes `errorvar = missing`, then sets it to `nothing` after successful evaluation
of `body`, which is only run when `!terminate`.

If there is an error, it is caught and stored in `errorvar`, and `terminate = true` is set.
"""
macro _sanity_check(terminate, errorvar, body)
    err = esc(errorvar)
    quote
        $(err) = missing
        if !$(esc(terminate))
            try
                $(esc(body))
                $(err) = nothing
            catch e
                $(err) = (e, catch_backtrace())
                $(esc(terminate)) = true
            end
        end
    end
end

"""
$(SIGNATURES) → checks

Check that the interface implemented to `implicit_problem` conforms to the expected API.

Checks are not necessarily comprehensive, and may change without major version changes.
The user can access the property `checks.all_ok::Bool`, the rest of the fields can be
used for debugging but are not part of the API.
"""
function API_sanity_checks(implicit_problem)
    # dimensions
    local n_x
    local n_y
    local n_r
    terminate = false
    @_sanity_check terminate check_dimensions begin
        dimensions = get_dimensions(implicit_problem)
        (; n_x, n_y, n_r) = dimensions
        @assert n_x isa Int && n_x > 0
        @assert n_y isa Int && n_y > 0
        @assert n_r isa Int && n_r > 0
    end
    # eltype
    local T
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
        implicit_residuals!(r, implicit_problem, x, y)
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
    # collate and return
    @label done
    SanityChecks(; check_dimensions, check_eltype, check_implicit_solve,
                 check_implicit_residuals, check_task_local_buffers, check_∂y∂x)
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
