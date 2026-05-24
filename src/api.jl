#####
##### the generic API
#####

"""
$(FUNCTIONNAME)(implicit_problem) → (; n_x, n_y, n_r)

Return the dimensions of the problem.
"""
function get_dimensions end

"""
$(FUNCTIONNAME)(implicit_problem) → T

Return the preferred element type for a problem. This is used for buffers and interim
quantities, and should allow for enough precision even with input/output arrays that
have less.
"""
function get_preferred_eltype end

"""
$(SIGNATURES)

Return `true` iff the problem is square (`x`, `y`, `r` have the same dimensions).
"""
function is_square(implicit_problem)
    (; n_x, n_y, n_r) = get_dimensions(implicit_problem)
    n_x == n_y == n_r
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

Return an object which containes the following buffers, accessible as properties:

- `buffer_y1`, `buffer_y2`, `buffer_y3`; the same length as `y`

The fallback method reallocates these for each use, implementations can provide shared
buffers but they are guaranteed to be task-local.

The element type of buffers should be consistent with [`get_preferred_eltype`](@ref).

$(BUFFER_DOCS)
"""
function task_local_buffers(implicit_problem)
    (; n_y) = get_dimensions(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    _v() = Vector{T}(undef, n_y)
    (; buffer_y1 = _v(), buffer_y2 = _v(), buffer_y3 = _v())
end

@concrete terse struct ∂Y∂X
    ∂g∂y_factor
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
    (; buffer_y1, buffer_y2, buffer_y3) = task_local_buffers(implicit_problem)
    ∂g∂y = _calculate_∂g∂y(implicit_problem, x, y, buffer_y1, buffer_y2, buffer_y3)
    ∂Y∂X(lu!(∂g∂y))
end

"""
$(SIGNATURES)

The return type of [`calculate_∂y∂x`](@ref). Used-defined methods should ensure consistency.
"""
function get_∂y∂x_type(implicit_problem)
    (; n_y) = get_dimensions(implicit_problem)
    T = get_preferred_eltype(implicit_problem)
    lu!(typeof(lu!(ones(T, n_y, n_y))))
end

"""
$(SIGNATURES)

Calculate the pushforward `dy = ∂y∂x ⋅ dx` into `dy`.

A fallback is provided using Enzyme, but an `implicit_problem` can define its own method.
"""
function calculate_pushforward!(dy, implicit_problem, x, y, ∂y∂x::∂Y∂X, dx)
    (; buffer_y1) = task_local_buffers(implicit_problem)
    _inplace_∂g∂x_v!(dy, dx, implicit_problem, x, y, buffer_y1)
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
    (; ∂g∂y_factor) = ∂y∂x
    # math:
    #     dy ⋅ ∂y/∂x = - (dy' / ∂g/∂y) ⋅ ∂g/∂x
    (; buffer_y1, buffer_y2, buffer_y3) = task_local_buffers(implicit_problem)
    buffer_y1 .= dy                # buffer_y1 == dy
    rdiv!(buffer_y1', ∂g∂y_factor) # a == dy' / ∂g∂y
    _inplace_v_∂g∂x!(buffer_y2, buffer_y1, implicit_problem, x, y,
                     buffer_y3) # b = (dy' / ∂g/∂y) ⋅ ∂g/∂x
    dx .-= buffer_y2
    nothing
end

####
#### sanity checks
####

"""
Implementation for `API_sanity_checks`, not part of the API.
"""
Base.@kwdef struct SanityChecks
    dimensions_error
    eltype_error
    implicit_solve_error
    implicit_residuals_error
    task_local_buffers_error
end

"""
$(SIGNATURES)

Helper macro for implementing [`API_sanity_checks`](@ref). Initializes `errorvar = missing`,
then sets it to `nothing` after successful evaluation of `body`.

If there is an error, it is caught and stored in `errorvar`, followed by `@goto done`.
"""
macro _sanity_check(errorvar, body)
    err = esc(errorvar)
    quote
        $(err) = missing
        try
            $(esc(body))
            $(err) = nothing
        catch e
            $(err) = e
            $(esc(Expr(:symbolicgoto, :done)))
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
    @_sanity_check dimensions_error begin
        dimensions = get_dimensions(implicit_problem)
        (; n_x, n_y, n_r) = dimensions
        @assert n_x isa Int && n_x > 0
        @assert n_y isa Int && n_y > 0
        @assert n_r isa Int && n_r > 0
    end
    # eltype
    local T
    @_sanity_check eltype_error begin
        T = get_preferred_eltype(implicit_problem)
        @argcheck T <: AbstractFloat
    end
    # implicit solve
    x = randn(T, n_x)
    y = fill(T(NaN), n_y)
    @_sanity_check implicit_solve_error begin
        implicit_solve!(y, implicit_problem, x)
        @argcheck all(isfinite, y)
    end
    # implicit residuals
    r = fill(T(NaN), n_y)
    @_sanity_check implicit_residuals_error begin
        implicit_residuals!(r, implicit_problem, x, y)
        @argcheck norm(r, 2) ≤ √eps(T) # FIXME this is hardcoded, API?
    end
    # task local buffers
    @_sanity_check task_local_buffers_error begin
        buffers = task_local_buffers(implicit_problem)
        function _check_y_buffer(b)
            b[1] += one(T)      # check mutability
            @argcheck b isa AbstractVector
            @argcheck eltype(b) ≡ T
            @argcheck length(b) == n_y
        end
        _check_y_buffer(buffers.buffer_y1)
        _check_y_buffer(buffers.buffer_y2)
        _check_y_buffer(buffers.buffer_y2)
    end
    # collate and return
    @label done
    SanityChecks(; dimensions_error, eltype_error, implicit_solve_error,
                 implicit_residuals_error, task_local_buffers_error)
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
        printstyled(io, "✔ all checks passed";
                    bold = true, color = :green)
    else
        printstyled(io, "✘"; color = :red)
    end
end
