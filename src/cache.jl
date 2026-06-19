#####
##### wrapper to cache y and ∂y∂x
#####

"A cached `y` and `∂y∂x` result, with the timestamp for the latest access."
mutable struct CacheEntry{Y,∂Y∂X}
    "the timestamp"
    timestamp::UInt64
    "the cached solution"
    y::Y
    "the cached derivative"
    ∂y∂x::Union{Nothing,∂Y∂X}
end

# NOTE: parametrization assumes `x` and `y` values have the same type `Y`
@concrete struct CacheImplicitProblem{Y,∂Y∂X}
    inner_problem
    min_size::Int
    max_size::Int
    lockable_dict
    y_hits
    ∂y∂x_hits
    function CacheImplicitProblem(inner_problem, min_size::Int, max_size::Int)
        @argcheck 0 < min_size < max_size
        T = get_preferred_eltype(inner_problem)
        Y = Vector{T}
        ∂Y∂X = get_∂y∂x_type(inner_problem)
        lockable_dict = Lockable(Dict{Y,CacheEntry{Y,∂Y∂X}}())
        y_hits = online_mean(UInt64)
        ∂y∂x_hits = online_mean(UInt64)
        new{Y,∂Y∂X,typeof(inner_problem),typeof(lockable_dict),
            typeof(y_hits),typeof(∂y∂x_hits)}(inner_problem, min_size, max_size,
                                              lockable_dict, y_hits, ∂y∂x_hits)
    end
end

function Base.show(io::IO, problem::CacheImplicitProblem)
    (; min_size, max_size, inner_problem, y_hits, ∂y∂x_hits) = problem
    print(io, "caching [$(min_size),$(max_size)] evaluations of $(inner_problem)",
          "\n    y hits: $(y_hits)",
          "\n    ∂y∂x hits: $(∂y∂x_hits)")
end

for f in [:get_dimensions, :get_preferred_eltype, :task_local_buffers, :get_∂y∂x_type]
    @eval ($f)(implicit_problem::CacheImplicitProblem) = ($f)(implicit_problem.inner_problem)
end

function get_statistics(problem::CacheImplicitProblem)
    (; inner_problem, y_hits, ∂y∂x_hits) = problem
    merge_disjoint(get_statistics(inner_problem),
                   (; average_y_hit = get_mean(y_hits),
                    average_∂y∂x_hit = get_mean(∂y∂x_hits)))
end

function implicit_residuals!(r, implicit_problem::CacheImplicitProblem, x, y)
    implicit_residuals!(r, implicit_problem.inner_problem, x, y)
end

"""
$(SIGNATURES)

Wrap an implicit problem so that `y` and `∂y∂x` are cached.

Specficially, at least `min_size` and at most `max_size` most recently used values are kept.

Supported statistics: those of the inner problem, `average_y_hit`, `average_∂y∂x_hit`.
"""
function cache_implicit_problem(inner_problem::P;
                                min_size::Int = 10, max_size = 2 * min_size) where P
    CacheImplicitProblem(inner_problem, min_size, max_size)
end

"""
$(SIGNATURES)

Cull dictionary to `min_size`, keeping the last `timestamp`s.

Locking is assumed to be handled by caller.
"""
function _cull!(dict::AbstractDict, min_size::Int)
    timestamps = [x.timestamp for x in values(dict)]
    partialsort!(timestamps, 1:min_size; rev = true)
    cutoff = timestamps[min_size]
    for (k, v) in pairs(dict)
        if v.timestamp < cutoff
            delete!(dict, k)
        end
    end
    nothing
end

_ensure_typed_copy(::Type{X}, x::X) where X = copy(x)

_ensure_typed_copy(::Type{_X}, x::X) where {_X,X} = _X(x)

function _new_cache_entry(implicit_problem::CacheImplicitProblem{Y,∂Y∂X},
                          x, timestamp::UInt64, internal_y::Y,
                          ∂y∂x::Union{Nothing,∂Y∂X} = nothing) where {Y,∂Y∂X}
    (; lockable_dict, min_size, max_size) = implicit_problem
    internal_x = _ensure_typed_copy(Y, x)
    entry = CacheEntry{Y,∂Y∂X}(timestamp, internal_y, ∂y∂x)
    lock(lockable_dict) do dict
        dict[internal_x] = entry
        length(dict) > max_size && _cull!(dict, min_size)
    end
    entry
end

function _update_timestamp(lockable_dict, entry, timestamp)
    lock(_ -> entry.timestamp = timestamp, lockable_dict)
end

function _add_∂y∂x(lockable_dict, entry, ∂y∂x)
    lock(lockable_dict) do _
        entry.∂y∂x = ∂y∂x
    end
end

function implicit_solve!(y, implicit_problem::CacheImplicitProblem{Y,∂Y∂X}, x) where {Y,∂Y∂X}
    (; inner_problem, lockable_dict, y_hits) = implicit_problem
    timestamp = time_ns()
    entry = lock(dict -> get(dict, x, nothing), lockable_dict)
    if entry ≡ nothing
        (; n_y) = get_dimensions(inner_problem)
        internal_y = Vector{get_preferred_eltype(inner_problem)}(undef, n_y)
        implicit_solve!(internal_y, inner_problem, x)
        _new_cache_entry(implicit_problem, x, timestamp, internal_y)
        copy!(y, internal_y)
        update!(y_hits, false)
    else
        copy!(y, entry.y)
        _update_timestamp(lockable_dict, entry, timestamp)
        update!(y_hits, true)
    end
    nothing
end

function calculate_∂y∂x(implicit_problem::CacheImplicitProblem{Y,∂Y∂X}, x, y) where {Y,∂Y∂X}
    (; inner_problem, lockable_dict, ∂y∂x_hits) = implicit_problem
    timestamp = time_ns()
    entry = lock(dict -> get(dict, x, nothing), lockable_dict)
    if entry ≡ nothing
        internal_y = _ensure_typed_copy(Y, y)
        ∂y∂x = calculate_∂y∂x(inner_problem, x, y)
        entry = _new_cache_entry(implicit_problem, x, timestamp, y, ∂y∂x)
        update!(∂y∂x_hits, false)
    else
        (; ∂y∂x) = entry
        _update_timestamp(lockable_dict, entry, timestamp)
        if ∂y∂x ≡ nothing # add ∂y∂x
            ∂y∂x = calculate_∂y∂x(inner_problem, x, y)
            _add_∂y∂x(lockable_dict, entry, ∂y∂x)
            update!(∂y∂x_hits, false)
        else
            update!(∂y∂x_hits, true)
        end
    end
    ∂y∂x
end
