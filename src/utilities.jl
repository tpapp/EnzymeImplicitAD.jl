#####
##### utilities
#####

####
#### Primitives for calculating Jacobians and Jacobian-vector products using Enzyme.
#### Buffers need to be provided explicitly, higher-level API manages them.
####

const BUFFER_DOCS = "`buffer_…` variables are for temporary storage, may be overwritten, their initial contents can be arbitrary. It is assumed that they are not shared between tasks."

"""
$(SIGNATURES)

Calculate the Jacobian `J = ∂g/∂y`, at `x` and `y`, which is assumed to be a valid
solution (not checked).

$(BUFFER_DOCS)
"""
function _calculate_∂g∂y(implicit_problem, x::AbstractVector, y::AbstractVector{T},
                         buffer_y, buffer_r, buffer_r2) where T
    # aliases for buffers
    dy = buffer_y
    r = buffer_r
    dr = buffer_r2
    # calculate Jacobian column by column
    make_zero!(dy)
    J = similar(y, T, axes(r, 1), axes(y, 1))
    for i in axes(y, 1)
        make_zero!(dr)          # FIXME do I need this?
        dy[i] = one(T)
        autodiff(Forward, implicit_residuals!, Duplicated(r, dr),
                 Const(implicit_problem), Const(x), Duplicated(y, dy))
        J[:, i] .= dr
        dy[i] = zero(T)
    end
    J
end

"""
$(SIGNATURES)

Calculate `∂g/∂x ⋅ v` and put the result in the first argument, using forward mode in Enzyme.

$(BUFFER_DOCS)

### Expected dimensions (not checked):

- `length(v) == n_x`
- `length(Jv) == n_r`
"""
function _inplace_∂g∂x_v!(Jv, v, implicit_problem, x, y, buffer_r)
    make_zero!(Jv)              # FIXME: do I need this?
    autodiff(Forward, implicit_residuals!, Duplicated(buffer_r, Jv), Const(implicit_problem),
             Duplicated(x, v), Const(y))
    nothing
end

"""
$(SIGNATURES)

Calculate `v ⋅ ∂g/∂x` and put the result in the first argument, using reverse mode in Enzyme.

**Caution:** may modify `v`.

$(BUFFER_DOCS)

### Expected dimensions (not checked):

- `length(v) == n_r`
- `length(Jv) == n_x`
"""
function _inplace_v_∂g∂x!(vJ, v, implicit_problem, x, y, buffer_r)
    make_zero!(vJ)              # FIXME: do I need this?
    autodiff(Reverse, implicit_residuals!, Duplicated(buffer_r, v), Const(implicit_problem),
             Duplicated(x, vJ), Const(y))
    nothing
end


"""
$(SIGNATURES)

Helper function to make buffers of the right dimension. Not part of the API. Return type
is consistent with `[_make_buffers_type](@ref)`.
"""
function _make_buffers(T; n_x::Int, n_y::Int, n_r::Int)
    _v(n) = Vector{T}(undef, n)
    (; buffer_y = _v(n_y), buffer_x = _v(n_x), buffer_r = _v(n_r), buffer_r2 = _v(n_r))
end

"""
$(SIGNATURES)

The return type of [`_make_buffers`](@ref).
"""
function _make_buffers_type(T)
    V = Vector{T}
    @NamedTuple{buffer_y::V,buffer_x::V,buffer_r::V,buffer_r2::V}
end

####
#### for statistics
####

"""
Implement a thread-safe online mean. Use [`online_mean`](@ref) as the entry point.
"""
mutable struct OnlineMean{T}
    count::UInt64
    sum::T
    OnlineMean{T}() where T = new(0, zero(T))
end

"""
$(SIGNATURES)

Return a thread-safe accumulator that supportes [`update!`](@ref) and [`get_mean`](@ref).
The sum is accumulated in a value of type `T`.
"""
online_mean(::Type{T}) where T = Lockable(OnlineMean{T}())

function update!(om::Lockable{<:OnlineMean{T}}, x) where T
    Tx = T(x)
    @lock om begin
        om[].count += 1
        om[].sum += T(x)
    end
    nothing
end

get_mean(om::Lockable{<:OnlineMean}) = @lock om om[].sum / om[].count

"""
$(SIGNATURES)

Merge two `NamedTuple`s, throw an error if they have names in common.
"""
function merge_disjoint(a::NamedTuple{A}, b::NamedTuple{B}) where {A,B}
    ab = merge(a, b)
    if length(a) + length(b) ≠ length(ab)
        throw(ArgumentError("found common names $(join(intersect(names(a), names(b)), \", \"))"))
    end
    ab
end
