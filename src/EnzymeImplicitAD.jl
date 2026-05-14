"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

public get_dimensions, implicit_solve!, implicit_residuals!, task_local_buffers, calculate_∂y∂x,
    calculate_pushforward!, accumulate_pullback!

using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using ConcreteStructs: @concrete

include("utilities.jl")
include("api.jl")
include("enzyme_ad.jl")
include("solvers.jl")

end # module
