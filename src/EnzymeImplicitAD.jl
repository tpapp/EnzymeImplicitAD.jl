"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES
using ConcreteStructs: @concrete

include("utilities.jl")
include("api.jl")
include("enzyme_ad.jl")
include("solvers.jl")
include("cache.jl")

end # module
