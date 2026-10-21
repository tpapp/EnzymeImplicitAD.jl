"""
Exploring implicit differentiation in Enzyme.
"""
module EnzymeImplicitAD

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, FIELDS, TYPEDEF

include("utilities.jl")
include("api.jl")
include("sanity_checks.jl")
include("benchmarks.jl")
include("enzyme_ad.jl")
include("cache.jl")
include("keep_failures.jl")

end # module
