include("setup.jl")

include("test_utilities.jl")
include("test_api.jl")
include("test_solvers.jl")
include("test_cache.jl")
include("test_benchmarks.jl")

####
#### QA
####

@testset "static analysis with JET.jl" begin
    import JET
    JET.test_package(EnzymeImplicitAD, target_modules=(EnzymeImplicitAD,))
end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(EnzymeImplicitAD)
end
