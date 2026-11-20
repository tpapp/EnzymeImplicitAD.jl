include("setup.jl")

@testset "static analysis with JET.jl" begin
    import JET
    JET.test_package(EnzymeImplicitAD, target_modules=(EnzymeImplicitAD,))
end

include("test_utilities.jl")
include("test_api.jl")
include("test_cache_results.jl")
include("test_benchmarks.jl")
include("test_keep_failures.jl")
include("test_preallocate_buffers.jl")

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(EnzymeImplicitAD)
end
