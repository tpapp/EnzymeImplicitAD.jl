@testset "preallocate buffers" begin
    P = E.preallocate_buffers(LinearProblem(; n_x = 4, n_y = 3))
    @test E.API_sanity_checks(P).all_ok
    @test E.get_buffers(P).x ≡ E.get_buffers(P).x
end
