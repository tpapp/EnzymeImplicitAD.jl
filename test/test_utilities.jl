@testset "∂g∂y, ∂g∂x_v, v_∂g∂x extraction" begin
    n_x, n_y = 3, 4
    (; A, B) = P = LinearProblem(; n_y, n_x)
    x = randn(n_x)
    y = randn(n_y)
    dy = similar(y)
    buffers = E.get_buffers(P)

    J = E._calculate_∂g∂y(P, x, y, buffers)
    @test J ≈ B

    Jv = similar(y)
    v = randn(n_x)
    E._inplace_∂g∂x_v!(Jv, v, P, x, y, buffers.r)
    @test Jv ≈ A * v

    vJ = similar(x)
    v = randn(n_y)
    E._inplace_v_∂g∂x!(vJ, copy(v), P, x, y, buffers.r)
    @test vJ ≈ A' * v
end
