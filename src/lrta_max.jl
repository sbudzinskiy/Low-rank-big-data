using LinearAlgebra: svd!, norm, Diagonal

function ttsvd!(A::AbstractArray, rs)
    d = ndims(A)
    N = size(A)
    cores = Vector{Array{Float64,3}}(undef, d)

    B = reshape(A, N[1], :)
    F = svd!(B)
    r_pre = 1
    r_cur = min(rs[1], size(B)...)
    cores[1] = reshape(view(F.U, :, 1:r_cur), r_pre, N[1], r_cur)
    r_pre = r_cur
    for i in 2:d-1
        B = Diagonal(view(F.S, 1:r_pre)) * view(F.Vt, 1:r_pre, :)
        B = reshape(B, r_pre * N[i], :)
        F = svd!(B)
        r_cur = min(rs[i], size(B)...)
        cores[i] = reshape(view(F.U, :, 1:r_cur), r_pre, N[i], r_cur)
        r_pre = r_cur
    end
    r_cur = 1
    B = Diagonal(view(F.S, 1:r_pre)) * view(F.Vt, 1:r_pre, :)
    B = reshape(B, r_pre, N[d], r_cur)
    cores[d] = B
    return cores
end

function tt2full(cores)
    d = length(cores)
    N = getindex.([size(x) for x in cores], 2)
    R = getindex.([size(x) for x in cores[2:d]], 1)
    A = cores[1]
    for i in 2:d
        A = reshape(A, :, R[i-1])
        B = reshape(cores[i], R[i-1], :)
        A = A * B
    end
    return reshape(A, N...)
end

function ttrand(rng, N, rs)
    d = length(N)
    cores = Vector{Array{Float64,3}}(undef, d)
    cores[1] = randn(rng, 1, N[1], rs[1]) ./ sqrt(rs[1])
    cores[d] = randn(rng, rs[d-1], N[d], 1) ./ sqrt(rs[d-1])
    for i in 2:d-1
        cores[i] = randn(rng, rs[i-1], N[i], rs[i]) ./ sqrt(rs[i-1] * rs[i])
    end
    return cores
end

# Quasioptimal alternating projections with "binary" search on the answer
function _lrta_qap_step(A, tt, rs, ε::Real)
    B = tt2full(tt) .- A
    clamp!(B, -ε, ε)
    C = A .+ B
    return ttsvd!(C, rs)
end

function lrta_qap(rng, A, rs, c₁::Real, c₂::Real, c₃::Real, c₄::Real)
    # Random initial approximation
    N = size(A)
    tt = ttrand(rng, N, rs)
    B = tt2full(tt) .- A
    err_pre = norm(B, Inf)
    ε₁ = 0.0
    ε₂ = err_pre

    while true # "binsearch"
        dε = ε₂ - ε₁
        (dε < c₁ * ε₂) && return (tt, ε₂)
        
        ε_cur = (ε₁ + ε₂) / 2
        while true # AP
            tt = _lrta_qap_step(A, tt, rs, ε_cur)
            C = tt2full(tt) .- A
            err_cur = norm(C, Inf)
            ε₂ = min(err_cur, ε₂)
            if (err_cur > err_pre * (1 - c₂))
                # Consecutive errors are close, convergence of QAP slowed down
                if (err_cur > ε₁ + dε * c₃)
                    # Achieved error is too big compared to the lower bound, increase the lower bound
                    ε₁ += dε * c₄
                end
                break # AP
            else
                err_pre = err_cur
            end
        end # AP
    end # "binsearch"
end
