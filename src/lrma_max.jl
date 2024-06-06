using LinearAlgebra: svd, Diagonal, norm

# SVD truncation based on the rank criterion
function svd_maxerr(A, rs)
    m,n = size(A)

    F = svd(A)
    norms = Vector{Float64}(undef,length(rs))
    for (i,r) in enumerate(rs)
        rr = min(r,m,n)
        B = view(F.U, :, 1:rr) * Diagonal(F.S[1:rr]) * view(F.Vt, 1:rr, :)
        norms[i] = norm(A .- B, Inf)
    end
    return norms
end

function svd_maxerr(A, r::Int)
    return svd_maxerr(A, [r])[1]
end

# Alternating projections with "binary" search on the answer
function _lrma_ap_step(A, U, Vt, r::Int, ε::Real)
    B = U * Vt .- A
    clamp!(B, -ε, ε)
    C = A .+ B
    FF = svd(C)
    return FF.U[:,1:r] * Diagonal(FF.S[1:r]), FF.Vt[1:r, :]
end

function lrma_ap(rng, A, r::Int, c₁::Real, c₂::Real, c₃::Real, c₄::Real)
    # Random initial approximation
    m, n = size(A)
    rr = min(r,m,n)

    U = randn(rng, m, rr) ./ sqrt(rr)
    Vt = randn(rng, rr, n) ./ sqrt(rr)
    B = U * Vt .- A
    err_pre = norm(B, Inf)
    ε₁ = 0.0
    ε₂ = err_pre

    while true # "binsearch"
        dε = ε₂ - ε₁
        (dε < c₁ * ε₂) && return (U, Vt, ε₂)
        
        ε_cur = (ε₁ + ε₂) / 2
        while true # AP
            U, Vt = _lrma_ap_step(A, U, Vt, rr, ε_cur)
            C = U * Vt .- A
            err_cur = norm(C, Inf)
            ε₂ = min(err_cur, ε₂)
            if (err_cur > err_pre * (1 - c₂))
                # Consecutive errors are close, convergence of AP slowed down
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
