using LinearAlgebra: normalize!, rmul!, dot, Diagonal
using Random: randn!

abstract type SamplingSet end
struct Sphere <: SamplingSet end
struct Ball <: SamplingSet end

abstract type SamplingRule end
struct RandomInd <: SamplingRule end
struct RandomSym <: SamplingRule end

function rand_sphere!(rng, A::Matrix{<:Real}, R::Real)
    randn!(rng, A)
    n = size(A,2)
    for i in 1:n
        normalize!(view(A,:,i), 2)
    end
    A .= A .* R
    return A
end

function rand_sphere(rng, m::Int, R::Real, n::Int=1)
    A = Matrix{Float64}(undef,m,n)
    return rand_sphere!(rng, A, R)
end

function rand_ball!(rng, A::Matrix{<:Real}, R::Real)
    rand_sphere!(rng, A, R)
    m, n = size(A)
    for i in 1:n
        u = rand(rng)
        rmul!(view(A,:,i), u^(1/m))
    end
    return A
end

function rand_ball(rng, m::Int, R::Real, n::Int=1)
    A = Matrix{Float64}(undef,m,n)
    return rand_ball!(rng, A, R)
end

_sample_points(rng, ::Type{Sphere}, m, R, n) = rand_sphere(rng, m,R,n)
_sample_points(rng, ::Type{Ball},   m, R, n) = rand_ball(rng, m,R,n)
function _sample_points_twice(rng, ::Type{RandomInd}, S::Type{<:SamplingSet}, m, R, n₁, n₂)
    XT = _sample_points(rng, S, m, R, n₁)
    YT = _sample_points(rng, S, m, R, n₂)
    return XT, YT
end
function _sample_points_twice(rng, ::Type{RandomSym}, S::Type{<:SamplingSet}, m, R, n₁, n₂)
    (n₁ != n₂) && throw(DimensionMismatch("The matrix needs to be square for symmetric sampling."))
    XT = _sample_points(rng, S, m, R, n₁)
    YT = XT
    return XT, YT
end
function _sample_points_times(rng, ::Type{RandomInd}, S::Type{<:SamplingSet}, m, R, N)
    d = length(N)
    samples = Vector{Matrix{Float64}}(undef, d)
    for i in 1:d
        samples[i] = _sample_points(rng, S, m, R, N[i])
    end
    return samples
end
function _sample_points_times(rng, ::Type{RandomSym}, S::Type{<:SamplingSet}, m, R, N)
    !allequal(N) && throw(DimensionMismatch("The tensor needs to have equal sides for symmetric sampling."))
    d = length(N)
    samples = Vector{Matrix{Float64}}(undef, d)
    samples[1] = _sample_points(rng, S, m, R, N[1])
    for i in 2:d
        samples[i] = samples[1]
    end
    return samples
end

function fgm_dotp(rng, SR::Type{<:SamplingRule}, SS::Type{<:SamplingSet}, f, m::Int, R::Real, n₁::Int, n₂::Int)
    XT, YT = _sample_points_twice(rng, SR, SS, m, R, n₁, n₂)
    return f.(XT' * YT)
end

function fgm_dist2(rng, SR::Type{<:SamplingRule}, SS::Type{<:SamplingSet}, f, m::Int, R::Real, n₁::Int, n₂::Int)
    XT, YT = _sample_points_twice(rng, SR, SS, m, R, n₁, n₂)
    A = Matrix{Float64}(undef, n₁, n₂)
    for i in 1:n₁, j in 1:n₂
        u = XT[:,i] - YT[:,j]
        A[i,j] = dot(u,u)
    end
    return f.(A)
end

function fgt_dotp(rng, SR::Type{<:SamplingRule}, SS::Type{<:SamplingSet}, f, m::Int, R::Real, n₁::Int, n₂::Int, n₃::Int)
    samples = _sample_points_times(rng, SR, SS, m, R, (n₁,n₂,n₃))
    A = zeros(Float64, n₁, n₂, n₃)
    for i in 1:n₁, j in 1:n₂, k in 1:n₃
        @views A[i,j,k] = dot(samples[1][:,i], Diagonal(samples[2][:,j]), samples[3][:,k])
    end
    return f.(A)
end
