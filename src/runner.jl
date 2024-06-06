using LinearAlgebra: norm

function parse_cl_args(ARGS::Vector{String})
# Parse input command-line arguments
# - filename -- output file name
# - d --- number of modes
# - any delimeter
# - n₁, ..., nₖ --- array of matrix sizes
# - any delimeter
# - m₁, ..., mₚ --- array of latent dimensions
# - any delimeter
# - r₁, ..., rₛ --- array of approximation ranks
# - any delimeter
# - ntrials --- number of runs with same parameters
# - sym --- sample points symmetrically or not
# - ap --- use alternating projections or SVD
    Ns = Vector{Int}(undef,0)
    Ms = Vector{Int}(undef,0)
    Rs = Vector{Int}(undef,0)

    iarg = 1
    filename = ARGS[iarg]
    iarg += 1

    d = tryparse(Int, ARGS[iarg])
    iarg += 2

    while iarg <= length(ARGS)
        n = tryparse(Int, ARGS[iarg])
        iarg += 1
        n == nothing && break
        append!(Ns, n)
    end
    while iarg <= length(ARGS)
        m = tryparse(Int, ARGS[iarg])
        iarg += 1
        m == nothing && break
        append!(Ms, m)
    end
    while iarg <= length(ARGS)
        r = tryparse(Int, ARGS[iarg])
        iarg += 1
        r == nothing && break
        append!(Rs, r)
    end

    ntrials = tryparse(Int, ARGS[iarg])
    iarg += 1

    sym = tryparse(Bool, ARGS[iarg])
    iarg += 1

    ap = tryparse(Bool, ARGS[iarg])
    iarg += 1

    return filename, d, Ns, Ms, Rs, ntrials, sym, ap
end

function collect_lrma_max_stats(rng, h, radius, Ns, Ms, Rs, ntrials, sample_sym, use_ap, ap_params)
    appr_errs = zeros(length(Ns), length(Ms), length(Rs), ntrials, 2)
    for (ins,n) in enumerate(Ns), (ims,m) in enumerate(Ms), t in 1:ntrials
        @show n, m, t
        flush(stdout)
        if sample_sym
            F = fgm_dist2(rng, RandomSym, Ball, h, m, radius, n, n)
        else
            F = fgm_dist2(rng, RandomInd, Ball, h, m, radius, n, n)
        end
        nrmF = norm(F,Inf)

        if use_ap
            for (irs,r) in enumerate(Rs)
                _, _, ε = lrma_ap(rng, F, r, ap_params...)
                appr_errs[ins, ims, irs, t, :] .= nrmF, ε
            end
        else
            εs = svd_maxerr(F, Rs)
            for irs in 1:length(Rs)
                appr_errs[ins, ims, irs, t, :] .= nrmF, εs[irs]
            end
        end
    end
    return appr_errs
end

function collect_lrta_max_stats(rng, h, radius, d, Ns, Ms, Rs, ntrials, ap_params)
    appr_errs = zeros(length(Ns), length(Ms), length(Rs), ntrials, 2)
    for (ins,n) in enumerate(Ns), (ims,m) in enumerate(Ms), t in 1:ntrials
        @show n, m, t
        F = fgt_dotp(rng, RandomInd, Ball, h, m, radius, [n for i in 1:d]...)  
        nrmF = norm(F,Inf)

        for (irs,r) in enumerate(Rs)
            @show r
            flush(stdout)
            _, ε = lrta_qap(rng, F, [r for i in 1:d-1], ap_params...)
            appr_errs[ins, ims, irs, t, :] .= nrmF, ε
        end
    end
    return appr_errs
end
