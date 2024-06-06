using Pkg
Pkg.activate(".")

using Random
using NPZ

include("src/generation.jl")
include("src/lrta_max.jl")
include("src/runner.jl")

rng = MersenneTwister(1)
radius = 1.0
ap_params = (1e-2, 1e-3, 2/3, 2e-2)

filename, d, Ns, Ms, Rs, ntrials, sample_sym, use_ap = parse_cl_args(ARGS)

h(x) = sinh(x)
appr_errs = collect_lrta_max_stats(rng, h, radius, d, Ns, Ms, Rs, ntrials, ap_params)

npzwrite("$filename.npz", Dict("errs" => appr_errs, "N" => Ns, "M" => Ms, "R" => Rs, "sym" => sample_sym, "ap" => use_ap))
