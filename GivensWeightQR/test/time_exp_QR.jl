using LinearAlgebra 
using Random 
using InPlace 
using Rotations 
using Householder
using BandStruct
using OrthWeight
using GivensWeightQR
using Plots
using BenchmarkTools
using Polynomials
 
function create_gw(
  rng::AbstractRNG,
  m::Int64,
  n::Int64,
  block_gap::Int64,
  upper_rank_max::Int64,
  lower_rank_max::Int64,
 )
  upper_blocks, lower_blocks =
    random_blocks_generator_no_overlap(rng, m, n, block_gap)
  num_blocks = length(upper_blocks)
  upper_ranks = Consts(length(upper_blocks), upper_rank_max)
  lower_ranks = Consts(length(upper_blocks), lower_rank_max)
  upper_ranks =
    constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)
  lower_ranks =
    constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)
  max_num_upper_rots =
    div(block_gap + upper_rank_max, 2, RoundUp)^2 + (
      (lower_rank_max) * (upper_rank_max + lower_rank_max - 1) +
      (block_gap - 1) * div(lower_rank_max * (lower_rank_max + 1), 2)
    )
  max_num_lower_rots = (block_gap + lower_rank_max - 1) * lower_rank_max
  upper_rank_max = 2 * block_gap + upper_rank_max + lower_rank_max
  gw = GivensWeight(
    Float64,
    TrailingDecomp(),
    rng,
    m,
    n;
    upper_rank_max = upper_rank_max,
    lower_rank_max = lower_rank_max,
    upper_ranks = upper_ranks,
    lower_ranks = lower_ranks,
    upper_blocks = upper_blocks,
    max_num_upper_blocks = num_blocks,
    lower_blocks = lower_blocks,
    max_num_lower_blocks = num_blocks,
    max_num_upper_rots = max_num_upper_rots,
    max_num_lower_rots = max_num_lower_rots,
  )
  return gw
end

function experiment_size(
  max_rank::Int64
  )
  block_gap = 1
  upper_rank_max = max_rank
  lower_rank_max = max_rank
  rng = MersenneTwister(1234)
  sizes = zeros(Int64,length(7:10))
  for k = 7:10 #7:20 (try bigger than 10^6 (million))
    sizes[k-6] = 2^k
  end
  t = zeros(length(sizes),1)
  for i in axes(sizes,1)
    m = sizes[i]
    gw = create_gw(rng, m, m, block_gap, upper_rank_max, lower_rank_max)
    t[i,1] = @elapsed qr(gw)
  end
  return sizes, t
end

function experiment_rank(
  ms::Int64
  ) 
  block_gap = 1
  max_rank = 80
  ranks = 1:5:max_rank
  rng = MersenneTwister(1234)
  m = ms
  t = zeros(length(ranks),1)
  for i in axes(ranks,1)
    urm = ranks[i]
    lrm = urm
    gw = create_gw(rng, m, m, block_gap, urm, lrm)
    t[i,1] = @elapsed qr(gw)
  end
  return ranks, t
end

function experiment_gap(
  ms::Int64
  )
  max_gap = 500
  gaps = 100:20:max_gap
  max_rank = 4
  urm = max_rank
  lrm = max_rank
  m = ms
  rng = MersenneTwister(1234)
  t = zeros(length(gaps),1)
  for i in axes(gaps,1)
    gap = gaps[i]
    gw = create_gw(rng, m, m, gap, urm, lrm)
    t[i,1] = @elapsed qr(gw)
  end
  return gaps, t 
end

# sizes1, ts1 = experiment_size(1)
# sizes2, ts2 = experiment_size(5)
# sizes3, ts3 = experiment_size(10) 
# plot([sizes3, sizes2, sizes1], [ts3, ts2, ts1], label=["rank 10" "rank 5" "rank 1"], markershape=[:utriangle :dtriangle :star5], xaxis=:log2, yaxis=:log10, xformatter=:plain, yformatter=:plain, gridalpha=:0.35, xwiden=1, xlims=(0.5*minimum(sizes1),1.41*maximum(sizes1)), dpi=300, html_output_format=:svg, legend=:topleft)
# savefig("GivensWeightQR/images/QR_Decomposition_sizes.svg")

# ranks1, tr1 = experiment_rank(100)#100000)
# ranks2, tr2 = experiment_rank(1500)#500000)
# ranks3, tr3 = experiment_rank(5000)#1000000)
# #Quadratic approximation
# qr1 = fit(ranks1, tr1[:,1],2)
# qr2 = fit(ranks2, tr2[:,1],2)
# qr3 = fit(ranks3, tr3[:,1],2)
# plot([ranks3, ranks2, ranks1], [tr3, tr2, tr1], label=["Size 10000" "Size 5000" "Size 1000"], markershape=[:utriangle :dtriangle :star5], xformatter=:plain, yformatter=:plain, gridalpha=:0.35, dpi=300, html_output_format=:svg, legend=:topleft)
# plot!(qr3, extrema(ranks3)...,ls=:dashdot, labels="", linecolor=:blue)
# plot!(qr2, extrema(ranks2)...,ls=:dashdot, labels="", linecolor=:orange)
# plot!(qr1, extrema(ranks1)...,ls=:dashdot, labels="", linecolor=:lightgreen)
# # savefig("GivensWeightQR/images/QR_Decomposition_ranks.svg")

gaps1, tg1 = experiment_gap(1000)
gaps2, tg2 = experiment_gap(2000)
gaps3, tg3 = experiment_gap(4000)
# #Quadratic approximation
qg1 = fit(gaps1, tg1[:,1],2)
qg2 = fit(gaps2, tg2[:,1],2)
qg3 = fit(gaps3, tg3[:,1],2)
plot([gaps3, gaps2, gaps1], [tg3, tg2, tg1], label=["Size 10000" "Size 5000" "Size 1000"], markershape=[:utriangle :dtriangle :star5], xformatter=:plain, yformatter=:plain, gridalpha=:0.35, dpi=300, html_output_format=:svg, legend=:topleft)
plot!(qg3, extrema(gaps3)...,ls=:dashdot, labels="", linecolor=:blue)
plot!(qg2, extrema(gaps2)...,ls=:dashdot, labels="", linecolor=:orange)
plot!(qg1, extrema(gaps1)...,ls=:dashdot, labels="", linecolor=:lightgreen)
# savefig("GivensWeightQR/images/QR_Decomposition_gaps.svg")