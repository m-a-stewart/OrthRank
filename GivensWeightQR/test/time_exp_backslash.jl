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
  max_rank
  )
  block_gap = 1
  upper_rank_max = max_rank
  lower_rank_max = max_rank
  rng = MersenneTwister(1234)
  sizes = zeros(Int64,length(7:12))
  for k = 7:12
    sizes[k-6] = 2^k
  end
  t = zeros(length(sizes),1)
  for i in axes(sizes,1)
    m = sizes[i]
    gw = create_gw(rng, m, m, block_gap, upper_rank_max, lower_rank_max)
    F = qr(gw)
    b = randn(m)
    t[i,1] = @elapsed F\b
  end
  return sizes, t
end

function experiment_rank(
  ms
  ) 
  block_gap = 1
  max_rank = 200
  ranks = 5:10:max_rank
  rng = MersenneTwister(1234)
  m = ms
  t = zeros(length(ranks),1)
  for i in axes(ranks,1)
    urm = ranks[i]
    lrm = urm
    gw = create_gw(rng, m, m, block_gap, urm, lrm)
    F = qr(gw)
    b = randn(m)
    t[i,1] = @elapsed F\b
  end
  return ranks, t
end

function experiment_gap(
  ms
  )
  max_gap = 10
  gaps = 1:2:max_gap
  max_rank = 1
  urm = max_rank
  lrm = max_rank
  m = ms
  rng = MersenneTwister(1234)
  t = zeros(length(gaps),1)
  for i in axes(gaps,1)
    gap = gaps[i]
    gw = create_gw(rng, m, m, gap, urm, lrm)
    F = qr(gw)
    b = randn(m)
    t[i,1] = @elapsed F\b
  end
  return gaps, t 
end

sizes1, ts1 = experiment_size(1)
sizes2, ts2 = experiment_size(5)
sizes3, ts3 = experiment_size(10) 
plot([sizes3, sizes2, sizes1], [ts3, ts2, ts1], label=["rank 10" "rank 5" "rank 1"], markershape=[:utriangle :dtriangle :star5],  xaxis=:log2, yaxis=:log10, xformatter=:plain, yformatter=:plain, gridalpha=:0.35, xwiden=1, xlims=(0.5*minimum(sizes1),1.41*maximum(sizes1)), dpi=300, html_output_format=:svg, legend=:topleft)
savefig("GivensWeightQR/images/backslash_sizes.svg")

ranks1, tr1 = experiment_rank(100000)
ranks2, tr2 = experiment_rank(500000)
ranks3, tr3 = experiment_rank(1000000)
plot([ranks3, ranks2, ranks1], [tr3, tr2, tr1], label=["Size 10000" "Size 5000" "Size 1000"], markershape=[:utriangle :dtriangle :star5], xformatter=:plain, yformatter=:plain, gridalpha=:0.35, dpi=300, html_output_format=:svg, legend=:topleft)
savefig("GivensWeightQR/images/backslash_ranks.svg")


# GAP GIvES TROUBLES OF NONSIGULARITY 
# gaps1, tg1 = experiment_gap(1000)
# gaps2, tg2 = experiment_gap(2000)
# gaps3, tg3 = experiment_gap(4000)
# plot([gaps3, gaps2, gaps1], [tg3, tg2, tg1], label=["Size 10000" "Size 5000" "Size 1000"], markershape=[:utriangle :dtriangle :star5], xformatter=:plain, yformatter=:plain, gridalpha=:0.35, dpi=300, html_output_format=:svg, legend=:topleft)
# savefig("GivensWeightQR/images/backslash_gaps.svg")