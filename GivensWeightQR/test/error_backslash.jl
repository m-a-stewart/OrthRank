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
using StatsPlots
 
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

rng = MersenneTwister(1234)
num_exp = 50
dims = [1000, 2000, 3000]
block_gap = 10
upper_rank_max = 5
lower_rank_max = 5
residuals = zeros(num_exp)
plot()
for m in dims
  n = m
  for i in 1:num_exp
    b = rand(n,1)
    c = copy(b)
    gw = create_gw(rng, m, n, block_gap, upper_rank_max, lower_rank_max)
    gwc = deepcopy(gw)
    x = gw\b
    # F = qr(gwc)
    # Q = Matrix(1.0I, m, m)
    # create_Q!(Q, F)
    # R = create_R(F)
    residuals[i] = norm(gwc*x - c,2)/(norm_2_gw(gwc,100)*norm(x,2)) #when vector, norm(_,2) is norm 2 of vectors
  end
  boxplot!(rand([string(m)],num_exp),log10.(residuals))
end
plot!(legend=false, ylabel="relative residual 2-norm = 10ᵏ", xlabel="matrix size")
savefig("GivensWeightQR/images/residual_backslash.svg")
