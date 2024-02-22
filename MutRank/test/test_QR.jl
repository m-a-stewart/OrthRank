using Random
using OrthWeight
using BandStruct
using Householder
using InPlace
using Rotations
using LinearAlgebra
using MutRank
using Test

function test_QR(
    m::Int64,
    num_blocks::Int64,
    upper_rank_max::Int64,
    lower_rank_max::Int64
    )
    tol = 1e-14  
    n = m
    upper_blocks, lower_blocks = random_blocks(m,num_blocks)
    upper_ranks = Consts(num_blocks, upper_rank_max)
    lower_ranks = Consts(num_blocks, lower_rank_max)
    upper_ranks = constrain_upper_ranks(m, n, blocks = upper_blocks, ranks = upper_ranks)
    lower_ranks = constrain_lower_ranks(m, n, blocks = lower_blocks, ranks = lower_ranks)
    max_num_upper_rots = 2*lower_rank_max * (2*(n÷num_blocks - 1) + upper_rank_max ) + 2*(n÷num_blocks - 1) + upper_rank_max
    max_num_lower_rots = 2*(n÷num_blocks - 1) + lower_rank_max
    upper_rank_max = 2*(n÷num_blocks - 1) + upper_rank_max
    lower_rank_max = 2*lower_rank_max
    max_num_upper_rots = 2*lower_rank_max * (2*(n÷num_blocks - 1) + upper_rank_max ) + 2*(n÷num_blocks - 1) + upper_rank_max
    max_num_lower_rots = 2*(n÷num_blocks - 1) + lower_rank_max
    upper_blocks, lower_blocks = random_blocks(n,num_blocks)
    gw1 = GivensWeight(
      Float64,
      TrailingDecomp(),
      MersenneTwister(),
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
    A=Matrix(gw1)  
    preparative_phase!(gw1)
    triang_rot = residual_phase!(gw1)
    QB = Matrix(1.0I,n,n)
    QT = Matrix(1.0I,n,n)
    apply_rotations!(triang_rot,QT)
    apply_rotations_reverse!(gw1.lowerRots,QB)
    gw1.lowerRots .= Rot(one(Float64), zero(Float64), 1)
    @testset "||A - QR||" begin
      @test norm(A - QB'*QT'*Matrix(gw1), Inf) <= tol
    end
  end
  