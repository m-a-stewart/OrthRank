using LinearAlgebra
using BandStruct
using OrthWeight.GivensWeightMatrices
using OrthWeight.BasicTypes
using Random
# using Test

# function power_iteration(
#   gw::GivensWeight,
#   num_it::Int64
#   )
#   m, n = size(gw.b)
#   if m >= n  #⇒b'*A*A'*b
#     b = randn(m,1)
#     for i in 1:num_it 
#       b = gw * leftmul(b', gw)'
#       b = b/norm(b)
#     end
#     aux = leftmul(b', gw)
#     max_s_value = sqrt(b' * (gw * aux'))
#   elseif n > m # ⇒ b'*A'*A*b
#     b = randn(n,1)
#     for i in 1:num_it
#       b = leftmul((gw * b)', gw)'
#       b = b/norm(b)
#     end
#     aux = gw * b
#     max_s_value = sqrt(b' * leftmul(aux', gw)') 
#   end
#   return max_s_val
# end

# function leftmul(
#     B::AbstractMatrix,
#     A::GivensWeight
#   )
#   ma, na = size(A.b)
#   mb, nb = size(B)
#   result = similar(B, mb, na)
#   nb != ma &&
#   throw(DimensionMismatch(lazy" B has dimensions ($mb,$nb) but A has dimensions ($ma,$na)"))
#   for k in 1:na
#     row_s = first_inband_index(A.b,:,k)
#     row_e = last_inband_index(A.b,:,k)
#     for l in 1:mb
#       result[l,k] = dot(B[l,row_s:row_e], gw.b[row_s:row_e,k])
#     end
#   end
#   return result
# end  

lower_blocks = givens_block_sizes([
  1 3 5 7 9
  1 2 4 5 10
])

upper_blocks = givens_block_sizes([
  1 3 5 7 9
  1 2 4 5 10
])

upper_ranks = Consts(length(upper_blocks), 4)
lower_ranks = Consts(length(lower_blocks), 4)
rng = MersenneTwister(1234)

m= 1000
n= 1200
E = Float64
decomp =TrailingDecomp()
gw = GivensWeight(
  E,
  decomp,
  rng,
  m,
  n;
  upper_rank_max = maximum(upper_ranks),
  lower_rank_max = maximum(lower_ranks),
  upper_ranks = upper_ranks,
  lower_ranks = lower_ranks,
  upper_blocks = upper_blocks,
  lower_blocks = lower_blocks,
  max_num_upper_rots=12, 
  max_num_lower_rots=12, 
)
A = Matrix(gw)
max_s_val = svd(A).S[1]
my_max_s_val = norm_2_gw(gw,100)
abs(my_max_s_val - max_s_val) < 1e-16
# B = Matrix(gw.b)
# b = randn(1,m)
# the_answer = b*B
# println(ans)
# my_answer = leftmul(b,gw)