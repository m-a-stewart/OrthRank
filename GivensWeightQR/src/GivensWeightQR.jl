module GivensWeightQR

using Random,
  OrthWeight, BandStruct, Householder, InPlace, Rotations, LinearAlgebra

export random_blocks_generator,
  random_blocks_generator_no_overlap,
  get_ranks,
  _cut_B,
  preparative_phase!,
  residual_phase!,
  @swap,
  apply_rotations_forward!,
  apply_rotations_backward!,
  create_Q!,
  create_R,
  solve!,
  QRGivensWeight,
  QGivensWeight,
  norm_2_gw,
  leftmul

"""
  	struct QGivensWeight{M<:AbstractMatrix}
Q matrix of a QR factorization storaged as Givens Rotations.
It is the result of applying qr(G) where G is a GivensWeight Matrix.
"""
struct QGivensWeight{M<:AbstractMatrix}
  Qb::M
  Qt::M
end

struct QRGivensWeight{S<:QGivensWeight,R<:AbstractGivensWeight} <:
       Factorization{R}
  Q::S
  R::R
end

#Convenience: return only where solution is storage.
_cut_B(x::AbstractVector, r::UnitRange) = length(r) < length(x) ? x[r] : x
_cut_B(X::AbstractMatrix, r::UnitRange) = length(r) < size(X, 1) ? X[r, :] : X

function last_inband_nonzero_index(
  B::BlockedBandColumn,
  ::Colon,
  col::Int,
)
  m, n = size(B)
  @inbounds if col < n
    ind = last_inband_index(B, :, col)
    while !iszero(ind) && iszero(B[ind, col])
      ind -= 1
    end
  elseif col == n
    ind = m
  end
  return ind
end

function last_inband_nonzero_index(
  B::BlockedBandColumn,
  row::Int,
  ::Colon
)
  m, n = size(B)
  @inbounds if row < m
    ind = last_inband_index(B, row, :)
    while !iszero(ind) && iszero(B[row, ind])
      ind -= 1
    end
  elseif row == m
    ind = n
  end
  return ind
end

function LinearAlgebra.qr(gw::GivensWeight)
  preparative_phase!(gw)
  triang_rot = residual_phase!(gw)
  Qb = QGivensWeight(gw.lowerRots, triang_rot)
  return QRGivensWeight(Qb, gw)
end

Base.:\(A::GivensWeight, B::AbstractVecOrMat) = qr(A) \ B
Base.:\(F::QRGivensWeight, B::AbstractVecOrMat) = ldiv(F, B)

function ldiv(F::QRGivensWeight, B::AbstractVecOrMat{<:Number})
  m, n = size(F.R.b)
  p = size(B)[1]
  m != p &&
    throw(DimensionMismatch("arguments must have the same number of rows"))
  for i = 1:min(m, n)
    iszero(F.R.b[i, i]) && throw(SingularException(i))
  end
  B isa Vector ? BB = zeros(Float64, max(m, n)) :
  BB = zeros(Float64, max(m, n), size(B, 2))
  m < n ? copyto!(view(BB, 1:m, :), B) : copyto!(view(BB, :, :), B)
  ldiv!(F, BB)
  return _cut_B(BB, 1:n)
end

LinearAlgebra.ldiv!(A::QRGivensWeight, b::AbstractVector) =
  ldiv!(A, reshape(b, length(b), 1))
LinearAlgebra.ldiv!(A::QRGivensWeight, b::AbstractMatrix) = solve!(A, b)

function solve!(A::QRGivensWeight, B::AbstractMatrix)
  m, n = size(A.R.b)
  s = min(m, n)
  #Apply lower block rotations to B
  for j in Iterators.reverse(axes(A.Q.Qb, 2))
    for i in axes(A.Q.Qb, 1)
      !iszero(A.Q.Qb[i, j].inds) && apply_inv!(A.Q.Qb[i, j], B)
    end
  end
  #Apply triangularizing rotations to b
  for j in axes(A.Q.Qt, 2)
    for i in axes(A.Q.Qt, 1)
      !iszero(A.Q.Qt[i, j].inds) && apply_inv!(A.Q.Qt[i, j], B)
    end
  end
  #Backward substitution
  @. B[s, :] = B[s, :] / A.R.b[s, s]
  B_copy = copy(B)
  ub_g_ind = length(A.R.b.upper_blocks)
  ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
  while s ≤ ub_mb
    ub_g_ind -= 1
    iszero(ub_g_ind) ? ub_mb = 0 : ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
  end
  for i = (s - 1):-1:1
    #Apply upper rot
    while i == ub_mb
      ub_rot_num = A.R.b.upper_blocks[ub_g_ind].num_rots
      for r = 1:ub_rot_num
        rot = A.R.upperRots[r, ub_g_ind]
        apply_inv!(rot, B_copy)
      end
      ub_g_ind -= 1
      iszero(ub_g_ind) ? ub_mb = 0 : ub_mb = A.R.b.upper_blocks[ub_g_ind].mb
    end
    #Solve
    last_el_in_row_ind = last_inband_nonzero_index(A.R.b, i, :)
    for k in axes(B, 2)
      B[i, k] =
        A.R.b[i, i] \ (
          B[i, k] - dot(
            view(A.R.b, i:i, (i + 1):last_el_in_row_ind),
            view(B_copy, (i + 1):last_el_in_row_ind, k),
          )
        )
      B_copy[i, k] = B[i, k]
    end
  end
end

macro swap(x, y)
  quote
    local tmp = $(esc(x))
    $(esc(x)) = $(esc(y))
    $(esc(y)) = tmp
  end
end

random_blocks_generator(rng::AbstractRNG, m::Int64; gap::Int64) =
  random_blocks_generator(rng, m, m, gap)

function random_blocks_generator_no_overlap(
  rng::AbstractRNG,
  m::Int64,
  n::Int64,
  gap::Int64,
)
  min(m, n) < gap && throw("Gap block bigger than smallest matrix size.")
  mn = min(m, n)
  δ = (gap + 1) / 2
  upper_rows = zeros(Int64, 1, mn) #upper_block doest not work with vectors
  upper_cols = zeros(Int64, 1, mn)
  lower_rows = zeros(Int64, 1, mn)
  lower_cols = zeros(Int64, 1, mn)
  i = 1
  while round(Int64, i * δ, RoundDown) ≤ mn
    δ_range =
      (round(Int64, (i - 1) * δ, RoundDown) + 1):round(Int64, i * δ, RoundDown)
    upper_rows[1, i] = rand(rng, δ_range)
    upper_cols[1, i] = rand(rng, δ_range)
    lower_rows[1, i] = rand(rng, δ_range)
    lower_cols[1, i] = rand(rng, δ_range)
    while lower_rows[1, i] < upper_rows[1, i] &&
    upper_cols[1, i] < lower_cols[1, i]
      upper_rows[1, i] = rand(rng, δ_range)
      upper_cols[1, i] = rand(rng, δ_range)
      lower_rows[1, i] = rand(rng, δ_range)
      lower_cols[1, i] = rand(rng, δ_range)
    end
    i += 1
  end
  upper_block = givens_block_sizes([
    upper_rows[1:1, 1:(i - 1)]
    upper_cols[1:1, 1:(i - 1)]
  ])
  lower_block = givens_block_sizes([
    lower_rows[1:1, 1:(i - 1)]
    lower_cols[1:1, 1:(i - 1)]
  ])
  return upper_block, lower_block
end

function random_blocks_generator(
  rng::AbstractRNG,
  m::Int64,
  n::Int64,
  gap::Int64,
)
  min(m, n) < gap && throw("Gap between blocks is bigger
   than the smallest size of the matrix.")
  mn = min(m, n)
  δ = (gap + 1) / 2
  upper_rows = zeros(Int64, 1, mn) #upper_block doest not work with vectors
  upper_cols = zeros(Int64, 1, mn)
  lower_rows = zeros(Int64, 1, mn)
  lower_cols = zeros(Int64, 1, mn)
  i = 1
  while round(Int64, i * δ, RoundDown) ≤ mn
    δ_range =
      (round(Int64, (i - 1) * δ, RoundDown) + 1):round(Int64, i * δ, RoundDown)
    upper_rows[1, i] = rand(rng, δ_range)
    upper_cols[1, i] = rand(rng, δ_range)
    lower_rows[1, i] = rand(rng, δ_range)
    lower_cols[1, i] = rand(rng, δ_range)
    #No diagonal (i, i) in block
    upper_cols[1, i] < upper_rows[1, i] &&
      @swap(upper_rows[1, i], upper_cols[1, i])
    lower_rows[1, i] < lower_cols[1, i]  &&
      @swap(lower_cols[1, i], lower_rows[1, i])
    i += 1
  end
  upper_block = givens_block_sizes([
    upper_rows[1:1, 1:(i - 1)]
    upper_cols[1:1, 1:(i - 1)]
  ])
  lower_block = givens_block_sizes([
    lower_rows[1:1, 1:(i - 1)]
    lower_cols[1:1, 1:(i - 1)]
  ])
  return upper_block, lower_block
end

function get_ranks(gw::GivensWeight)
  num_lower_blocks = length(gw.b.lower_blocks)
  num_upper_blocks = length(gw.b.upper_blocks)
  lower_ranks = zeros(Int64, num_lower_blocks)
  upper_ranks = zeros(Int64, num_upper_blocks)
  for i = 1:num_lower_blocks
    lower_ranks[i] = gw.b.lower_blocks.data[i].block_rank
  end
  for i = 1:num_upper_blocks
    upper_ranks[i] = gw.b.upper_blocks.data[i].block_rank
  end
  return lower_ranks, upper_ranks
end

function preparative_phase!(gw::GivensWeight)
  m, n = size(gw.b)
  ub_g_ind = length(gw.b.upper_blocks)
  while gw.b.upper_blocks[ub_g_ind].mb == m && !iszero(ub_g_ind)
    ub_g_ind -= 1
  end
  iszero(ub_g_ind) ? ub_mb = 0 : ub_mb = gw.b.upper_blocks[ub_g_ind].mb
  @inbounds for lb_ind in filter_compressed(Iterators.Reverse(gw.b.lower_blocks))
    lb_g_ind = gw.b.lower_blocks[lb_ind].givens_index
    lb_rot_num = gw.b.lower_blocks[lb_g_ind].num_rots
    lb_rank = gw.b.lower_blocks[lb_g_ind].block_rank
    lb_nb = gw.b.lower_blocks[lb_g_ind].nb
    for lb_rot_nth = 1:lb_rot_num
      lb_rot = gw.lowerRots[lb_rot_nth, lb_g_ind]
      lb_rot_inds = lb_rot.inds
      lb_rot_rank = lb_rot_inds + lb_rank
      while lb_rot_inds ≤ ub_mb && ub_mb < lb_rot_rank
        ub_rot_num = gw.b.upper_blocks[ub_g_ind].num_rots
        ub_active_rows = (ub_mb + 1):(lb_rot_inds + lb_rank)
        for ub_rot_num = 1:ub_rot_num
          ub_rot = gw.upperRots[ub_rot_num, ub_g_ind]
          ub_active_cols = 1:(ub_rot.inds + 1)
          ub_vb = view(gw.b, ub_active_rows, ub_active_cols)
          apply!(ub_vb, ub_rot)
        end
        #Reduce fill-up caused by extension.
        for ub_fill_up_row in ub_active_rows
          start_column_ind =
            last_inband_nonzero_index(gw.b, ub_fill_up_row - 1, :)
          finish_column_ind = last_inband_nonzero_index(gw.b, ub_fill_up_row,:)
          fill_up_vb = view(gw.b, ub_active_rows, 1:finish_column_ind)
          for j = (finish_column_ind - 1):-1:(start_column_ind + 1)
            surv = j
            kill = j + 1
            fill_up_rot = rgivens(gw.b[ub_fill_up_row, surv], 
              gw.b[ub_fill_up_row, kill], surv)
            apply!(fill_up_vb, fill_up_rot)
            gw.b[ub_fill_up_row, kill] = zero(eltype(gw.b)) #Fix NaN issue.
            #notch_upper do not notch below diagonal.
            gw.b.upper_blocks[ub_g_ind].num_rots += 1 
            gw.upperRots[gw.b.upper_blocks[ub_g_ind].num_rots, ub_g_ind] = 
              fill_up_rot
          end
        end
        gw.b.upper_blocks[ub_g_ind].mb = lb_rot_inds + lb_rank
        ub_new_last_col_ind =
          last_inband_nonzero_index(gw.b, lb_rot_inds + lb_rank, :)
        gw.b.upper_blocks[ub_g_ind].block_rank =
          ub_new_last_col_ind - gw.b.upper_blocks[ub_g_ind].nb
        ub_g_ind -= 1
        iszero(ub_g_ind) ? ub_mb = 0 : ub_mb = gw.b.upper_blocks[ub_g_ind].mb
      end
      #Apply lb_rot in the compl. of lb_block
      if lb_nb != n
        last_col_el = last_inband_nonzero_index(gw.b, lb_rot_inds + 1, :)
        lb_active_cols = (lb_nb + 1):last_col_el
        lb_active_rows = 1:(lb_rot_inds + 1)
        lb_vb = view(gw.b, lb_active_rows, lb_active_cols)
        apply_inv!(lb_rot, lb_vb)
      end
    end
  end
end

function rotations_needed(
  ::Type{E},
  B::AbstractBandColumn,
  n::Int64,
) where {R<:Real,E<:Union{R,Complex{R}}}
  m = [0]
  last_el_index = one(Int64)
  for d_index = 1:n
    try
      last_el_index = last_inband_nonzero_index(B, :, d_index)
    catch _
      last_el_index = n
    end
    amount = last_el_index - d_index
    m[1] < amount && setindex!(m, [amount], [1])
  end
  matrix_of_rotations = fill(Rot(zero(R), zero(E), 0),m[1], n)
  return matrix_of_rotations
end

function residual_phase!(gw::GivensWeight)
  m, n = size(gw.b)
  diag_size = min(m, n)
  triang_rot = rotations_needed(Float64, gw.b, diag_size)
  last_ub_g_ind = length(gw.b.upper_blocks)
  ub_g_ind = 1
  d_ind_inside_ub_block = false
  while iszero(gw.b.upper_blocks[ub_g_ind].mb) && ub_g_ind ≤ last_ub_g_ind
    ub_g_ind += 1
  end
  last_ub_g_ind < ub_g_ind ? ub_mb = 0 : 
    ub_mb = gw.b.upper_blocks[ub_g_ind].mb
  for d_ind = 1:diag_size
    leud = last_inband_nonzero_index(gw.b, :, d_ind) #LastElementUnderDiagonal
    qty_und_diag = leud - d_ind
    leud == m ? virtual_leud = m + 1 : virtual_leud = leud
    lb_act_rows = 1:leud
    # Regression action
    last_ub_g_ind < ub_g_ind ? ub_nb = diag_size + 1 :
    ub_nb = gw.b.upper_blocks[ub_g_ind].nb
    if ub_nb ≤ d_ind
      d_ind_inside_ub_block = true
    end
    while (d_ind ≤ ub_mb < virtual_leud)   || d_ind_inside_ub_block
      ub_act_rows =  d_ind:ub_mb
      for ub_rot_num = (gw.b.upper_blocks[ub_g_ind].num_rots):-1:1
        ub_rot = gw.upperRots[ub_rot_num, ub_g_ind]
        ub_act_cols = 1:(ub_rot.inds + 1)
        ub_vb = view(gw.b, ub_act_rows, ub_act_cols)
        apply_inv!(ub_vb, ub_rot)
      end
      gw.b.upper_blocks[ub_g_ind].mb = d_ind - 1
      ub_g_ind += 1
      last_ub_g_ind < ub_g_ind ? ub_mb = 0 :
        ub_mb = gw.b.upper_blocks[ub_g_ind].mb
      d_ind_inside_ub_block = false
    end
    #Create zeros under diagonal
    for k = qty_und_diag:-1:1
      surv = d_ind + k - 1
      kill = d_ind + k
      rot = lgivens(gw.b[surv, d_ind], gw.b[kill, d_ind], surv)
      last_el_in_row = last_inband_nonzero_index(gw.b, kill, :)
      lb_act_cols = d_ind:last_el_in_row
      vb = view(gw.b, lb_act_rows, lb_act_cols)
      apply_inv!(rot, vb)
      triang_rot[1 + qty_und_diag - k, d_ind] = rot
    end
  end
  return triang_rot
end

function apply_rotations_forward!(A::Matrix, rot::Matrix)
  for j in axes(rot, 2)
    for i in Iterators.reverse(axes(rot, 1))
      !iszero(rot[i, j].inds) && apply!(rot[i, j], A)
    end
  end
end

function apply_rotations_backward!(A::Matrix, rot::Matrix)
  for j in Iterators.reverse(axes(rot, 2))
    for i in Iterators.reverse(axes(rot, 1))
      !iszero(rot[i, j].inds) && apply!(rot[i, j], A)
    end
  end
end

function create_Q!(A::Matrix, F::QRGivensWeight)
  apply_rotations_backward!(A, F.Q.Qt)
  apply_rotations_forward!(A, F.Q.Qb)
end

function create_R(F::QRGivensWeight)
  F.R.lowerRots .= Rot(one(Float64), zero(Float64), 1)
  A = Matrix(F.R)
  return A
end

# function power_iteration(
#   A::AbstractMatrix,
#   num_it::Int64
#   )
#   #Power iteration Section 
#   m, n = size(A)
#   b = randn(m,1)
#   for i in 1:num_it
#     b = A * A' * b
#     b = b/norm(b)
#   end
#   da_norm = sqrt((b'*(A*A'*b))/(b'*b))
#   return da_norm
# end

function norm_2_gw(
  gw::GivensWeight,
  num_it::Int64
  )
  m, n = size(gw.b)
  if m >= n  #⇒b'*A*A'*b
    b = randn(m,1)
    for i in 1:num_it 
      b = gw * leftmul(b', gw)'
      b = b/norm(b)
    end
    aux = leftmul(b', gw)
    max_s_value = sqrt(b' * (gw * aux'))
  elseif n > m # ⇒ b'*A'*A*b
    b = randn(n,1)
    for i in 1:num_it
      b = leftmul((gw * b)', gw)'
      b = b/norm(b)
    end
    aux = gw * b
    max_s_value = sqrt(b' * leftmul(aux', gw)') 
  end
  return max_s_value[1,1]
end

function leftmul(
    B::AbstractMatrix,
    gw::GivensWeight,
  )
  ma, na = size(gw.b)
  mb, nb = size(B)
  work = similar(B, mb, na)
  nb != ma &&
  throw(DimensionMismatch(lazy" B has dimensions ($mb,$nb) but A has dimensions ($ma,$na)"))
  for k in 1:na
    row_s = first_inband_index(gw.b,:,k)
    row_e = last_inband_index(gw.b,:,k)
    for l in 1:mb
      work[l,k] = dot(B[l,row_s:row_e], gw.b[row_s:row_e,k])
    end
  end
  return work
end  

using PrecompileTools
@setup_workload begin
  include("Precompile/Precompile.jl")
  import .Precompile
  @compile_workload begin
    Precompile.run_all()
  end
end

end #module
