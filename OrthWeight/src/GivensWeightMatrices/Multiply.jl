#Right Multiplication
# C ← α A B + β C
function LinearAlgebra.mul!(
  C::AbstractMatrix{E},
  A::GivensWeight,
  B::AbstractVecOrMat,
  α,
  β,
  work::AbstractMatrix{E} = similar(
    B,
    max(size(C, 1), size(B, 1)),
    max(size(C, 2), size(B, 2)),
  ),
) where {E<:Number}

  Base.require_one_based_indexing(C, B, work)

  ma, na = size(A.b)
  mb, nb = size(B)
  mc, nc = size(C)

  na != mb &&
    throw(DimensionMismatch(lazy"A has dimensions ($ma,$na) but B has dimensions ($mb,$nb)"))

  (mc != ma || nc != nb) &&
    throw(DimensionMismatch(lazy"C has dimensions $(size(C)), should have ($ma,$nb)"))

  @. C = β * C
  for l = 1:nc
    for k = 1:na
      αbkl = α * B[k, l]
      for j in middle_inband_index_range(A.b, :, k)
        C[j, l] += A.b[j, k] * αbkl
      end
    end
  end

  if A.lower_decomp[] isa LeadingDecomp
    tmp = view(work, 1:mb, 1:nb)
    tmp .= B
    for lb_ind ∈ A.b.lower_blocks
      g_ind = A.b.lower_blocks[lb_ind].givens_index
      for j ∈ 1:(A.b.lower_blocks[lb_ind].num_rots)
        r = A.lowerRots[j, g_ind]
        apply_inv!(r, tmp)
      end
      rows_lb, cols_lb = lower_block_ranges(A.b, lb_ind)
      if !isempty(rows_lb) && !isempty(cols_lb)
        cols = first_inband_index(A.b, first(rows_lb), :):last(cols_lb)
        lb1 = next_list_index(A.b.lower_blocks, lb_ind)
        rows_lb1, _ = lower_block_ranges(A.b, lb1)
        row_gap = setdiffᵣ(rows_lb, rows_lb1)
        r0 = last_inband_index(A.b, :, last(cols_lb))
        r1 = isempty(row_gap) ? r0 : last(row_gap)
        rows = first(rows_lb):min(r0, r1)
        mul!(
          view(C, rows, :),
          view(A.b, rows, cols),
          view(tmp, cols, :),
          α,
          one(E),
        )
      end
    end
  elseif A.lower_decomp[] isa TrailingDecomp
    tmp = view(work, 1:mc, 1:nc)
    tmp .= zero(E)
    for lb_ind ∈ A.b.lower_blocks
      rows_lb, cols_lb = lower_block_ranges(A.b, lb_ind)
      if !isempty(rows_lb) && !isempty(cols_lb)
        rows = first(rows_lb):last_inband_index(A.b, :, last(cols_lb))
        lb1 = prev_list_index(A.b.lower_blocks, lb_ind)
        _, cols_lb1 = lower_block_ranges(A.b, lb1)
        col_gap = setdiffᵣ(cols_lb, cols_lb1)
        c0 = first_inband_index(A.b, first(rows_lb), :)
        c1 = isempty(col_gap) ? c0 : first(col_gap)
        cols = max(c0, c1):last(cols_lb)
        mul!(
          view(tmp, rows, :),
          view(A.b, rows, cols),
          view(B, cols, :),
          one(E),
          one(E),
        )
      end
      g_ind = A.b.lower_blocks[lb_ind].givens_index
      for j ∈ (A.b.lower_blocks[lb_ind].num_rots):-1:1
        r = A.lowerRots[j, g_ind]
        apply!(r, tmp)
      end
    end
    @. C += α * tmp
  else
    throw(NeitherLeadingNorTrailing(lazy"Lower"))
  end

  if A.upper_decomp[] isa LeadingDecomp
    tmp = view(work, 1:mc, 1:nc)
    tmp .= zero(E)
    for ub_ind ∈ Iterators.Reverse(A.b.upper_blocks)
      rows_ub, cols_ub = upper_block_ranges(A.b, ub_ind)
      if !isempty(rows_ub) && !isempty(cols_ub)
        rows = first_inband_index(A.b, :, first(cols_ub)):last(rows_ub)
        ub1 = next_list_index(A.b.upper_blocks, ub_ind)
        _, cols_ub1 = upper_block_ranges(A.b, ub1)
        col_gap = setdiffᵣ(cols_ub, cols_ub1)
        c0 = last_inband_index(A.b, last(rows_ub), :)
        c1 = isempty(col_gap) ? c0 : last(col_gap)
        cols = first(cols_ub):min(c0, c1)
        mul!(
          view(tmp, rows, :),
          view(A.b, rows, cols),
          view(B, cols, :),
          one(E),
          one(E),
        )
      end
      g_ind = A.b.upper_blocks[ub_ind].givens_index
      for j ∈ (A.b.upper_blocks[ub_ind].num_rots):-1:1
        r = A.upperRots[j, g_ind]
        apply!(r, tmp)
      end
    end
    @. C += α * tmp
  elseif A.upper_decomp[] isa TrailingDecomp
    tmp = view(work, 1:mb, 1:nb)
    tmp .= B
    for ub_ind ∈ Iterators.reverse(A.b.upper_blocks)
      g_ind = A.b.upper_blocks[ub_ind].givens_index
      for j ∈ 1:(A.b.upper_blocks[ub_ind].num_rots)
        r = A.upperRots[j, g_ind]
        apply_inv!(r, tmp)
      end
      rows_ub, cols_ub = upper_block_ranges(A.b, ub_ind)
      if !isempty(rows_ub) && !isempty(cols_ub)
        cols = first(cols_ub):last_inband_index(A.b, last(rows_ub), :)
        ub1 = prev_list_index(A.b.upper_blocks, ub_ind)
        rows_ub1, _ = upper_block_ranges(A.b, ub1)
        row_gap = setdiffᵣ(rows_ub, rows_ub1)
        r0 = first_inband_index(A.b, :, first(cols_ub))
        r1 = isempty(row_gap) ? r0 : first(row_gap)
        rows = max(r0, r1):last(rows_ub)
        mul!(
          view(C, rows, :),
          view(A.b, rows, cols),
          view(tmp, cols, :),
          α,
          one(E),
        )
      end
    end
  else
    throw(NeitherLeadingNorTrailing(lazy"Upper"))
  end

  return C
end

function LinearAlgebra.mul!(
  c::AbstractVector,
  A::GivensWeight,
  b::AbstractVector,
  α::Number,
  β::Number,
)
  mul!(reshape(c, length(c), 1), A, reshape(b, length(b), 1), α, β)
  return c
end


function LinearAlgebra.mul!(
  C::AbstractMatrix{E},
  A::GivensWeight,
  B::AbstractVecOrMat,
) where {E <: Number}

  C .= zero(E)
  mul!(C, A, B, one(E), zero(E))
  return C
  
end

function LinearAlgebra.mul!(
  C::AbstractVector{E},
  A::GivensWeight,
  B::AbstractVecOrMat,
) where {E <: Number}

  C .= zero(E)
  mul!(C, A, B, one(E), zero(E))
  return C
  
end

function Base.:*(A::GivensWeight,
                 b::AbstractVector)
  c = similar(b, size(A,1))
  c .= zero(eltype(c))
  mul!(c, A, b)
  return c
end

function Base.:*(A::GivensWeight,
                 B::AbstractMatrix)
  C = similar(B, size(A,1), size(B,2))
  C .= zero(eltype(C))
  mul!(C, A, B)
  return C
end