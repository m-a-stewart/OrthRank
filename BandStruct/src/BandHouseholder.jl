module BandHouseholder

using LinearAlgebra
using Printf
using LoopVectorization

using Householder.Compute
using Householder.WY

using ..BandColumnMatrices
using ..BlockedBandColumnMatrices
using InPlace

# Lifted from Octavian Benchmarks.
function lvmul_threads!(C, A, B)
    @avxt for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = zero(eltype(C))
        for k ∈ indices((A,B), (2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

@inline function lvmul_minus_threads!(C, A, B)
    @avxt for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = C[m,n]
        for k ∈ indices((A,B), (2,1))
            Cmn -= A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

@inline function lvmul_no_threads!(C, A, B)
    @avx for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = zero(eltype(C))
        for k ∈ indices((A,B), (2,1))
            Cmn += A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

function lvmul_minus_no_threads!(C, A, B)
    @avx for n ∈ indices((C,B), 2), m ∈ indices((C,A), 1)
        Cmn = C[m,n]
        for k ∈ indices((A,B), (2,1))
            Cmn -= A[m,k] * B[k,n]
        end
        C[m,n] = Cmn
    end
end

# vector and work.
Base.@propagate_inbounds function Compute.householder(
  bc::AbstractBandColumn{S,E},
  js::UnitRange{Int},
  k::Int,
  l::Int,
  offs::Int,
  v::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number,S}
  lv = length(v)
  ljs = length(js)
  @boundscheck begin
    lv >= ljs || throw(DimensionMismatch(@sprintf(
      """
      In lhouseholder(bc::AbstractBandColumn{S,E}, js, k, l, offs, v, work),
      length(v) is %d and length(js) is %d.  These should be equal.
      """,
      lv,
      ljs
    )))
    ((1:ljs) .+ offs) ⊆ storable_index_range(bc, :, k) ||
      throw(SubcolumnIndicesNotInband(js, k))
  end

  storage_offs = storage_offset(bc, k)
  bc_els = band_elements(bc)

  @inbounds for j ∈ 1:ljs
    v[j] = bc_els[j + offs - storage_offs, k]
  end

  @views lhouseholder(v[1:ljs], l, offs, work)
end

Base.@propagate_inbounds function Compute.householder(
  bc::AbstractBandColumn{S,E},
  j::Int,
  ks::UnitRange{Int},
  l::Int64,
  offs::Int64,
  v::AbstractArray{E,1},
  work::AbstractArray{E,1},
) where {E<:Number,S}
  lv = length(v)
  lks = length(ks)
  @boundscheck begin
    lv >= lks || throw(DimensionMismatch(@sprintf(
      """
      In lhouseholder(bc::AbstractBandColumn{S,E}, j, ks, l, offs, v, work),
      length(v) is %d and length(ks) is %d.  These should be equal.
      """,
      lv,
      lks
    )))
    ((1:lks) .+ offs) ⊆ storable_index_range(bc, j, :) ||
      throw(SubrowIndicesNotInband(j, ks))
  end

  bc_els = band_elements(bc)

  @inbounds for k ∈ 1:lks
    storage_offs = storage_offset(bc, k + offs)
    v[k] = bc_els[j - storage_offs, k + offs]
  end
  @views rhouseholder(v[1:lks], l, offs, work)
end


@inline function InPlace.apply!(
  bc::AbstractBandColumn{S,E},
  h::HouseholderTrans{E},
) where {E<:Number,S}

  m = h.size
  v = h.v
  offs = h.offs
  β = h.β

  work = h.work
  lw = length(work)

  k_first = offs + 1
  k_last = offs + h.size
  n_bc = k_last - k_first + 1

  js = inband_hull(bc, :, k_first:k_last)
  m_bc = length(js)

  bc_els = band_elements(bc)

  if m_bc > 0 && n_bc > 0
    j_first = first(js)
    j_last = last(js)
    @boundscheck begin
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)

      lw >= m_bc || throw(DimensionMismatch(@sprintf(
        """
        In bc ⊛ h, h has work array of length %d. The active block
        of bc is %d×%d and requires a work array of length %d
        (the number of rows in the block).
        """,
        length(work),
        m_bc,
        n_bc,
        m_bc
      )))
    end

    @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)
      work[1:m_bc] .= zero(E)

      # Accumulate w = bc * v in work array by a linear combination of
      # columns of bc.
      for k ∈ 1:m
        storage_offs = storage_offset(bc, k + offs)
        x=v[k]
        @simd for j ∈ j_first:j_last
          work[j - j_first + 1] += bc_els[j - storage_offs, k + offs] * x
        end
      end
      # Subtract β * w * vᴴ from bc.
      for k ∈ 1:m
        storage_offs = storage_offset(bc, k + offs)
        x = β * conj(v[k])
        @simd for j ∈ j_first:j_last
          bc_els[j - storage_offs, k + offs] -= work[j - j_first + 1] * x
        end
      end
    end
  end
  nothing
end

@inline function InPlace.apply!(
  h::HouseholderTrans{E},
  bc::AbstractBandColumn{S,E},
) where {E<:Number,S}

  m = h.size
  v = h.v
  offs = h.offs
  β = h.β

  j_first = offs + 1
  j_last = offs + h.size
  m_bc = j_last - j_first + 1

  ks = inband_hull(bc, j_first:j_last, :)

  n_bc = length(ks)
  bc_els = band_elements(bc)

  if m_bc > 0 && n_bc > 0
    k_first = first(ks)
    k_last = last(ks)
    @boundscheck begin
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)
    end
    @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)
      for k ∈ k_first:k_last
        x = zero(E)
        storage_offs = storage_offset(bc, k)
        # Form x = vᴴ * bc[:,k].
        @simd for j ∈ j_first:j_last
          x += conj(v[j - offs]) * bc_els[j - storage_offs, k]
        end
        # Subtract v * x from bc[:,k].
        x = β * x
        @simd for j ∈ 1:m
          bc_els[offs + j - storage_offs, k] -= v[j] * x
        end
      end
    end
  end
  nothing
end

@inline function InPlace.apply_inv!(
  bc::AbstractBandColumn{S,E},
  h::HouseholderTrans{E},
) where {E<:Number,S}

  m = h.size
  v = h.v
  offs = h.offs
  β̄ = conj(h.β)

  work = h.work
  lw = length(work)

  k_first = offs + 1
  k_last = offs + h.size
  n_bc = k_last - k_first + 1

  js = inband_hull(bc, :, k_first:k_last)
  m_bc = length(js)

  bc_els = band_elements(bc)

  if m_bc > 0 && n_bc > 0
    j_first = first(js)
    j_last = last(js)

    @boundscheck begin
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)

      lw >= m_bc || throw(DimensionMismatch(@sprintf(
        """
        In bc ⊛ h, h has work array of length %d. The active block
        of bc is %d×%d and requires a work array of length %d
        (the number of rows in the block).
        """,
        length(work),
        m_bc,
        n_bc,
        m_bc
      )))
    end

    @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)
      work[1:m_bc] .= zero(E)

      # Accumulate w = bc * v in work array by a linear combination of
      # columns of bc.
      for k ∈ 1:m
        storage_offs = storage_offset(bc, k + offs)
        x=v[k]
        @simd for j ∈ j_first:j_last
          work[j - j_first + 1] += bc_els[j - storage_offs, k + offs] * x
        end
      end
      # Subtract β̄ * w * vᴴ from bc.
      for k ∈ 1:m
        storage_offs = storage_offset(bc, k + offs)
        x = β̄ * conj(v[k])
        @simd for j ∈ j_first:j_last
          bc_els[j - storage_offs, k + offs] -=
            work[j - j_first + 1] * x
        end
      end
    end
  end
  nothing
end

@inline function InPlace.apply_inv!(
  h::HouseholderTrans{E},
  bc::AbstractBandColumn{S,E},
) where {E<:Number,S}

  m = h.size
  v=h.v
  offs = h.offs
  β̄ = conj(h.β)

  j_first = offs + 1
  j_last = offs + h.size
  m_bc = j_last - j_first + 1

  ks = inband_hull(bc, j_first:j_last, :)
  n_bc = length(ks)

  bc_els = band_elements(bc)

  if m_bc > 0 && n_bc > 0
    k_first = first(ks)
    k_last = last(ks)
    @boundscheck begin
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)
    end
    work=h.work
    @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)
      for k ∈ k_first:k_last
        x=zero(E)
        storage_offs = storage_offset(bc, k)
        # Form x = vᴴ * bc[:,k].
        @simd for j ∈ 1:m
          x += conj(v[j]) * bc_els[j - storage_offs+offs, k]
        end
        work[k-k_first+1]=x
      end
      # Subtract v * x from bc[:,k].
      for k ∈ k_first:k_last
        storage_offs = storage_offset(bc, k)
        x = β̄ * work[k-k_first+1] 
        @simd for j ∈ 1:m
          bc_els[offs + j - storage_offs, k] -= v[j] * x
        end
      end
    end
  end
  nothing
end

@inline function InPlace.apply!(
  bc::AbstractBandColumn{S,E},
  wy::WYTrans{E},
  k::Int,
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:nbc ||
      throw_ColumnRange_DimensionMismatch(mbc, nbc, inds .+ offset)
  end

  lw = length(wy.work)

  n_bc0 = wy.sizes[k]
  k_first = offset + 1
  k_last = offset + n_bc0

  js = inband_hull(bc, :, k_first:k_last)
  m_bc0 = length(js)

  if m_bc0 > 0 && n_bc0 > 0

    j_first = first(js)
    j_last = last(js)

    @boundscheck begin
      lw >= m_bc0 * num_hs + m_bc0 * n_bc0 ||
        throw_WorkSizeError(mbc,
                            nbc,
                            m_bc0 * num_hs + m_bc0 * n_bc0,
                            length(wy.work))
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)
    end

    @views @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)
      work = reshape(wy.work[1:m_bc0*num_hs], m_bc0, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]

      work .= zero(E)

      tmp0 = reshape(
        work[(m_bc0 * num_hs + 1):(m_bc0 * num_hs + m_bc0 * n_bc0)],
        m_bc0,
        n_bc0,
      )

      copyto!(tmp0, bc[js, k_first:k_last])
      lvmul_threads!(work, tmp0, W)

      lvmul_minus_threads!(tmp0, work, Y')
      copyto!(bc[js, k_first:k_last], tmp0)
    end
  end
  nothing
end


@inline function InPlace.apply_inv!(
  bc::AbstractBandColumn{S,E},
  wy::WYTrans{E},
  k::Int
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:nbc ||
      throw_ColumnRange_DimensionMismatch(mbc, nbc, inds .+ offset)
  end

  lw = length(wy.work)
  n_bc0 = wy.sizes[k]
  k_first = offset + 1
  k_last = offset + n_bc0

  js = inband_hull(bc, :, k_first:k_last)
  m_bc0 = length(js)

  if m_bc0 > 0 && n_bc0 > 0
    j_first = first(js)
    j_last = last(js)

    @boundscheck begin
      lw >= m_bc0 * num_hs + m_bc0 * n_bc0 ||
        throw_WorkSizeError(mbc,
                            nbc,
                            m_bc0 * num_hs + m_bc0 * n_bc0,
                            length(wy.work))
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)
    end

    @views @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)

      work = reshape(wy.work[1:m_bc0*num_hs], m_bc0, num_hs)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]

      work .= zero(E)

      tmp0 = reshape(
        work[(m_bc0 * num_hs + 1):(m_bc0 * num_hs + m_bc0 * n_bc0)],
        m_bc0,
        n_bc0,
      )

      copyto!(tmp0, bc[js, k_first:k_last])
      lvmul_threads!(work,tmp0,Y)
      
      lvmul_minus_threads!(tmp0, work, W')
      copyto!(bc[js, k_first:k_last], tmp0)
    end
  end
  nothing
end

@inline function InPlace.apply!(
  wy::WYTrans{E},
  k::Int,
  bc::AbstractBandColumn{S,E}
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    m_bc0 = wy.sizes[k]
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:mbc ||
      throw_RowRange_DimensionMismatch(mbc, nbc, inds .+ offset)
  end

  lw = length(wy.work)

  j_first = offset + 1
  j_last = offset + m_bc0

  ks = inband_hull(bc, j_first:j_last, :)
  n_bc0 = length(ks)

  if m_bc0 > 0 && n_bc0 > 0
    
    k_first = first(ks)
    k_last = last(ks)

    @boundscheck begin
      lw >= n_bc0 * num_hs + m_bc0 * n_bc0 ||
        throw_WorkSizeError(mbc,
                            nbc,
                            n_bc0 * num_hs + m_bc0 * n_bc0,
                            length(wy.work))
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)
    end
    
    @views @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)

      work = reshape(wy.work[1:n_bc0*num_hs], num_hs, n_bc0)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]

      work .= zero(E)

      tmp0 = reshape(
        work[(n_bc0 * num_hs + 1):(n_bc0 * num_hs + m_bc0 * n_bc0)],
        m_bc0,
        n_bc0,
      )

      copyto!(tmp0, bc[j_first:j_last, ks])
      lvmul_threads!(work, Y', tmp0)

      lvmul_minus_threads!(tmp0, W, work)
      copyto!(bc[j_first:j_last, ks], tmp0)
    end
  end
  nothing
end

@inline function InPlace.apply_inv!(
  wy::WYTrans{E},
  k::Int,
  bc::AbstractBandColumn{S,E}
) where {E<:Number,S}

  num_WY = wy.num_WY[]
  @boundscheck k ∈ 1:num_WY || throw_WYBlockNotAvailable(k, num_WY)
  @inbounds begin
    inds = 1:wy.sizes[k]
    offset = wy.offsets[k]
    (mbc,nbc) = size(bc)
    m_bc0 = wy.sizes[k]
    num_hs= wy.num_hs[k]
  end

  @boundscheck begin
    inds .+ offset ⊆ 1:mbc ||
      throw_RowRange_DimensionMismatch(mbc, nbc, inds .+ offset)
  end

  lw = length(wy.work)

  j_first = offset + 1
  j_last = offset + m_bc0

  ks = inband_hull(bc, j_first:j_last, :)
  n_bc0 = length(ks)

  if m_bc0 > 0 && n_bc0 > 0
    
    k_first = first(ks)
    k_last = last(ks)

    @boundscheck begin
      lw >= n_bc0 * num_hs + m_bc0 * n_bc0 ||
        throw_WorkSizeError(mbc,
                            nbc,
                            n_bc0 * num_hs + m_bc0 * n_bc0,
                            length(wy.work))
      check_bc_storage_bounds(bc, j_first, k_last)
      check_bc_storage_bounds(bc, j_last, k_first)
    end
    
    @views @inbounds begin
      bulge_upper!(bc, j_first, k_last)
      bulge_lower!(bc, j_last, k_first)

      work = reshape(wy.work[1:n_bc0*num_hs], num_hs, n_bc0)
      W = wy.W[inds, 1:num_hs, k]
      Y = wy.Y[inds, 1:num_hs, k]

      work .= zero(E)

      tmp0 = reshape(
        work[(n_bc0 * num_hs + 1):(n_bc0 * num_hs + m_bc0 * n_bc0)],
        m_bc0,
        n_bc0,
      )

      copyto!(tmp0, bc[j_first:j_last, ks])
      lvmul_threads!(work, W', tmp0)

      lvmul_minus_threads!(tmp0, Y, work)
      copyto!(bc[j_first:j_last, ks], tmp0)
    end
  end
  nothing
end

end # module
