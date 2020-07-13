module LeadingBandColumnMatrices
using MLStyle
using Printf
using Random

import Base: size, getindex, setindex!, @propagate_inbounds, show, print, copy

if isdefined(@__MODULE__, :LanguageServer)
  include("src/BandColumnMatrices.jl")
  import .BandColumnMatrices:
    get_m_els,
    get_m,
    get_n,
    get_offset_all,
    get_upper_bw_max,
    get_upper_bw,
    get_middle_bw,
    get_lower_bw,
    get_first_super,
    get_band_element,
    extend_upper_band,
    extend_lower_band,
    set_band_element!,
    viewbc
  using .BandColumnMatrices
else
  import BandStruct.BandColumnMatrices:
    get_m_els,
    get_m,
    get_n,
    get_offset_all,
    get_upper_bw_max,
    get_upper_bw,
    get_middle_bw,
    get_lower_bw,
    get_first_super,
    get_band_element,
    extend_upper_band,
    extend_lower_band,
    set_band_element!,
    viewbc
  using BandStruct.BandColumnMatrices
end

export LeadingBandColumn,
  print_wilk,
  size,
  get_wilk,
  getindex,
  setindex!,
  leading_lower_ranks_to_bw,
  leading_upper_ranks_to_bw,
  leading_constrain_lower_ranks,
  leading_constrain_upper_ranks

#=
A banded matrix with structure defined by leading blocks and
stored in a compressed column-wise format.

Given

A = X | U | O   O   O | N | N
    _ ⌋   |           |   |  
    L   X | O   O   O | N | N
          |           |   |  
    L   X | U   U   U | O | O
    _ _ _ ⌋           |   |  
    O   L   X   X   X | O | O
    _ _ _ _ _ _ _ _ _ |   |  
    N   O   O   O   L | O | O
                      |   |  
    N   N   O   O   L | U | U
    _ _ _ _ _ _ _ _ _ ⌋ _ |  
    N   N   O   O   O   L   X

    N   N   N   N   N   L   X

where U, X, and L denote upper, middle, and lower band elements.  O
indicates an open location with storage available for expandning
either the lower or upper bandwidth.  N represents a location for
which there is no storage.

The matrix is stored as

elements = 

Num rows|Elements
----------------------
ubwmax  |O O O O O O O
        |O O O O O O O
        |O O O O O O O
        |O U U U U U U
----------------------
mbwmax+ |X X X X X L X
lbwmax  |L X O O L L X
        |L L O O L O O
        |O O O O O O O

where
    m =                  8
    n =                  7
    m_els =              8
    num_blocks =         6
    upper_bw_max =       4
    middle_bw_max =      2
    lower_bw_max =       2
    bws =                [ 1, 1, 1, 1, 1, 1, 1;  # upper
                           1, 2, 1, 1, 1, 0, 2;  # middle
                           2, 1, 0, 0, 2, 1, 0;  # lower
                           0, 1, 3, 3, 3, 6, 6 ] # first superdiagonal.
    leading_blocks =     [ 1, 3, 4, 6, 6, 8;     # rows
                           1, 2, 5, 5, 6, 7 ]    # columns
=#

struct LeadingBandColumn{
  E<:Number,
  AE<:AbstractArray{E,2},
  AI<:AbstractArray{Int,2},
} <: AbstractBandColumn{E,AE,AI}
  m::Int             # Matrix number of rows.
  n::Int             # Matrix number of columns.
  m_els::Int         # number of elements rows.
  num_blocks::Int    # Number of leading blocks.
  upper_bw_max::Int  # maximum upper bandwidth.
  middle_bw_max::Int # maximum middle bandwidth.
  lower_bw_max::Int  # maximum lower bandwidth.
  bws::AI            # 4xn matrix: upper, middle, and lower bw +
                     # first superdiagonal postion in A.
  leading_blocks::AI # 2xn matrix, leading block row and column counts.
  band_elements::AE
end

@as_record LeadingBandColumn

# Construct an empty (all zero) structure from the matrix size, bounds
# on the upper and lower bandwidth, and blocksizes.
function LeadingBandColumn(
  ::Type{E},
  m::Int,
  n::Int,
  upper_bw_max::Int,
  lower_bw_max::Int,
  leading_blocks::Array{Int,2},
) where {E}
  num_blocks = size(leading_blocks,2)
  bws = zeros(Int, 4, n)
  bws[2, 1] = leading_blocks[1,1]
  bws[4, 1] = 0
  block = 1
  for k = 2:n
    cols_in_block = leading_blocks[2,block]
    k <= cols_in_block || while (leading_blocks[2,block] == cols_in_block)
      block += 1
    end
    bws[2, k] = leading_blocks[1,block] - leading_blocks[1,block-1]
    bws[4, k] = leading_blocks[1,block-1]
  end
  middle_bw_max = maximum(bws[2, :])
  m_els = upper_bw_max + middle_bw_max + lower_bw_max
  band_elements = zeros(E, m_els, n)
  LeadingBandColumn(
    m,
    n,
    m_els,
    num_blocks,
    upper_bw_max,
    middle_bw_max,
    lower_bw_max,
    bws,
    leading_blocks,
    band_elements,
  )
end

function rand!(
  rng::AbstractRNG,
  lbc::LeadingBandColumn{E,AE,AI},
) where {E,AE,AI}

  for k = 1:(lbc.n)
    for j = first_storage_el(lbc, k):last_storage_el(lbc, k)
      lbc.band_elements[j, k] = rand(rng,E)
    end
  end

end

function LeadingBandColumn(
  rng::AbstractRNG,
  T::Type{E},
  m::Int,
  n::Int,
  upper_bw_max::Int,
  lower_bw_max::Int,
  leading_blocks::Array{Int,2},
  upper_ranks::Array{Int,1},
  lower_ranks::Array{Int,1},
) where {E}
  lbc = LeadingBandColumn(
    T,
    m,
    n,
    upper_bw_max,
    lower_bw_max,
    leading_blocks,
  )

  leading_lower_ranks_to_bw(lbc, lower_ranks)
  leading_upper_ranks_to_bw(lbc, upper_ranks)
  rand!(rng,lbc)
  lbc
end

##
## Functions implementing AbstractBandColumn.
##

@inline get_m_els(lbc::LeadingBandColumn) = lbc.m_els
@inline get_m(lbc::LeadingBandColumn) = lbc.m
@inline get_n(lbc::LeadingBandColumn) = lbc.n
@inline get_offset_all(lbc::LeadingBandColumn) = 0
@inline get_upper_bw_max(lbc::LeadingBandColumn) = lbc.upper_bw_max
@inline get_upper_bw(lbc::LeadingBandColumn, k::Int) = lbc.bws[1, k]
@inline get_middle_bw(lbc::LeadingBandColumn, k::Int) = lbc.bws[2, k]
@inline get_lower_bw(lbc::LeadingBandColumn, k::Int) = lbc.bws[3, k]
@inline get_first_super(lbc::LeadingBandColumn, k::Int) = lbc.bws[4, k]

@propagate_inbounds @inline get_band_element(
  lbc::LeadingBandColumn,
  j::Int,
  k::Int,
) = lbc.band_elements[j, k]

@propagate_inbounds @inline function set_band_element!(
  lbc::LeadingBandColumn{E,AE,AI},
  x::E,
  j::Int,
  k::Int,
) where {E,AE,AI}
  lbc.band_elements[j, k] = x
end

# Expand bandwidths to accommodate element j in column k
@inline function extend_upper_band(lbc::LeadingBandColumn, j::Int, k::Int)
  lbc.bws[1, k] = max(lbc.bws[1, k], lbc.bws[4, k] - j + 1)
end

@inline function extend_lower_band(lbc::LeadingBandColumn, j::Int, k::Int)
  lbc.bws[3, k] = max(lbc.bws[3, k], j - lbc.bws[4, k] - lbc.bws[2, k])
end

@inline size(lbc::LeadingBandColumn{T}) where {T} = (lbc.m, lbc.n)

function show(io::IO, lbc::LeadingBandColumn)
  print(
    io,
    typeof(lbc),
    "(",
    lbc.m,
    ", ",
    lbc.n,
    ", ",
    lbc.m_els,
    ", ",
    lbc.num_blocks,
    ", ",
    lbc.upper_bw_max,
    ", ",
    lbc.middle_bw_max,
    ", ",
    lbc.lower_bw_max,
    ", ",
    lbc.bws,
    ", ",
    lbc.leading_blocks,
    "): ",
  )
  for j = 1:lbc.m
    println()
    for k = 1:lbc.n
      if check_bc_storage_bounds(Bool, lbc, j, k)
        if bc_index_stored(lbc, j, k)
          @printf("%10.2e", lbc[j, k])
        else
          print("         O")
        end
      else
        print("         N")
      end
    end
  end
end

show(io::IO, ::MIME"text/plain", lbc::LeadingBandColumn) = show(io, lbc)

print(io::IO, lbc::LeadingBandColumn) = print(
  io,
  typeof(lbc),
  "(",
  lbc.m,
  ", ",
  lbc.n,
  ", ",
  lbc.m_els,
  ", ",
  lbc.num_blocks,
  ", ",
  lbc.upper_bw_max,
  ", ",
  lbc.middle_bw_max,
  ", ",
  lbc.lower_bw_max,
  ", ",
  lbc.bws,
  ", ",
  lbc.band_elements,
  ")",
)

print(io::IO, ::MIME"text/plain", lbc::LeadingBandColumn) = print(io, lbc)

##
## Index operations.  Scalar operations are defined for
## AbstractBandColumn matrices.
##

@inline function viewbc(
  lbc::LeadingBandColumn,
  i::Tuple{UnitRange{Int},UnitRange{Int}},
)
  (rows, cols) = i
  @when let LeadingBandColumn(;
      m,
      n,
      m_els,
      upper_bw_max = ubw_max,
      middle_bw_max = mbw_max,
      lower_bw_max = lbw_max,
      bws,
      band_elements = els,
    ) = lbc,
    UnitRange(j0, j1) = rows,
    UnitRange(k0, k1) = cols

    @boundscheck begin
      checkbounds(lbc, j0, k0)
      checkbounds(lbc, j1, k1)
    end
    BandColumn(
      max(j1 - j0 + 1, 0),
      max(k1 - k0 + 1, 0),
      m_els,
      1 - j0,
      ubw_max,
      mbw_max,
      lbw_max,
      view(bws, :, cols),
      view(els, :, cols),
    )
    @otherwise
    throw(MatchError)
  end
end

@inline function getindex(
  lbc::LeadingBandColumn,
  rows::UnitRange{Int},
  cols::UnitRange{Int},
)

  @when let LeadingBandColumn(;
      m,
      n,
      m_els,
      num_blocks,
      upper_bw_max = ubw_max,
      middle_bw_max = mbw_max,
      lower_bw_max = lbw_max,
      bws,
      band_elements = els,
    ) = lbc,
    UnitRange(j0, j1) = rows,
    UnitRange(k0, k1) = cols

    @boundscheck begin
      checkbounds(lbc, j0, k0)
      checkbounds(lbc, j1, k1)
    end
    BandColumn(
      max(j1 - j0 + 1, 0),
      max(k1 - k0 + 1, 0),
      m_els,
      1 - j0,
      ubw_max,
      mbw_max,
      lbw_max,
      bws[:, cols],
      els[:, cols],
    )
    @otherwise
    throw(MatchError)
  end
end

# Find bounds for lower left block l in A.
@inline function get_lower_block(
  lbc::LeadingBandColumn{T},
  l::Integer,
) where {T<:Number}
  (m, _) = size(lbc)
  j1 = lbc.leading_blocks[1, l] + 1
  k2 = lbc.leading_blocks[2, l]
  ((j1, m), (1, k2))
end

# Take lower and upper rank sequences and constrain them to be
# consistent with the size of the blocks (and preceding ranks).
function leading_constrain_lower_ranks(
  blocks::AbstractArray{Int,2},
  lower_ranks::AbstractArray{Int,1},
)
  lr = similar(lower_ranks)
  lr .= 0
  num_blocks = size(blocks, 2)
  m = blocks[1,num_blocks]

  lr[1] = min(m - blocks[1, 1], blocks[2, 1], lower_ranks[1])

  for l = 2:(num_blocks - 1)
    j0 = blocks[1, l - 1] + 1
    k0 = blocks[2, l - 1]
    j1 = blocks[1, l] + 1
    k1 = blocks[2, l]
    m1 = m - j1 + 1
    n1 = k1 - k0 + lr[l - 1]
    lr[l] = min(m1, n1, lower_ranks[l])
  end
  lr

end

function leading_constrain_upper_ranks(
  blocks::AbstractArray{Int,2},
  upper_ranks::AbstractArray{Int,1},
)

  ur = similar(upper_ranks)
  ur .= 0
  num_blocks = size(blocks, 2)
  n = blocks[2,num_blocks]

  ur[1] = min(blocks[1, 1], n - blocks[2, 1], upper_ranks[1])

  for l = 2:(num_blocks - 1)
    j0 = blocks[1, l - 1]
    k0 = blocks[2, l - 1] + 1
    j1 = blocks[1, l]
    k1 = blocks[2, l] + 1
    n1 = n - k1 + 1
    m1 = j1 - j0 + ur[l - 1]
    ur[l] = min(m1, n1, upper_ranks[l])
  end
  ur

end

# Find bounds for upper right block l in A.
@inline function get_upper_block(lbc::LeadingBandColumn, l::Integer)
  (_, n) = size(lbc)
  j2 = lbc.leading_blocks[1, l]
  k1 = lbc.leading_blocks[2, l] + 1
  ((1, j2), (k1, n))
end

# Set lower bandwidth appropriate for a given lower rank sequence.
function leading_lower_ranks_to_bw(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  lbc.bws[3, :] .= 0
  rs1 = leading_constrain_lower_ranks(lbc.leading_blocks, rs)

  for l = 1:(lbc.num_blocks - 1)
    ((j0, _), (_, k0)) = get_lower_block(lbc, l)
    ((j1, _), (_, k1)) = get_lower_block(lbc, l + 1)
    d = j1 - j0
    lbc.bws[3, (k0 - rs1[l] + 1):k0] .+= d
  end
end

# Set upper bandwidth appropriate for a given lower rank sequence.
function leading_upper_ranks_to_bw(
  lbc::LeadingBandColumn,
  rs::AbstractArray{Int},
)

  (m, n) = size(lbc)
  lbc.bws[1, :] .= 0
  rs1 = leading_constrain_upper_ranks(lbc.leading_blocks, rs)

  for l = 1:(lbc.num_blocks - 1)
    (_, (k0, _)) = get_upper_block(lbc, l)
    (_, (k1, _)) = get_upper_block(lbc, l + 1)
    lbc.bws[1, k0:(k1 - 1)] .= rs1[l]
  end
end


# Functions for printing out the matrix structure associated with
# a LeadingBandColumn struct.
function print_wilk(a::AbstractArray{Char,2})
  (m, n) = size(a)
  for j = 1:m
    println()
    for k = 1:n
      print(a[j, k], " ")
    end
  end
end

function print_wilk(lbc::LeadingBandColumn)
  print_wilk(get_wilk(lbc))
end

@views function get_wilk(lbc::LeadingBandColumn)
  (m, n) = size(lbc)
  num_blocks = lbc.num_blocks - 1 # leave off the full matrix.
  a = fill('N', (2 * m, 2 * n))
  # insert spaces
  fill!(a[2:2:(2 * m), :], ' ')
  fill!(a[:, 2:2:(2 * n)], ' ')
  # insert boundaries for leading blocks.
  for j = 1:num_blocks
    row = lbc.leading_blocks[1, j]
    col = lbc.leading_blocks[2, j]
    fill!(a[1:(2 * row - 1), 2 * col], '|')
    a[2 * row, 2 * col] = '⌋'
    fill!(a[2 * row, 1:(2 * col - 1)], '_')
  end
  for k = 1:n
    j = lbc.bws[4, k] + 1

    fill!(
      a[
        max(1, 2 * (j - lbc.upper_bw_max) - 1):2:min(
          2 * m,
          2 * (j + lbc.m_els - lbc.upper_bw_max - 1) - 1,
        ),
        2 * k - 1,
      ],
      'O',
    )
    fill!(a[(2 * (j - lbc.bws[1, k]) - 1):2:(2 * (j - 1) - 1), 2 * k - 1], 'U')
    fill!(a[(2 * j - 1):2:(2 * (j + lbc.bws[2, k] - 1) - 1), 2 * k - 1], 'X')
    fill!(
      a[
        (2 * (j + lbc.bws[2, k]) - 1):2:(2 * (j + lbc.bws[2, k] + lbc.bws[
          3,
          k,
        ] - 1) - 1),
        2 * k - 1,
      ],
      'L',
    )
  end
  a::Array{Char,2}
end

function copy(lbc::LeadingBandColumn)
  @when let LeadingBandColumn(
      m,
      n,
      m_els,
      num_blocks,
      ubw_max,
      mbw_max,
      lbw_max,
      bws,
      lblocks,
      bels,
    ) = lbc
    LeadingBandColumn(
      m,
      n,
      m_els,
      num_blocks,
      ubw_max,
      mbw_max,
      lbw_max,
      bws,
      lblocks,
      bels,
    )
    @otherwise
    throw(MatchError)
  end
end


end