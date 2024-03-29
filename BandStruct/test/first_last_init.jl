# Test correct bandwidths for conversion between leading and trailing.
#
# Zero rank structure

# X X X X|   |     |   |   | 
#        +---+-----+---+---+-
# X X X X X X|     |   |   | 
# ---+       |     |   |   | 
#    |X X X X|     |   |   | 
#    |       +-----+---+---+-
#    |X X X X X X X|   |   | 
#    |             +---+---+-
#    |X X X X X X X X X|   | 
# ---+-----+           |   | 
#    |     |X X X X X X|   | 
# ---+-----+-+         +---+-
#    |     | |X X X X X X X| 
#    |     | |             | 
#    |     | |X X X X X X X| 
# ---+-----+-+-------+     +-
#    |     | |       |X X X X
# ---+-----+-+-------+---+   
#    |     | |       |   |X X
#    |     | |       |   |   
#    |     | |       |   |X X


# Rank 2 Trailing Structure:

# X X X X|U U|     |   |   | 
#        +---+-----+---+---+-
# X X X X X X|U U  |   |   | 
# ---+       |     |   |   | 
# L L|X X X X|U U  |   |   | 
#    |       +-----+---+---+-
# L L|X X X X X X X|U U|   | 
#    |             +---+---+-
#    |X X X X X X X X X|U U| 
# ---+-----+           |   | 
#    |L L L|X X X X X X|U U| 
# ---+-----+-+         +---+-
#    |L L L|L|X X X X X X X|U
#    |     | |             | 
#    |     |L|X X X X X X X|U
# ---+-----+-+-------+     +-
#    |     | |L L L L|X X X X
# ---+-----+-+-------+---+   
#    |     | |L L L L|L L|X X
#    |     | |       |L L|   
#    |     | |       |   |X X


# Trailing to Leading Conversion Fill-in:

# X X X X|U U|O O  |   |   | 
#        +---+-----+---+---+-
# X X X X X X|U U O|O O|   | 
# ---+       |     |   |   | 
# L L|X X X X|U U O|O O|O O| 
#    |       +-----+---+---+-
# L L|X X X X X X X|U U|O O| 
#    |             +---+---+-
# O O|X X X X X X X X X|U U|O
# ---+-----+           |   | 
# O O|L L L|X X X X X X|U U|O
# ---+-----+-+         +---+-
# O O|L L L|L|X X X X X X X|U
#    |     | |             | 
#    |  O O|L|X X X X X X X|U
# ---+-----+-+-------+     +-
#    |    O|O|L L L L|X X X X
# ---+-----+-+-------+---+   
#    |    O|O|L L L L|L L|X X
#    |     | |       |   |   
#    |     | |    O O|L L|X X

# Rank 2 Leading Structure:

# X X X X|U U|     |   |   | 
#        +---+-----+---+---+-
# X X X X X X|U U U|   |   | 
# ---+       |     |   |   | 
# L L|X X X X|U U U|U U|   | 
#    |       +-----+---+---+-
# L L|X X X X X X X|U U|   | 
#    |             +---+---+-
# L L|X X X X X X X X X|U U| 
# ---+-----+           |   | 
#    |  L L|X X X X X X|U U| 
# ---+-----+-+         +---+-
#    |    L|L|X X X X X X X|U
#    |     | |             | 
#    |    L L|X X X X X X X|U
# ---+-----+-+-------+     +-
#    |     | |    L L|X X X X
# ---+-----+-+-------+---+   
#    |     | |       |L L|X X
#    |     | |       |   |   
#    |     | |       |L L|X X

# Rank 2 Leading to Trailing Fill-in:

# X X X X|U U|O O  |   |   | 
#        +---+-----+---+---+-
# X X X X X X|U U U|O O|   | 
# ---+       |     |   |   | 
# L L|X X X X|U U U|U U|O O| 
#    |       +-----+---+---+-
# L L|X X X X X X X|U U|O O| 
#    |             +---+---+-
# L L|X X X X X X X X X|U U|O
# ---+-----+           |   | 
# O O|O L L|X X X X X X|U U|O
# ---+-----+-+         +---+-
# O O|O O L|L|X X X X X X X|U
#    |     | |             | 
#    |  O L L|X X X X X X X|U
# ---+-----+-+-------+     +-
#    |    O|O|O O L L|X X X X
# ---+-----+-+-------+---+   
#    |    O|O|O O O O|L L|X X
#    |     | |       |   |   
#    |     | |    O O|L L|X X


@safetestset "BandStruct First Last Init" begin
  using BandStruct

  @testset "cols_first_last test." begin

    lbl = transpose([ 2 2
                      5 5
                      6 6
                      8 10
                      9 12])

    ubl = transpose([ 1 4
                      3 6
                      4 9
                      6 11
                      8 13])

    lower_blocks = IndexList([BlockSize(lbl[1,j], lbl[2,j]) for j∈1:5], max_length = 5)
    upper_blocks = IndexList([BlockSize(ubl[1,j], ubl[2,j]) for j∈1:5], max_length = 5)


    cols_first_last = [ 1  1  1  1   1   1   1   1   2   2   2   3   3   5
                        1  1  1  1   2   2   4   4   4   5   5   7   7   9
                        0  0  0  0   1   1   3   3   3   4   4   6   6   8
                        3  3  6  6   6   7   9   9   9   9  10  10  12  12
                        2  2  5  5   5   6   8   8   8   8   9   9  11  11
                        7  7  7  8  10  10  10  10  11  11  11  11  11  11 ]

    @test get_cols_first_last(11, 14, upper_blocks, lower_blocks, 2, 2) ==
      cols_first_last

    rows_first_last = [ 1   1   0   5   4   8
                        1   1   0   7   6  11
                        1   3   2   7   6  13
                        1   3   2  10   9  13
                        1   3   2  12  11  14
                        1   6   5  12  11  14
                        1   7   6  14  13  14
                        4   7   6  14  13  14
                        5  11  10  15  14  14
                        5  13  12  15  14  14
                        9  13  12  15  14  14 ]

    @test get_rows_first_last(11, 14, upper_blocks, lower_blocks, 2, 2) ==
      rows_first_last
  end

  @testset "Small zero rank case." begin

    # Small example with zero ranks.

    lbl = transpose([ 1 1
                      2 2
                      3 4 ])

    ubl = transpose([ 1 1
                      2 2
                      2 3 ])
    
    m=6
    n=5

    lower_blocks = IndexList([BlockSize(lbl[1,j], lbl[2,j]) for j∈1:size(lbl,2)])
    upper_blocks = IndexList([BlockSize(ubl[1,j], ubl[2,j]) for j∈1:size(ubl,2)])

    cols_first_last = [ 1 2 3 3 3
                        1 2 3 3 3
                        0 1 2 2 2
                        2 3 4 4 7
                        1 2 3 3 6
                        1 2 3 3 6 ]

    @test get_cols_first_last(m, n, upper_blocks, lower_blocks, 0, 0) == cols_first_last

    rows_first_last = [ 1 1 0 2 1 1
                        2 2 1 3 2 2
                        3 3 2 6 5 5
                        5 5 4 6 5 5
                        5 5 4 6 5 5
                        5 5 4 6 5 5 ]


    @test get_rows_first_last(m, n, upper_blocks, lower_blocks, 0, 0) == rows_first_last

  end

  ToListIndex(i::Int) = ListIndex(i)
  ToListIndex(x) = x

  @testset "Forward tracking of intersecting blocks by column." begin

    lbl = transpose([ 1 1
                      2 2
                      3 4
                      6 5])

    ubl = transpose([ 0 0
                      1 1
                      2 2
                      2 3 ])

    lower_blocks = IndexList([BlockSize(lbl[1,j], lbl[2,j]) for j∈1:size(lbl,2)])
    upper_blocks = IndexList([BlockSize(ubl[1,j], ubl[2,j]) for j∈1:size(ubl,2)])

    ubs = Union{ListIndex, Before, After}[]
    ub = Before()
    lbs = Union{ListIndex, Before, After}[]
    lb = Before()
    m = 6
    n = 5
    for k ∈ 1:n
      ub = forward_last_trailing_including_index(upper_blocks, k, 2, m, n, ub)
      lb = forward_first_leading_including_index(lower_blocks, k, 2, m, n, lb)
      push!(ubs, ub)
      push!(lbs, lb)
    end
    a = [ (ubs[k], lbs[k])[j] for j in 1:2, k in 1:n ]
    @test a == ListIndex.([ 1 2 3 4 4 ;
                            1 2 3 3 4 ])
  end

  @testset "Backward tracking of intersecting blocks by column." begin

    lbl = transpose([ 1 1
                      2 2
                      3 4 ])

    ubl = transpose([ 1 1
                      2 2
                      2 3 ])

    lower_blocks = IndexList([BlockSize(lbl[1,j], lbl[2,j]) for j∈1:size(lbl,2)])
    upper_blocks = IndexList([BlockSize(ubl[1,j], ubl[2,j]) for j∈1:size(ubl,2)])

    ubs = Union{ListIndex, Before, After}[]
    ub = After()
    lbs = Union{ListIndex, Before, After}[]
    lb = After()
    m = 6
    n = 5
    for k ∈ n:-1:1
      ub = backward_last_trailing_including_index(upper_blocks, k, 2, m, n, ub)
      lb = backward_first_leading_including_index(lower_blocks, k, 2, m, n, lb)
      pushfirst!(ubs, ub)
      pushfirst!(lbs, lb)
    end
    a = [ (ubs[k], lbs[k])[j] for j in 1:2, k in 1:n ]
    @test a == ToListIndex.([ Before() 1 2 3 3
                              1 2 3 3 After()
                              ])
  end

end
nothing
