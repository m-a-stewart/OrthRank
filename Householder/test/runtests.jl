if isdefined(@__MODULE__, :LanguageServer)
  include("src/Compute.jl")
  using .Compute
else
  using Householder.Compute
end

using LinearAlgebra
using Random
using InPlace
using ShowTests

tol = 1e-14
l=2
m=3

print("""
Real Left Multiplcation Tests

""")

# Real Left Multiplication

a=rand(Complex{Float64},m,m) .- 0.5
h = lhouseholder(copy(a[:,1]),l,0)

a0 = copy(a)
h ⊘ a
column_nonzero!(a,l,1)

show_error_result(
  "Singular value errors for left multiplication, Real",
  norm(svdvals(a) - svdvals(a0)),
  tol,
)

show_equality_result(
  "Exact zero test for column_nonzero!, Real",
  0.0,
  norm(a[1:(l - 1), 1]) + norm(a[(l + 1):m, 1]),
)

h ⊛ a
show_error_result(
  "Left inverse error test, Real",
  norm(a-a0),
  tol,
)

# Real Right Multiplication
print("""

Real Right Multiplcation Tests

""")

a=rand(Float64,m,m) .- 0.5
h = rhouseholder(copy(a[1,:]),l,0)

a0 = copy(a)
a ⊛ h
row_nonzero!(a,1,l)

show_error_result(
  "Singular value errors for right multiplication, Real",
  norm(svdvals(a) - svdvals(a0)),
  tol,
)

show_equality_result(
  "Exact zero test for row_nonzero!, Real",
  0.0,
  norm(a[1,1:(l - 1)]) + norm(a[1, (l + 1):m]),
)

a ⊘ h
show_error_result(
  "Right inverse error test, Real",
  norm(a-a0),
  tol,
)

print("""

Complex Left Multiplcation Tests

""")

a=rand(Complex{Float64},m,m) .- (0.5+0.5im)
h = lhouseholder(copy(a[:,1]),l,0)

a0 = copy(a)
h ⊘ a
column_nonzero!(a,l,1)

show_error_result(
  "Singular value errors for left multiplication, Complex",
  norm(svdvals(a) - svdvals(a0)),
  tol,
)

show_equality_result(
  "Exact zero test for column_nonzero!, Complex",
  0.0,
  norm(a[1:(l - 1), 1]) + norm(a[(l + 1):m, 1]),
)

h ⊛ a
show_error_result(
  "Left inverse error test, Complex",
  norm(a-a0),
  tol,
)

print("""

Complex Right Multiplcation Tests

""")

a=rand(Complex{Float64},m,m) .- (0.5+0.5im)
h = rhouseholder(copy(a[1,:]),l,0)

a0 = copy(a)
a ⊛ h
row_nonzero!(a,1,l)
show_error_result(
  "Singular value errors for right multiplication, Real",
  norm(svdvals(a) - svdvals(a0)),
  tol,
)

show_equality_result(
  "Exact zero test for row_nonzero!, Real",
  0.0,
  norm(a[1,1:(l - 1)]) + norm(a[1, (l + 1):m]),
)

a ⊘ h
show_error_result(
  "Right inverse error test, Real",
  norm(a-a0),
  tol,
)