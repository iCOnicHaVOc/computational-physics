'''from Mylib import MyComplex

# Create two complex numbers
c1 = MyComplex(0.75, 1.25)
c2 = MyComplex(-1.5, -2.0)
c3 = MyComplex(0.0, 0.0)

print("\nTwo complex")
c1.display_cmplx()
c2.display_cmplx()

print("\nAdding two complex")
c3.add_cmplx(c1, c2)
c3.display_cmplx()

print("\nSubtracting two complex")
c3.sub_cmplx(c1, c2)
c3.display_cmplx()

print("\nMultiplying two complex")
c3.mul_cmplx(c1, c2)
c3.display_cmplx()

print("\nModulus of two complex")
mod = c1.mod_cmplx()
print(f"c1 mod = {mod:.3f}")
mod = c2.mod_cmplx()
print(f"c2 mod = {mod:.3f}")'''


# class point:
#     '''represent point in 2D space'''
# blank = point()
# blank.x=3
# blank.y=4

# x= blank.x
# print(x)

import matplotlib.pyplot as plt

# Initial seed between 0 and 1
x = 0.5  

# Logistic map parameter (close to 4 for chaos)
c = 3.99  

# Logistic Map PRNG function
def logistic_prng():
    global x
    x = c * x * (1 - x)
    return x

# Generate 100 values
logistic_numbers = [logistic_prng() for _ in range(100)]

# Plot the sequence
plt.figure(figsize=(8, 4))
plt.plot(range(1, 101), logistic_numbers, marker='o', linestyle='-', markersize=4)
plt.title("Pseudo-Random Numbers (Logistic Map)")
plt.xlabel("Index")
plt.ylabel("Random Value")
plt.grid(True)
plt.show()

#LCG
# Pseudo-Random Number Generator (LCG method)

# Seed value (can be any integer > 0)
seed = 123456789  

def prng():
    global seed
    # LCG parameters
    a = 1664525        # multiplier
    c = 1013904223     # increment
    m = 2**32          # modulus
    seed = (a * seed + c) % m
    return seed / m    # scale to [0, 1)

# Generate 10 pseudo-random numbers
for _ in range(10):
    print(prng())
