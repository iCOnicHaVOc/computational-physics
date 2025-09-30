import matplotlib.pyplot as plt

l = []   # will be reused for each c value
oi = []  # not really needed for bifurcation, but kept

def random_seq(x, c, iterations):
    """Generate logistic-map sequence for given x0, c."""
    l.clear()
    oi.clear()
    for i in range(iterations):
        oi.append(i)
        x = c * x * (1 - x)
        l.append(x)
    return l

def bifurcation_from_random(x0=0.1, c_min=2.5, c_max=4.0, 
                            c_steps=1000, iterations=1000, last=100):
    c_values = []
    x_values = []

    c_range = [c_min + (c_max - c_min) * i / c_steps for i in range(c_steps)]
    
    for c in c_range:
        seq = random_seq(x0, c, iterations)
        # keep only the last `last` points to remove transient behavior
        for x in seq[-last:]:
            c_values.append(c)
            x_values.append(x)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(c_values, x_values, s=0.1, color='black')
    plt.title("Bifurcation Diagram of Logistic Map")
    plt.xlabel("c")
    plt.ylabel("x")
    plt.show()

# Run it
bifurcation_from_random()

def LCG(p, q, r, k, n):
    out = []
    for _ in range(n):
        k = (p * k + q) % r
        out.append(k / r)      # uniform in [0,1)
    return out

def estimate_pi_lcg(n_points, params1, params2):
    # params = (p, q, r, seed)
    x = LCG(*params1, n_points)
    y = LCG(*params2, n_points)   # use a DIFFERENT seed (or params) than x
    inside = sum((xi*xi + yi*yi) <= 1.0 for xi, yi in zip(x, y))
    return 4.0 * inside / n_points

# Example params (choose good LCG params for quality):
# Here’s a common set: p=1664525, q=1013904223, r=2**32
p, q, r = 1664525, 1013904223, 2**32
pi_est = estimate_pi_lcg(
    1_000_00,                 # number of points
    (p, q, r, 12345),         # x-sequence seed
    (p, q, r, 67890),         # y-sequence seed (different!)
)
print(pi_est)