import numpy as np
import matplotlib.pyplot as plt

def generate_random_unit_vectors(n, d):
    """Generate n random unit vectors in R^d."""
    vectors = np.random.randn(n, d)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    return unit_vectors

def compute_pairwise_dot_products(vectors):
    """Compute all pairwise dot products of the vectors."""
    n = vectors.shape[0]
    dot_products = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dot_products[i, j] = np.dot(vectors[i], vectors[j])
            dot_products[j, i] = dot_products[i, j]
    return dot_products

def expected_dot_product_statistics(n, d, num_trials=100):
    """Compute the expected dot product and its variance over multiple trials."""
    expected_dps = []
    for _ in range(num_trials):
        vectors = generate_random_unit_vectors(n, d)
        dot_products = compute_pairwise_dot_products(vectors)
        # We only consider the upper triangle to avoid double-counting and self-dots
        upper_triangle = dot_products[np.triu_indices(n, k=1)]
        expected_dps.append(np.mean(upper_triangle))
    return np.abs(np.mean(expected_dps)), np.var(expected_dps)

# Parameters
n = 1000  # Number of vectors
d_values = [10, 50, 100, 500, 1000]  # Different dimensions
num_trials = 25

# Compute statistics for each dimension
results = {}
for d in d_values:
    mean_dp, var_dp = expected_dot_product_statistics(n, d, num_trials)
    results[d] = (mean_dp, var_dp)
    print(f"Dimension {d}: Mean Dot Product = {mean_dp}, Variance = {var_dp}")

# Plotting
dims = list(results.keys())
means = [results[d][0] for d in dims]
variances = [results[d][1] for d in dims]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(dims, means, marker='o')
plt.xlabel('Dimension (d)')
plt.ylabel('Absolute Value of Mean Dot Product')
plt.title('Mean Dot Product vs Dimension')

plt.subplot(1, 2, 2)
plt.plot(dims, variances, marker='o', color='orange')
plt.xlabel('Dimension (d)')
plt.ylabel('Variance of Dot Product')
plt.title('Variance of Dot Product vs Dimension')

plt.savefig("simfig.png")
plt.tight_layout()
plt.show()