import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

N = int(1e06)
rng = np.random.default_rng(seed=42)

s = rng.uniform(size=N)

m = s.mean()
v = s.var()
z = (m - 0.5) / np.sqrt(v / N)
if z <= 1.96 and z >= -1.96:
    print(f"z={z:.2f}, OK?->YES")
else:
    print(f"z={z:.2f}, OK?->NO")

chi2 = (N - 1) * v**2 / (1 / 12) ** 2
if chi2 <= 1000840 and chi2 >= 999160:
    print(f"chi2={chi2:.0f}, OK?->YES")
else:
    print(f"chi2={chi2:.0f}, OK?->NO")


k_values = [1, 10, 100, 1000, 10000, 10000, 100000, 500000]
k_values = np.linspace(1, N+1, 1000, dtype=int)

# Theoretical values
theoretical_correlation = 1 / 4
theoretical_variance = 7 / 144

# Compute two-point correlations and z-scores
nseed = 6
seeds = rng.integers(1, 100000, size=nseed)
for i in range(nseed):
    rng = np.random.default_rng(seed=seeds[i])
    s = rng.uniform(size=N)
    results = []
    ok = np.zeros(len(k_values))
    for j, k in np.ndenumerate(k_values):
        # Compute C_k
        C_k = np.mean(s[:-k] * s[k:])

        # Compute variance of C_k
        sample_variance = theoretical_variance / (N - k)

        # Compute z-score
        z_score = (C_k - theoretical_correlation) / np.sqrt(sample_variance)
        if z >= 1.96 or z <= -1.96:
            ok[j] = 1

        # Append results
        results.append((k, C_k, z_score))
    k_vals, C_ks, _ = zip(*results)
    plt.plot(k_vals, C_ks, label=f"seed={seeds[i]}, n_z outside={ok.sum()}")

# Display results
# print(f"{'k':<10} {'C_k':<10} {'z-score':<10} {'OK?': <5}")
# for k, C_k, z_score in results:
#    if z <= 1.96 and z >= -1.96:
#        ok = "YES"
#    else:
#        ok = "NO"
#
#    print(f"{k:<10} {C_k:<10.6f} {z_score:<10.3f} {ok:<5}")

# Plot the correlations
plt.axhline(
    theoretical_correlation, color="r", linestyle="--", label="Theoretical Value (1/4)"
)
plt.xlabel("Lag (k)")
plt.ylabel("Two-Point Correlation C_k")
plt.title("Two-Point Correlation vs Lag")
plt.legend()
plt.grid()
# plt.xscale("log")
plt.show()
