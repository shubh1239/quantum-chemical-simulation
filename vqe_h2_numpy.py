"""
VQE Prototype — H2 Ground State Energy (NumPy/SciPy)
An open-source verification of the H2 ground state utilizing an analytical 
expectation value framework alongside a classical matrix optimization loop.
"""

import numpy as np
from scipy.optimize import minimize

# 1. Pauli matrices 
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(a, b):
    """2-qubit operator from two single-qubit operators."""
    return np.kron(a, b)

# 2. Build the H2 Hamiltonian as a 4x4 matrix
#    H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*Y0Y1 + g5*X0X1
g0, g1, g2, g3, g4, g5 = -0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910

H = (
    g0 * kron(I2, I2)
    + g1 * kron(Z, I2)
    + g2 * kron(I2, Z)
    + g3 * kron(Z, Z)
    + g4 * kron(Y, Y)
    + g5 * kron(X, X)
)

print("H2 Hamiltonian (4x4, basis order |00>,|01>,|10>,|11>):")
print(np.round(H.real, 4))
print()

# 3. Ansatz: |psi(theta)> = cos(theta/2)|01> - sin(theta/2)|10>
ket_01 = np.array([0, 1, 0, 0], dtype=complex)
ket_10 = np.array([0, 0, 1, 0], dtype=complex)


def ansatz(theta):
    return np.cos(theta / 2) * ket_01 - np.sin(theta / 2) * ket_10


def energy_numeric(theta):
    """E(theta) = <psi|H|psi>, computed by direct matrix multiplication."""
    psi = ansatz(theta)
    return np.real(np.conj(psi) @ H @ psi)


def energy_analytic(theta):
    """Closed-form formula derived analytically."""
    return (g0 - g3) + (g1 - g2) * np.cos(theta) - (g4 + g5) * np.sin(theta)


# Sanity check: do the numeric and hand-derived analytic formulas agree?
test_thetas = [0, 0.5, 1.0, 2.0, 2.911, np.pi]
print("Sanity check — numeric vs. hand-derived analytic formula:")
for t in test_thetas:
    en, ea = energy_numeric(t), energy_analytic(t)
    print(f"  theta={t:6.3f} rad | numeric={en:9.5f} | analytic={ea:9.5f} | match={np.isclose(en, ea)}")
print()

# 4. Classical optimizer loop — the "VQE" part.
best_result = None
for theta0 in np.linspace(0, 2 * np.pi, 8):
    result = minimize(energy_numeric, x0=[theta0], method="COBYLA")
    if best_result is None or result.fun < best_result.fun:
        best_result = result

theta_opt = best_result.x[0]
energy_opt = best_result.fun

print("VQE Result ")
print(f"Optimal theta: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f} deg)")
print(f"Ground state energy (VQE, optimizer): {energy_opt:.6f} Hartree")

# 5. Cross-check against exact diagonalization.
eigenvalues, _ = np.linalg.eigh(H)
exact_ground_energy = eigenvalues[0]

print(f"Exact ground state energy (direct diagonalization): {exact_ground_energy:.6f} Hartree")
print(f"Difference (VQE vs exact): {abs(energy_opt - exact_ground_energy):.2e} Hartree")

CHEMICAL_ACCURACY_HARTREE = 0.0016
within_chem_accuracy = abs(energy_opt - exact_ground_energy) < CHEMICAL_ACCURACY_HARTREE
print(f"Within chemical accuracy (<{CHEMICAL_ACCURACY_HARTREE} Ha)? {within_chem_accuracy}")
print(f"\nFinal Verification: E = {energy_opt:.6f} Ha vs Exact = {exact_ground_energy:.6f} Ha")