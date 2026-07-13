"""
VQE Prototype — H2 Ground State Energy (Qiskit Port)
Translates the analytical matrix formulation into a parameterized quantum circuit
infrastructure interacting with native Qiskit primitives.
"""

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

# 1. Hamiltonian as a SparsePauliOp
g0, g1, g2, g3, g4, g5 = -0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910

hamiltonian = SparsePauliOp.from_list([
    ("II", g0),
    ("IZ", g1),
    ("ZI", g2),
    ("ZZ", g3),
    ("YY", g4),
    ("XX", g5),
])

print("Hamiltonian (SparsePauliOp):")
print(hamiltonian)
print()

# 2. Ansatz circuit using a particle-conserving Givens rotation layout
def build_ansatz(theta):
    qc = QuantumCircuit(2)
    qc.x(1) 
    qc.cx(1, 0)
    qc.cry(theta, 0, 1)
    qc.cx(1, 0)
    return qc


print("Example ansatz circuit (theta = 1.0 rad):")
print(build_ansatz(1.0).draw(output="text"))
print()

# 3. Estimator Primitive initialization
estimator = StatevectorEstimator()


def energy_qiskit(theta):
    val = theta[0] if hasattr(theta, "__len__") else theta
    qc = build_ansatz(val)
    job = estimator.run([(qc, hamiltonian)])
    result = job.result()
    return float(result[0].data.evs)


def energy_analytic(theta):
    return (g0 - g3) + (g1 - g2) * np.cos(theta) - (g4 + g5) * np.sin(theta)


print("Sanity check — Qiskit circuit energy vs. hand-derived analytic formula:")
for t in [0, 0.5, 1.0, 2.0, 2.911, np.pi]:
    eq, ea = energy_qiskit(t), energy_analytic(t)
    print(f"  theta={t:6.3f} rad | qiskit={eq:9.5f} | analytic={ea:9.5f} | match={np.isclose(eq, ea)}")
print()

# 4. Classical optimization tracking loop
best_result = None
for theta0 in np.linspace(0, 2 * np.pi, 8):
    result = minimize(energy_qiskit, x0=[theta0], method="COBYLA")
    if best_result is None or result.fun < best_result.fun:
        best_result = result

theta_opt = best_result.x[0]
energy_opt = best_result.fun

print("VQE Result (Qiskit circuit + StatevectorEstimator) ")
print(f"Optimal theta: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f} deg)")
print(f"Ground state energy: {energy_opt:.6f} Hartree")

# 5. Verification validation
H_matrix = hamiltonian.to_matrix()
eigenvalues = np.linalg.eigvalsh(H_matrix)
print(f"Exact ground state energy (diagonalization): {eigenvalues[0]:.6f} Hartree")
print(f"Difference vs Exact: {abs(energy_opt - eigenvalues[0]):.2e} Hartree")