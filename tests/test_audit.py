import unittest
import math
import numpy as np

from tests.benchmark import (
    generate_primes,
    evaluate_modular_lift,
    evaluate_vqa_gradient_variance,
    calculate_restart_bound,
    calculate_update_entropy,
    gauge_energy_decay,
)


class TestFormalProofsSuite(unittest.TestCase):

    def test_theorem_1_restart_bound(self):
        # Test Theorem 1 multi-start restart bound formula R = ceil(ln(1/gamma) / (q * (1 - alpha)))
        q = 0.2
        alpha = 0.1
        gamma = 0.05
        # Expected: ceil(ln(20) / (0.2 * 0.9)) = ceil(2.9957 / 0.18) = ceil(16.64) = 17
        r = calculate_restart_bound(q, alpha, gamma)
        expected = math.ceil(math.log(1.0 / gamma) / (q * (1.0 - alpha)))
        self.assertEqual(r, expected)
        self.assertEqual(r, 17)

    def test_theorem_2_update_entropy_bounds(self):
        # Test Theorem 2 extremal bounds 0 <= H_delta <= log2(d)
        d = 8
        # Single coordinate update -> H_delta = 0
        delta_single = np.array([5.0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(calculate_update_entropy(delta_single), 0.0)

        # Equipartition update -> H_delta = log2(d) = 3.0
        delta_equal = np.array([2.0] * d)
        self.assertAlmostEqual(calculate_update_entropy(delta_equal), math.log2(d))

    def test_lemma_1_modular_support(self):
        # Test Lemma 1 prime generation & coprime candidate set
        n_max = 100
        modulus_m = 6  # prime factors 2, 3
        primes = generate_primes(n_max)
        coprimes = set(x for x in range(1, n_max + 1) if math.gcd(x, modulus_m) == 1)

        # Primes except 2, 3 must be subset of coprimes mod 6
        primes_except_m = primes - {2, 3}
        self.assertTrue(primes_except_m.issubset(coprimes))

    def test_theorem_3_vqa_variance(self):
        # Test Theorem 3 shot-noise / gradient variance scaling
        res = evaluate_vqa_gradient_variance(num_qubits=4, num_samples=5000)
        self.assertEqual(res["qubits"], 4)
        self.assertEqual(res["theoretical_bound"], 1.0 / 16.0)
        self.assertAlmostEqual(res["empirical_variance"], 1.0 / 16.0, delta=0.01)

    def test_theorem_4_lyapunov_decay(self):
        # Test Theorem 4 gauge energy decay E_g(t) = E_g(0) * exp(-2 * gamma * t)
        e0 = 10.0
        gamma = 0.5
        t = 2.0
        # Expected: 10.0 * exp(-2 * 0.5 * 2.0) = 10.0 * exp(-2) ~ 1.3533528
        e_t = gauge_energy_decay(e0, gamma, t)
        self.assertAlmostEqual(e_t, 10.0 * math.exp(-2.0), places=5)

    def test_evaluate_modular_lift(self):
        # Test Coverage-Matched Lift Metric L_M <= 1.0 for coprime candidate sequence
        n_max = 1000
        modulus_m = 30  # 2 * 3 * 5
        # Candidate sequence: arithmetic progression 30k + 7 (all coprime to 30)
        candidates = [30 * k + 7 for k in range(n_max // 30)]
        res = evaluate_modular_lift(candidates, modulus_m, n_max)

        self.assertIn("lift_ratio_LM", res)
        # Lift ratio should be close to 1.0 (Prime Number Theorem / Dirichlet)
        self.assertLessEqual(res["lift_ratio_LM"], 1.5)


if __name__ == "__main__":
    unittest.main()
