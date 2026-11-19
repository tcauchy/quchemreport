import unittest
import numpy as np

from quchemreport.data_enrichment.nto import build_nto_pairs


class NTOTests(unittest.TestCase):
    def test_two_independent_transitions_are_two_pairs(self):
        transitions = [((0, 0), (2, 0), .8), ((1, 0), (3, 0), .6)]
        pairs, norm = build_nto_pairs(transitions, [np.eye(4)], [1], min_weight=0)
        np.testing.assert_allclose([p["weight"] for p in pairs], [.64, .36])
        self.assertAlmostEqual(norm, 1)
        np.testing.assert_allclose(np.abs(pairs[0]["hole"]), [1, 0, 0, 0])
        np.testing.assert_allclose(np.abs(pairs[0]["electron"]), [0, 0, 1, 0])

    def test_shared_final_orbital_produces_one_pair(self):
        transitions = [((0, 0), (2, 0), .8), ((1, 0), (2, 0), -.6)]
        pairs, _ = build_nto_pairs(transitions, [np.eye(3)], [1], min_weight=0)
        self.assertEqual(len(pairs), 1)
        np.testing.assert_allclose(np.abs(pairs[0]["hole"]), [.8, .6, 0])
        self.assertLess(pairs[0]["hole"][0] * pairs[0]["hole"][1], 0)

    def test_open_shell_weights_share_one_normalization(self):
        transitions = [((0, 0), (1, 0), .8), ((0, 1), (1, 1), .6)]
        pairs, _ = build_nto_pairs(transitions, [np.eye(2), np.eye(2)], [0, 0])
        self.assertEqual([p["spin"] for p in pairs], ["alpha", "beta"])
        np.testing.assert_allclose([p["weight"] for p in pairs], [.64, .36])

    def test_rejects_missing_or_incompatible_orbitals(self):
        with self.assertRaises(ValueError):
            build_nto_pairs([((0, 0), (3, 0), 1.)], [np.eye(2)], [0])
        with self.assertRaises(ValueError):
            build_nto_pairs([((0, 0), (1, 1), 1.)], [np.eye(2), np.eye(2)], [0, 0])


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _data(energies, oscs, rots):
        from types import SimpleNamespace as NS
        es = NS(et_energies=[energies], et_oscs=[oscs], et_rot=[rots])
        return NS(results=NS(excited_states=es))

    def test_dominant_only_without_rotatory_strengths(self):
        from quchemreport.data_enrichment.compute import select_excited_states_indices
        data = self._data([1.0] * 8, [0.0] * 5 + [0.5, 0.01, 0.2], None)
        self.assertEqual(select_excited_states_indices(data, 0, "dominant_only"), [0, 1, 2, 3, 4, 5, 7])

    def test_dominant_only_with_rotatory_strengths(self):
        from quchemreport.data_enrichment.compute import select_excited_states_indices
        data = self._data([1.0] * 7, [0.0] * 7, [0.0] * 5 + [50.0, 1.0])
        self.assertEqual(select_excited_states_indices(data, 0, "dominant_only"), [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
