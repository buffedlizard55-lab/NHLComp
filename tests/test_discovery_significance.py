"""Tests for the multiple-testing correction in the discovery layer.

Why this file exists: the discovery scan evaluates hundreds of candidate triggers on every
run.  At a per-test 5% threshold, a search of that size guarantees false survivors on data
with no edge at all -- so a "significant" trigger found by searching is not evidence of
anything until the p-value is judged against a threshold that accounts for how much was
searched.  These tests pin down:

* the z/p arithmetic (against hand-computed values), including the cases where the test is
  undefined and must return None rather than a flattering 0;
* Holm's step-down procedure -- that it stops at the first failure, that tests with no
  p-value still count toward the family size, and that the *same* evidence is significant
  when it was the only thing tested and not significant when it was found by searching
  thousands of triggers;
* that the scan annotates every candidate with its corrected verdict, and that promotion
  records the corrected p-values on the strategy's public record;
* that a candidate which does not clear the correction is still promotable -- to FORWARD
  TEST, which is what forward testing is for -- but its evidence says in terms that no edge
  is claimed.
"""
from __future__ import annotations

import json
import math
import unittest

from nhlcomp.discovery import (
    FAMILY_ALPHA,
    Candidate,
    DiscoveryEngine,
    binomial_z,
    holm_significant,
    p_value_upper,
)
from nhlcomp.store import Store


def _candidate(**kw) -> Candidate:
    base = dict(feature="home_back_to_back", operator=">=", threshold=1.0, bet_side="home",
                rationale="home team on the second night of a back-to-back", train_n=45, train_hit=1.0,
                train_lift=0.38, valid_n=30, valid_hit=1.0, valid_lift=0.38,
                verdict="edge")
    base.update(kw)
    return Candidate(**base)


def _synthetic_rows(n: int = 600, every: int = 4) -> list[dict]:
    """Games where ``home_back_to_back`` is a genuinely strong trigger: home wins all of them.

    Everything else alternates, so the base rate sits near 0.6 and the lift is real.  No
    prices -- :class:`TestPricedSignificance` adds those separately.
    """
    rows = []
    for i in range(n):
        trigger = (i % every == 0)
        home_wins = True if trigger else (i % 2 == 0)
        rows.append({"game_id": f"G{i:04d}", "home_id": 1, "away_id": 2,
                     "_winner": 1 if home_wins else 2, "home_back_to_back": 1 if trigger else 0})
    return rows


class TestStatistics(unittest.TestCase):
    def test_binomial_z_matches_hand_computation(self):
        # 60 hits in 100 trials against p0=0.5:  z = (60-50)/sqrt(100*0.25) = 2.0
        self.assertAlmostEqual(binomial_z(60, 100, 0.5), 2.0, places=9)
        self.assertAlmostEqual(p_value_upper(2.0), 0.5 * math.erfc(2.0 / math.sqrt(2)), places=12)
        self.assertAlmostEqual(p_value_upper(2.0), 0.02275, places=4)

    def test_null_z_is_a_coin_flip(self):
        self.assertAlmostEqual(binomial_z(50, 100, 0.5), 0.0, places=12)
        self.assertAlmostEqual(p_value_upper(0.0), 0.5, places=12)

    def test_undefined_tests_return_none_never_zero(self):
        """A test that cannot be computed must not be reported as a perfect p-value."""
        self.assertIsNone(binomial_z(None, 100, 0.5))
        self.assertIsNone(binomial_z(60, 0, 0.5))
        self.assertIsNone(binomial_z(60, None, 0.5))
        self.assertIsNone(binomial_z(60, 100, None))
        self.assertIsNone(binomial_z(60, 100, 0.0))   # degenerate reference probability
        self.assertIsNone(binomial_z(60, 100, 1.0))
        self.assertIsNone(p_value_upper(None))

    def test_z_is_used_against_a_reference_not_fitted_to_the_sample(self):
        """The price test compares wins to the offer actually paid -- not to 50%."""
        # 15 wins out of 15 at an average price of 0.50 is far stronger evidence than the
        # same record bought at 0.90, where winning was already the favourite's job.
        z_cheap = binomial_z(15, 15, 0.50)
        z_dear = binomial_z(15, 15, 0.90)
        self.assertAlmostEqual(z_cheap, 3.873, places=3)
        self.assertAlmostEqual(z_dear, (15 - 13.5) / math.sqrt(15 * 0.9 * 0.1), places=9)
        self.assertLess(p_value_upper(z_cheap), p_value_upper(z_dear))


class TestHolmCorrection(unittest.TestCase):
    def test_classic_three_test_family(self):
        # m=3, alpha=0.05 -> thresholds 0.0167, 0.025, 0.05 for the ordered p-values
        got = holm_significant([("a", 0.001), ("b", 0.02), ("c", 0.4)], alpha=0.05)
        self.assertTrue(got["a"]["significant"])
        self.assertTrue(got["b"]["significant"])
        self.assertFalse(got["c"]["significant"])
        self.assertAlmostEqual(got["a"]["holm_threshold"], 0.05 / 3, places=12)
        self.assertAlmostEqual(got["b"]["holm_threshold"], 0.05 / 2, places=12)
        self.assertEqual([got[k]["rank"] for k in ("a", "b", "c")], [1, 2, 3])

    def test_step_down_stops_at_the_first_failure(self):
        """Once one ordered p-value fails, nothing weaker is declared significant."""
        got = holm_significant([("weak", 0.04), ("weaker", 0.045)], alpha=0.05)
        # 0.04 > 0.05/2 = 0.025, so the procedure stops: 0.045 is not significant either,
        # even though it would clear a naive per-test 5% threshold.
        self.assertFalse(got["weak"]["significant"])
        self.assertFalse(got["weaker"]["significant"])
        self.assertGreater(got["weaker"]["holm_threshold"], got["weak"]["holm_threshold"])

    def test_same_evidence_verdict_depends_on_search_size(self):
        """The headline: searching more makes identical evidence weaker, not stronger."""
        alone = holm_significant([("solo", 0.001)], alpha=0.05)["solo"]
        searched = holm_significant(
            [("found", 0.001)] + [(f"pad{i}", None) for i in range(2999)], alpha=0.05)["found"]
        self.assertTrue(alone["significant"])
        self.assertFalse(searched["significant"])
        self.assertAlmostEqual(searched["holm_threshold"], 0.05 / 3000, places=15)

    def test_tests_without_a_p_value_still_count_toward_the_family(self):
        """Padding with undefined tests must tighten the threshold, not be ignored."""
        fam = holm_significant([("a", 0.01)] + [(f"x{i}", None) for i in range(99)], alpha=0.05)
        self.assertFalse(fam["a"]["significant"])          # 0.01 > 0.05/100 = 5e-4
        self.assertAlmostEqual(fam["a"]["holm_threshold"], 0.05 / 100, places=15)
        self.assertEqual(len(fam), 100)
        # a padded test has no rank and no threshold, and is never declared significant
        for k, v in fam.items():
            if k == "a":
                continue
            self.assertFalse(v["significant"])
            self.assertIsNone(v["holm_threshold"])
            self.assertIsNone(v["rank"])
        # the identical p-value tested on its own clears easily: the search size is what moved
        self.assertTrue(holm_significant([("a", 0.01)], alpha=0.05)["a"]["significant"])

    def test_empty_family(self):
        self.assertEqual(holm_significant([], alpha=0.05), {})

    def test_family_alpha_is_preregistered_not_tuned(self):
        self.assertAlmostEqual(FAMILY_ALPHA, 0.05, places=12)


class TestScanAnnotation(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def test_scan_annotates_every_candidate_with_a_corrected_verdict(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        cands = eng.scan(_synthetic_rows())
        self.assertTrue(cands, "the planted trigger must survive the split gates")
        for c in cands:
            self.assertEqual(c.n_tests, eng.tested)
            self.assertAlmostEqual(c.alpha_family, FAMILY_ALPHA, places=12)
            self.assertIsNotNone(c.alpha_holm)
            self.assertGreater(c.alpha_holm, 0.0)
            self.assertIn(c.significant_accuracy, (True, False))
            self.assertIn(c.significant_priced, (True, False))
            self.assertTrue(c.significance_note.startswith("Clears")
                            or c.significance_note.startswith("Does NOT survive"))
            if c.significant_accuracy or c.significant_priced:
                self.assertIn("Holm", c.significance_note)
            else:
                self.assertIn("NO EDGE IS CLAIMED", c.significance_note)
        # the trigger we planted has a real effect, so at least one p-value is computed
        planted = [c for c in cands if c.feature == "home_back_to_back" and c.bet_side == "home"]
        self.assertTrue(planted)
        self.assertIsNotNone(planted[0].valid_p)
        self.assertLess(planted[0].valid_p, 0.01)

    def test_annotated_z_and_p_are_consistent_with_the_counts(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        cands = eng.scan(_synthetic_rows())
        c = [x for x in cands if x.feature == "home_back_to_back" and x.bet_side == "home"][0]
        self.assertAlmostEqual(c.valid_z, binomial_z(round(c.valid_hit * c.valid_n),
                                                     c.valid_n,
                                                     c.valid_hit - c.valid_lift), places=6)
        self.assertAlmostEqual(c.valid_p, p_value_upper(c.valid_z), places=12)

    def test_caveat_states_the_corrected_threshold(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        eng.scan(_synthetic_rows())
        cav = eng.search_size_caveat()
        self.assertIn(str(eng.tested), cav)
        self.assertIn("Holm", cav)
        self.assertIn("family-wise error rate", cav)
        self.assertIn("NO EDGE IS CLAIMED", cav)
        self.assertIn("by chance alone", cav)
        self.assertIn(f"{eng.significance['holm_threshold_first']:.2e}", cav)

    def test_annotation_is_a_no_op_on_an_empty_scan(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        out = eng.annotate_significance([])
        self.assertEqual(out["candidates"], 0)
        self.assertEqual(out["significant_accuracy"], 0)
        self.assertEqual(eng.scan([]), [])


class TestPricedSignificance(unittest.TestCase):
    """The price test -- wins against the offer actually paid -- is the one that matters for
    a betting edge, so the scan must compute it whenever closing prices exist."""

    def setUp(self):
        self.store = Store(":memory:")

    def test_priced_z_is_computed_from_the_paid_offer(self):
        rows = _synthetic_rows()
        for r in rows:
            r["mkt_close_home_ask"] = 0.50
            r["mkt_close_is_latest"] = False
        eng = DiscoveryEngine(self.store, verbose=False)
        cands = eng.scan(rows)
        c = [x for x in cands if x.feature == "home_back_to_back" and x.bet_side == "home"][0]
        self.assertEqual(c.valid_avg_price, 0.50)
        # every validation game is priced in this fixture, so the two tests share their
        # trials but use different references: the split's base rate vs the offer paid
        self.assertEqual(c.valid_priced_n, c.valid_n)
        self.assertAlmostEqual(
            c.priced_z,
            binomial_z(round(c.valid_hit * c.valid_priced_n), c.valid_priced_n,
                       c.valid_avg_price), places=9)
        self.assertAlmostEqual(c.priced_z, 5.477, places=3)
        self.assertAlmostEqual(c.priced_p, p_value_upper(c.priced_z), places=12)
        self.assertLess(c.priced_p, c.alpha_holm)
        self.assertTrue(c.significant_priced)
        self.assertTrue(c.priced_p <= c.alpha_holm or not c.significant_priced)
        # whichever way it fell, the note must state the verdict explicitly
        self.assertTrue(c.significance_note.startswith("Clears")
                        or "NO EDGE IS CLAIMED" in c.significance_note,
                        c.significance_note)
        if c.significant_priced:
            self.assertIn("the price test", c.significance_note)
            self.assertIn("CANDIDATE", c.significance_note)

    def test_unpriced_rows_leave_the_price_test_undefined(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        cands = eng.scan(_synthetic_rows())
        for c in cands:
            if c.price_verdict == "unpriced":
                self.assertIsNone(c.priced_z)
                self.assertIsNone(c.priced_p)
                self.assertFalse(c.significant_priced)


class TestPromotionRecord(unittest.TestCase):
    """What lands on the public record when a search survivor is promoted."""

    def setUp(self):
        self.store = Store(":memory:")

    def _promote(self, eng, cands):
        created = eng.promote(cands)
        recs = []
        for d in created:
            exp = self.store.one(
                "SELECT * FROM experiments WHERE strategy_id=? AND version=?",
                (d["strategy_id"], d["version"]))
            self.assertIsNotNone(exp, "every promotion must write an experiment row")
            recs.append((d, exp))
        return recs

    def test_promotion_carries_the_corrected_p_values(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        cands = eng.scan(_synthetic_rows())
        recs = self._promote(eng, cands)
        self.assertTrue(recs)
        for d, exp in recs:
            mt = json.loads(exp["result_json"])["multiple_testing"]
            self.assertEqual(mt["n_tests"], eng.tested)
            self.assertAlmostEqual(mt["alpha_family"], FAMILY_ALPHA, places=12)
            self.assertIsNotNone(mt["holm_threshold"])
            self.assertIn("z_accuracy", mt)
            self.assertIn("p_accuracy", mt)
            self.assertIn("z_price", mt)
            self.assertIn("p_price", mt)
            self.assertIn(mt["significant_accuracy"], (True, False))
            self.assertIn(mt["significant_priced"], (True, False))
            self.assertEqual(mt["note"], d["hypothesis"].split("MULTIPLE TESTING: ")[1])
            self.assertIn("MULTIPLE TESTING:", d["hypothesis"])
            self.assertEqual(exp["hypothesis"], d["hypothesis"])
            self.assertEqual(d["status"], "candidate")
            self.assertEqual(d["test_mode"], "FORWARD TEST")
            if not (mt["significant_accuracy"] or mt["significant_priced"]):
                self.assertIn("no edge is claimed", exp["conclusion"])
                self.assertIn("Holm", exp["conclusion"])

    def test_a_candidate_that_fails_correction_is_still_forward_testable(self):
        """Failing the corrected test blocks the *claim*, not the experiment."""
        eng = DiscoveryEngine(self.store, verbose=False)
        c = _candidate()
        eng.tested = 5000
        eng.annotate_significance([c])
        self.assertFalse(c.significant_accuracy)
        self.assertFalse(c.significant_priced)
        self.assertIn("NO EDGE IS CLAIMED", c.significance_note)
        recs = self._promote(eng, [c])
        self.assertEqual(len(recs), 1, "it must still be promoted to forward testing")
        d, exp = recs[0]
        self.assertEqual(d["status"], "candidate")
        self.assertIn("no edge is claimed", exp["conclusion"])
        self.assertIn("5000 triggers", d["hypothesis"])

    def test_a_candidate_that_clears_correction_says_so(self):
        eng = DiscoveryEngine(self.store, verbose=False)
        c = _candidate(valid_z=6.0, valid_p=p_value_upper(6.0))
        eng.tested = 3
        eng.annotate_significance([c])
        self.assertTrue(c.significant_accuracy)
        self.assertIn("Clears the accuracy test", c.significance_note)
        self.assertIn("CANDIDATE", c.significance_note)
        recs = self._promote(eng, [c])
        self.assertEqual(len(recs), 1)
        d, exp = recs[0]
        self.assertNotIn("no edge is claimed", exp["conclusion"])
        self.assertIn("3 triggers", d["hypothesis"])
        self.assertEqual(json.loads(exp["result_json"])["multiple_testing"]["n_tests"], 3)


if __name__ == "__main__":
    unittest.main()
