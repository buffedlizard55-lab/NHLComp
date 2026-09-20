"""Tests for the storage layer: append-only bets, audit trail, irregularities."""

import json
import os
import tempfile
import unittest

from nhlcomp.store import Store, utcnow


def make_store() -> Store:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return Store(path)


def bet(bet_id="B1", **kw):
    row = {
        "bet_id": bet_id, "strategy_id": "S", "strategy_version": 1, "username": "S_001",
        "test_mode": "FORWARD TEST", "game_id": 1, "game_date": "2026-01-01",
        "matchup": "A@B", "market": "moneyline", "selection": "home",
        "bet_type": "binary_contract", "provider": "kalshi", "odds_format": "binary",
        "price": 0.50, "implied_prob": 0.50, "model_prob": 0.60, "edge": 0.10,
        "decision_ts": utcnow(), "bet_ts": utcnow(), "stake": 100.0, "filled_size": 200.0,
        "entry_price": 0.50, "result": "OPEN", "created_at": utcnow(),
    }
    row.update(kw)
    return row


class TestStore(unittest.TestCase):
    def setUp(self):
        self.s = make_store()
        self.s.execute("INSERT INTO teams(team_id, abbrev, full_name) VALUES(1,'AAA','Team A')")
        self.s.execute("INSERT INTO teams(team_id, abbrev, full_name) VALUES(2,'BBB','Team B')")
        self.s.commit()

    def tearDown(self):
        self.s.close()

    def test_record_bet_is_append_only(self):
        self.assertTrue(self.s.record_bet(bet("B1")))
        # a second insert of the same id must be refused, not overwrite
        self.assertFalse(self.s.record_bet(bet("B1", stake=999.0)))
        row = self.s.one("SELECT stake FROM bets WHERE bet_id='B1'")
        self.assertEqual(float(row["stake"]), 100.0)

    def test_settlement_writes_audit_trail(self):
        self.s.record_bet(bet("B2"))
        self.s.settle_bet("B2", result="WIN", pnl=100.0, reason="official result 3-2")
        audit = self.s.query("SELECT * FROM bet_audit WHERE bet_id='B2' ORDER BY id")
        self.assertEqual([a["action"] for a in audit], ["INSERT", "SETTLE"])
        self.assertIn("3-2", audit[-1]["reason"])
        row = self.s.one("SELECT result, pnl, roi FROM bets WHERE bet_id='B2'")
        self.assertEqual(row["result"], "WIN")
        self.assertAlmostEqual(float(row["roi"]), 1.0)

    def test_resettlement_is_recorded_not_silent(self):
        self.s.record_bet(bet("B3"))
        self.s.settle_bet("B3", result="WIN", pnl=100.0)
        self.s.settle_bet("B3", result="LOSS", pnl=-100.0, reason="score corrected")
        audit = self.s.query("SELECT * FROM bet_audit WHERE bet_id='B3'")
        self.assertEqual(len(audit), 3)
        self.assertIn("re-settlement over WIN", audit[-1]["reason"])

    def test_amend_restricts_fields_and_audits(self):
        self.s.record_bet(bet("B4"))
        with self.assertRaises(ValueError):
            self.s.amend_bet("B4", {"strategy_id": "OTHER"}, reason="nope")
        self.s.amend_bet("B4", {"notes": "corrected price source"}, reason="source was wrong")
        row = self.s.one("SELECT amended, notes FROM bets WHERE bet_id='B4'")
        self.assertEqual(row["amended"], 1)
        self.assertEqual(row["notes"], "corrected price source")

    def test_flag_is_idempotent(self):
        self.assertTrue(self.s.flag("missing_result", "game 1 FINAL without scores"))
        self.assertFalse(self.s.flag("missing_result", "game 1 FINAL without scores"))
        self.assertEqual(self.s.one("SELECT COUNT(*) c FROM irregularities")["c"], 1)

    def test_bankroll_reserves_open_exposure(self):
        self.s.upsert_strategy({"strategy_id": "S", "version": 1, "username": "S_001",
                                "name": "S", "category": "test", "hypothesis": "h",
                                "data_used": "d", "entry_rule": "e", "price_rule": "p",
                                "settlement_rule": "s", "markets": "moneyline",
                                "created_at": utcnow(), "starting_bankroll": 1000.0,
                                "bankroll": 1000.0})
        self.s.record_bet(bet("B5", strategy_id="S"))
        self.s.sync_bankroll("S", 1)
        self.assertAlmostEqual(float(self.s.one(
            "SELECT bankroll FROM strategies WHERE strategy_id='S'")["bankroll"]), 900.0)
        self.s.settle_bet("B5", result="WIN", pnl=100.0)
        self.s.sync_bankroll("S", 1)
        self.assertAlmostEqual(float(self.s.one(
            "SELECT bankroll FROM strategies WHERE strategy_id='S'")["bankroll"]), 1100.0)

    def test_strategy_versioning_never_overwrites(self):
        base = {"strategy_id": "S", "username": "S_001", "name": "S", "category": "test",
                "hypothesis": "h", "data_used": "d", "entry_rule": "e", "price_rule": "p",
                "settlement_rule": "s", "markets": "moneyline", "created_at": utcnow(),
                "starting_bankroll": 1000.0, "bankroll": 1000.0, "origin": "self_generated"}
        self.s.upsert_strategy({**base, "version": 1, "hypothesis": "original"})
        self.s.upsert_strategy({**base, "version": 2, "hypothesis": "revised"})
        rows = self.s.query("SELECT version, hypothesis FROM strategies WHERE strategy_id='S'")
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["version"]: r["hypothesis"] for r in rows},
                         {1: "original", 2: "revised"})
        self.s.set_strategy_status("S", 2, "active", reason="promoted")
        life = self.s.query("SELECT * FROM strategy_lifecycle WHERE strategy_id='S'")
        self.assertEqual(life[0]["to_status"], "active")


class TestRawProvenance(unittest.TestCase):
    def test_raw_response_is_deduplicated_and_digested(self):
        s = make_store()
        body = json.dumps({"a": 1})
        i1 = s.record_raw("https://example.test/x", body)
        i2 = s.record_raw("https://example.test/x", body)
        self.assertEqual(i1, i2)
        row = s.one("SELECT * FROM raw_response WHERE id=?", (i1,))
        self.assertEqual(len(row["sha256"]), 64)
        self.assertEqual(row["byte_size"], len(body.encode()))
        s.close()


if __name__ == "__main__":
    unittest.main()
