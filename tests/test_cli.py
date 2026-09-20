"""CLI argument-parsing tests.

These exist because a global flag placed after the subcommand was silently ignored once:
the subparser's own default overwrote the value the top-level parser had already read, so
CI wrote to the default database path while claiming to use ``--db``.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest

from nhlcomp import cli

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse(argv):
    """Rebuild the parser the same way main() does and parse without dispatching."""
    common = cli._common()
    p = argparse.ArgumentParser(prog="nhlcomp", parents=[common])
    sub_common = cli._common(suppress=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", parents=[sub_common]).set_defaults(fn=cli.cmd_init)
    sp = sub.add_parser("run", parents=[sub_common])
    cli._ingest_args(sp)
    sp.set_defaults(fn=cli.cmd_run)
    sub.add_parser("report", parents=[sub_common]).set_defaults(fn=cli.cmd_report)
    return p.parse_args(argv)


class TestCliParsing(unittest.TestCase):
    def test_global_flag_before_subcommand(self):
        a = parse(["--db", "/tmp/x.db", "report"])
        self.assertEqual(a.db, "/tmp/x.db")

    def test_global_flag_after_subcommand(self):
        a = parse(["report", "--db", "/tmp/x.db"])
        self.assertEqual(a.db, "/tmp/x.db")

    def test_both_positions_agree(self):
        self.assertEqual(parse(["--db", "/tmp/x.db", "report"]).db,
                         parse(["report", "--db", "/tmp/x.db"]).db)

    def test_default_when_absent(self):
        self.assertEqual(parse(["report"]).db, cli.DEFAULT_DB)

    def test_after_subcommand_wins_over_before(self):
        a = parse(["--db", "/tmp/first.db", "report", "--db", "/tmp/second.db"])
        self.assertEqual(a.db, "/tmp/second.db")

    def test_subcommand_options_still_parse(self):
        a = parse(["run", "--db", "/tmp/x.db", "--seasons", "20252026", "--settled-pages", "3"])
        self.assertEqual(a.seasons, "20252026")
        self.assertEqual(a.settled_pages, 3)
        self.assertEqual(a.db, "/tmp/x.db")

    def test_offline_flag(self):
        self.assertTrue(parse(["--offline", "report"]).offline)
        self.assertTrue(parse(["report", "--offline"]).offline)
        self.assertFalse(parse(["report"]).offline)


class TestCliModuleRuns(unittest.TestCase):
    """Invoke the real module in a subprocess, so a packaging regression is caught."""

    def _run(self, *args):
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
        return subprocess.run([sys.executable, "-m", "nhlcomp", *args],
                              capture_output=True, text=True, env=env, cwd=ROOT)

    def test_init_then_report_on_a_temp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        r1 = self._run("init", "--db", path)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertIn(path, r1.stdout)
        from nhlcomp.sources.registry import SOURCES
        self.assertIn(f"{len(SOURCES)} sources registered", r1.stdout)
        r2 = self._run("report", "--db", path)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("competition status", r2.stdout)

    def test_unknown_subcommand_exits_nonzero(self):
        """A failing command must not look like success to a shell pipeline."""
        r = self._run("definitely-not-a-command")
        self.assertNotEqual(r.returncode, 0)

    def test_pipefail_semantics_are_respected_in_the_workflow(self):
        lines = open(os.path.join(ROOT, ".github", "workflows", "research.yml"),
                     encoding="utf-8").read().splitlines()
        # every step that pipes a command into tee must set pipefail, otherwise a failing
        # command is masked by tee's exit status and CI reports a false success
        checked = 0
        for i, line in enumerate(lines):
            if "| tee data/" not in line:
                continue
            checked += 1
            # walk back to the start of this step's run block
            j = i
            while j > 0 and "run: |" not in lines[j]:
                j -= 1
            block = "\n".join(lines[j:i + 1])
            self.assertIn("pipefail", block,
                          f"step piping to tee without pipefail: {line.strip()}")
        self.assertGreaterEqual(checked, 4, "expected the log-capturing steps to be checked")


if __name__ == "__main__":
    unittest.main()
