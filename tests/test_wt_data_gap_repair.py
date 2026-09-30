import json
import sys
import tempfile
import unittest
import zipfile
from ctypes import Structure, c_char, c_double, c_uint32
from pathlib import Path

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import wt_data_gap_repair as repair


class GapWindowTests(unittest.TestCase):
    def test_august_27_window_encodings(self):
        window = repair.GapWindow(20260827, 900, 925)

        self.assertEqual(window.bar_start, 202608270900)
        self.assertEqual(window.bar_end, 202608270925)
        self.assertEqual(window.tick_start, 20260827090000000)
        self.assertEqual(window.tick_end, 20260827092500000)

    def test_rejects_invalid_time(self):
        with self.assertRaisesRegex(ValueError, "Invalid end_hhmm"):
            repair.GapWindow(20260827, 900, 1260)

    def test_simple_dump_and_merge_commands_are_available(self):
        parser = repair.build_parser()
        dump = parser.parse_args(
            [
                "dump",
                "--source",
                "source",
                "--output",
                "patch",
                "--date",
                "20260827",
            ]
        )
        merge = parser.parse_args(
            [
                "merge",
                "--production",
                "production",
                "--patch",
                "patch",
                "--output",
                "merged",
            ]
        )

        self.assertEqual(dump.command, "dump")
        self.assertEqual(merge.command, "merge")


class SpliceTests(unittest.TestCase):
    def setUp(self):
        self.window = repair.GapWindow(20260827, 900, 925)

    def test_bar_splice_keeps_production_outside_and_donor_inside(self):
        production = pd.DataFrame(
            {
                "bartime": [202608270859, 202608270901, 202608270925, 202608270926],
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        )
        donor = pd.DataFrame(
            {
                "bartime": [202608270901, 202608270925],
                "close": [20.0, 30.0],
            }
        )

        result = repair._splice_frames(production, donor, "min1", self.window)

        self.assertEqual(result["bartime"].tolist(), [202608270859, 202608270901, 202608270925, 202608270926])
        self.assertEqual(result["close"].tolist(), [1.0, 20.0, 30.0, 4.0])

    def test_tick_splice_allows_multiple_ticks_at_same_timestamp(self):
        production = pd.DataFrame(
            {
                "time": [20260827085959999, 20260827090100000, 20260827092600000],
                "price": [1.0, 2.0, 3.0],
            }
        )
        donor = pd.DataFrame(
            {
                "time": [20260827090100000, 20260827090100000],
                "price": [20.0, 21.0],
            }
        )

        result = repair._splice_frames(production, donor, "ticks", self.window)

        self.assertEqual(result["price"].tolist(), [1.0, 20.0, 21.0, 3.0])


class ArchiveTests(unittest.TestCase):
    def _make_patch(self, root: Path):
        data_file = root / "min1" / "SHFE" / "rb2610.dsb"
        data_file.parent.mkdir(parents=True)
        data_file.write_bytes(b"small dsb patch")
        manifest = {
            "format": "wt-data-gap-patch",
            "version": 1,
            "window": repair.GapWindow(20260827, 900, 925).as_dict(),
            "files": [
                {
                    "period": "min1",
                    "exchange": "SHFE",
                    "contract": "rb2610",
                    "relative_path": "min1/SHFE/rb2610.dsb",
                    "row_count": 25,
                    "first_time": 202608270901,
                    "last_time": 202608270925,
                    "sha256": repair.sha256_file(data_file),
                }
            ],
        }
        (root / repair.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    def test_zip_and_verified_extract_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            patch = base / "patch"
            patch.mkdir()
            self._make_patch(patch)
            archive = base / "patch.zip"
            digest = repair.create_patch_zip(patch, archive)

            extracted = base / "extracted"
            manifest = repair.extract_patch_zip(archive, extracted, expected_sha256=digest)

            self.assertEqual(manifest["files"][0]["row_count"], 25)
            self.assertEqual((extracted / "min1" / "SHFE" / "rb2610.dsb").read_bytes(), b"small dsb patch")

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = base / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../outside.txt", "unsafe")

            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                repair.extract_patch_zip(archive, base / "output")


class LosslessTickWriterTests(unittest.TestCase):
    def test_preserves_non_level_zero_and_auxiliary_fields(self):
        class FakeTickStruct(Structure):
            _fields_ = [
                ("exchg", c_char * 16),
                ("code", c_char * 32),
                ("upper_limit", c_double),
                ("volume", c_double),
                ("bid_price_9", c_double),
                ("ask_qty_9", c_double),
                ("action_date", c_uint32),
            ]

        class FakeHelper:
            saved = None

            def store_ticks(self, tickFile, firstTick, count):
                self.saved = {
                    "path": tickFile,
                    "count": count,
                    "upper_limit": firstTick[0].upper_limit,
                    "volume": firstTick[0].volume,
                    "bid_price_9": firstTick[0].bid_price_9,
                    "ask_qty_9": firstTick[0].ask_qty_9,
                    "action_date": firstTick[0].action_date,
                }

        helper = FakeHelper()
        frame = pd.DataFrame(
            {
                "time": [20260827090100000],
                "exchg": [b"SHFE"],
                "code": [b"rb2610"],
                "upper_limit": [4100.0],
                "volume": [2.0],
                "bid_price_9": [3791.0],
                "ask_qty_9": [12.0],
                "action_date": [20260827],
            }
        )

        with tempfile.TemporaryDirectory() as temporary:
            repair._store_ticks_lossless(
                frame,
                Path(temporary) / "rb2610.dsb",
                "SHFE",
                "rb2610",
                {"helper": helper, "tick_struct": FakeTickStruct},
            )

        self.assertEqual(helper.saved["count"], 1)
        self.assertEqual(helper.saved["upper_limit"], 4100.0)
        self.assertEqual(helper.saved["volume"], 2.0)
        self.assertEqual(helper.saved["bid_price_9"], 3791.0)
        self.assertEqual(helper.saved["ask_qty_9"], 12.0)
        self.assertEqual(helper.saved["action_date"], 20260827)


if __name__ == "__main__":
    unittest.main()
