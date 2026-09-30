"""Create and apply small WtPy historical-data gap patches.

The patch archive contains only records inside the requested time window.  It
does not contain complete contract histories.  The intended workflow is:

1. ``export`` the gap from the donor (normally a local history store),
2. ``zip`` and transfer the patch directory,
3. ``unzip`` it on the production computer,
4. ``combine`` it with production data into a staging directory,
5. ``validate`` the staging directory, and
6. explicitly ``install`` the validated files with backups.

WtPy imports are deliberately lazy so archive and dataframe tests can run on a
computer that does not have the WtPy runtime installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_EXCHANGES = ("CFFEX", "DCE", "CZCE", "SHFE", "INE", "GFEX")
BAR_PERIODS = ("min1", "min5")
ALL_PERIODS = BAR_PERIODS + ("ticks",)
BAR_PERIOD_CODES = {"min1": "m1", "min5": "m5"}
WT_BAR_TIME_OFFSET = 199000000000
MANIFEST_NAME = "manifest.json"
MERGE_MANIFEST_NAME = "merge_manifest.json"


@dataclass(frozen=True)
class GapWindow:
    """An inclusive local-time repair window for one trading date."""

    trading_date: int
    start_hhmm: int
    end_hhmm: int

    def __post_init__(self) -> None:
        try:
            parsed_date = dt.datetime.strptime(str(self.trading_date), "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid trading date: {self.trading_date}") from exc
        object.__setattr__(self, "_date", parsed_date)
        for label, value in (("start_hhmm", self.start_hhmm), ("end_hhmm", self.end_hhmm)):
            hours, minutes = divmod(value, 100)
            if not (0 <= hours <= 23 and 0 <= minutes <= 59):
                raise ValueError(f"Invalid {label}: {value:04d}")
        if self.start_hhmm > self.end_hhmm:
            raise ValueError("start_hhmm must not be later than end_hhmm")

    @property
    def date(self) -> dt.date:
        return self._date

    @property
    def bar_start(self) -> int:
        return self.trading_date * 10000 + self.start_hhmm

    @property
    def bar_end(self) -> int:
        return self.trading_date * 10000 + self.end_hhmm

    @property
    def tick_start(self) -> int:
        return self.bar_start * 100000

    @property
    def tick_end(self) -> int:
        return self.bar_end * 100000

    def bounds(self, period: str) -> tuple[int, int]:
        if period in BAR_PERIODS:
            return self.bar_start, self.bar_end
        if period == "ticks":
            return self.tick_start, self.tick_end
        raise ValueError(f"Unsupported period: {period}")

    def as_dict(self) -> dict[str, int]:
        return {
            "trading_date": self.trading_date,
            "start_hhmm": self.start_hhmm,
            "end_hhmm": self.end_hhmm,
            "bar_start": self.bar_start,
            "bar_end": self.bar_end,
            "tick_start": self.tick_start,
            "tick_end": self.tick_end,
        }


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_wt_runtime() -> dict[str, Any]:
    """Load WtPy and repository DSB writers only when DSB work is requested."""

    try:
        from wtpy.wrapper import WtDataHelper
    except ModuleNotFoundError:
        common_checkout = Path("C:/dev/wtpy")
        if common_checkout.is_dir() and str(common_checkout) not in sys.path:
            sys.path.append(str(common_checkout))
        try:
            from wtpy.wrapper import WtDataHelper
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WtPy is required for DSB operations. Run this command in the "
                "same Python environment used by the data collector."
            ) from exc

    from wtpy.WtCoreDefs import WTSTickStruct
    from pycmqlib3.utility.process_wt_data import save_bars_to_dsb

    return {
        "helper": WtDataHelper(),
        "save_bars": save_bars_to_dsb,
        "tick_struct": WTSTickStruct,
    }


def _require_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _bar_time_column(frame: pd.DataFrame) -> str:
    if "bartime" in frame.columns:
        return "bartime"
    if "time" in frame.columns:
        return "time"
    raise ValueError("Bar dataframe has neither 'bartime' nor 'time'")


def _time_column(frame: pd.DataFrame, period: str) -> str:
    return _bar_time_column(frame) if period in BAR_PERIODS else "time"


def _read_dsb(path: Path, period: str, runtime: dict[str, Any]) -> pd.DataFrame:
    helper = runtime["helper"]
    if period in BAR_PERIODS:
        records = helper.read_dsb_bars(str(path), isDay=False)
    else:
        records = helper.read_dsb_ticks(str(path))
    if records is None:
        raise ValueError(f"WtPy could not read DSB file: {path}")
    return records.to_df().copy()


def _write_dsb(
    frame: pd.DataFrame,
    path: Path,
    period: str,
    exchange: str,
    contract: str,
    runtime: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if period in BAR_PERIODS:
        writable = frame.copy()
        if "bartime" in writable.columns:
            writable = writable.rename(
                columns={"bartime": "time", "money": "turnover", "hold": "open_interest"}
            )
        if len(writable) and writable["time"].min() >= WT_BAR_TIME_OFFSET:
            writable["time"] = writable["time"].astype("int64") - WT_BAR_TIME_OFFSET
        writable["time"] = writable["time"].astype("int64")
        runtime["save_bars"](
            writable,
            contract=contract,
            folder_loc=str(path.parent),
            period=BAR_PERIOD_CODES[period],
        )
    else:
        _store_ticks_lossless(frame, path, exchange, contract, runtime)


def _store_ticks_lossless(
    frame: pd.DataFrame,
    path: Path,
    exchange: str,
    contract: str,
    runtime: dict[str, Any],
) -> None:
    """Store every WTSTickStruct field exposed by ``read_dsb_ticks``.

    The older repository ``save_ticks_to_dsb`` helper writes only level-zero
    depth and omits several tick fields.  A repair must preserve the complete
    production and donor records, so this writer copies all available struct
    fields, including ten levels of market depth.
    """

    tick_struct = runtime["tick_struct"]
    buffer_type = tick_struct * len(frame)
    buffer = buffer_type()
    struct_fields = [name for name, _ in tick_struct._fields_]
    columns = list(frame.columns)
    column_positions = {name: index for index, name in enumerate(columns)}
    copied_fields = [name for name in struct_fields if name in column_positions]
    integer_fields = {"trading_date", "action_date", "action_time", "reserve"}

    for row_index, values in enumerate(frame.itertuples(index=False, name=None)):
        target = buffer[row_index]
        for field in copied_fields:
            value = values[column_positions[field]]
            if pd.isna(value):
                value = 0
            if field in {"exchg", "code"}:
                if isinstance(value, str):
                    value = value.encode("utf-8")
                elif not isinstance(value, bytes):
                    value = bytes(value)
            elif field in integer_fields:
                value = int(value)
            else:
                value = float(value)
            setattr(target, field, value)

        # Some DSB readers may omit these constant identity fields.
        if "exchg" not in copied_fields:
            target.exchg = exchange.encode("utf-8")
        if "code" not in copied_fields:
            target.code = contract.encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    runtime["helper"].store_ticks(
        tickFile=str(path),
        firstTick=buffer,
        count=len(frame),
    )


def _filter_window(frame: pd.DataFrame, period: str, window: GapWindow) -> pd.DataFrame:
    time_column = _time_column(frame, period)
    start, end = window.bounds(period)
    values = frame[time_column].astype("int64")
    return frame.loc[(values >= start) & (values <= end)].copy()


def _candidate_bar_files(
    source_root: Path,
    period: str,
    exchange: str,
    window: GapWindow,
    only_modified_on_date: bool,
) -> Iterable[Path]:
    candidates = sorted((source_root / period / exchange).glob("*.dsb"))
    if not only_modified_on_date:
        return candidates
    return [
        path
        for path in candidates
        if dt.datetime.fromtimestamp(path.stat().st_mtime).date() == window.date
    ]


def _source_readiness_summary(
    source_root: Path,
    exchanges: Sequence[str],
    periods: Sequence[str],
) -> str:
    latest_bar: Path | None = None
    if any(period in BAR_PERIODS for period in periods):
        for period in BAR_PERIODS:
            if period not in periods:
                continue
            for exchange in exchanges:
                for path in (source_root / period / exchange).glob("*.dsb"):
                    if latest_bar is None or path.stat().st_mtime > latest_bar.stat().st_mtime:
                        latest_bar = path
    latest_tick_date: int | None = None
    if "ticks" in periods:
        for exchange in exchanges:
            tick_root = source_root / "ticks" / exchange
            for path in tick_root.iterdir() if tick_root.is_dir() else ():
                if path.is_dir() and path.name.isdigit() and len(path.name) == 8:
                    latest_tick_date = max(latest_tick_date or 0, int(path.name))

    details = []
    if latest_bar is not None:
        modified = dt.datetime.fromtimestamp(latest_bar.stat().st_mtime).astimezone()
        details.append(f"latest bar-file modification={modified.isoformat()}")
    if latest_tick_date is not None:
        details.append(f"latest tick folder={latest_tick_date}")
    return "; ".join(details) if details else "no source DSB files were found"


def export_gap_patch(
    source_root: os.PathLike[str] | str,
    patch_root: os.PathLike[str] | str,
    window: GapWindow,
    exchanges: Sequence[str] = DEFAULT_EXCHANGES,
    periods: Sequence[str] = ALL_PERIODS,
    *,
    only_bar_files_modified_on_date: bool = True,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export only non-empty records inside ``window`` into small DSB files.

    Bar scanning defaults to files modified on the trading date.  This avoids
    opening thousands of expired contracts after the collector has saved the
    current session.  Pass ``only_bar_files_modified_on_date=False`` to scan all
    bar files if the collector preserves historical file modification times.
    """

    invalid_periods = set(periods) - set(ALL_PERIODS)
    if invalid_periods:
        raise ValueError(f"Unsupported periods: {sorted(invalid_periods)}")
    source = Path(source_root).resolve()
    patch = Path(patch_root).resolve()
    _require_empty_directory(patch)
    runtime = runtime or _load_wt_runtime()

    entries: list[dict[str, Any]] = []
    for period in periods:
        for exchange in exchanges:
            if period in BAR_PERIODS:
                candidates = _candidate_bar_files(
                    source, period, exchange, window, only_bar_files_modified_on_date
                )
            else:
                candidates = sorted(
                    (source / "ticks" / exchange / str(window.trading_date)).glob("*.dsb")
                )
            for source_file in candidates:
                frame = _read_dsb(source_file, period, runtime)
                filtered = _filter_window(frame, period, window)
                if filtered.empty:
                    continue
                time_column = _time_column(filtered, period)
                filtered = filtered.sort_values(time_column, kind="stable").reset_index(drop=True)
                if period in BAR_PERIODS and filtered[time_column].duplicated().any():
                    raise ValueError(f"Duplicate bar timestamps in donor file: {source_file}")

                if period in BAR_PERIODS:
                    relative = Path(period) / exchange / source_file.name
                else:
                    relative = Path("ticks") / exchange / str(window.trading_date) / source_file.name
                target = patch / relative
                _write_dsb(filtered, target, period, exchange, source_file.stem, runtime)
                written = _read_dsb(target, period, runtime)
                written_gap = _filter_window(written, period, window)
                _assert_same_frame(
                    _canonical_frame(written_gap, period),
                    _canonical_frame(filtered, period),
                    f"exported patch {relative.as_posix()}",
                )
                entries.append(
                    {
                        "period": period,
                        "exchange": exchange,
                        "contract": source_file.stem,
                        "relative_path": relative.as_posix(),
                        "row_count": int(len(filtered)),
                        "first_time": int(filtered[time_column].iloc[0]),
                        "last_time": int(filtered[time_column].iloc[-1]),
                        "sha256": sha256_file(target),
                        "source_mtime": source_file.stat().st_mtime,
                    }
                )

    if not entries:
        raise ValueError(
            "No records were found in the requested window. "
            + _source_readiness_summary(source, exchanges, periods)
            + ". The collector may not have saved the requested trading date yet. "
            "Only use --scan-all-bars if the requested date is already present but "
            "the collector preserves old file modification times."
        )
    missing_periods = [period for period in periods if not any(e["period"] == period for e in entries)]
    if missing_periods:
        raise ValueError(
            "The patch would be incomplete because no records were exported for: "
            + ", ".join(missing_periods)
        )

    manifest = {
        "format": "wt-data-gap-patch",
        "version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "source_root": str(source),
        "window": window.as_dict(),
        "bar_file_selection": (
            "modified_on_trading_date" if only_bar_files_modified_on_date else "all_files"
        ),
        "files": entries,
        "totals": {
            "files": len(entries),
            "rows": sum(entry["row_count"] for entry in entries),
        },
    }
    (patch / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_patch_manifest(patch_root: os.PathLike[str] | str) -> dict[str, Any]:
    path = Path(patch_root) / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "wt-data-gap-patch" or manifest.get("version") != 1:
        raise ValueError(f"Unsupported patch manifest: {path}")
    return manifest


def verify_patch_files(patch_root: os.PathLike[str] | str) -> dict[str, Any]:
    patch = Path(patch_root).resolve()
    manifest = load_patch_manifest(patch)
    for entry in manifest["files"]:
        path = patch / entry["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Patch file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise ValueError(f"Patch checksum mismatch: {path}")
    return manifest


def create_patch_zip(
    patch_root: os.PathLike[str] | str,
    archive_path: os.PathLike[str] | str,
) -> str:
    patch = Path(patch_root).resolve()
    archive = Path(archive_path).resolve()
    verify_patch_files(patch)
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(item for item in patch.rglob("*") if item.is_file()):
            zf.write(path, path.relative_to(patch).as_posix())
    return sha256_file(archive)


def extract_patch_zip(
    archive_path: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    if expected_sha256 and sha256_file(archive).lower() != expected_sha256.lower():
        raise ValueError("Archive SHA-256 does not match the expected value")
    target = Path(destination).resolve()
    _require_empty_directory(target)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_target = (target / member.filename).resolve()
            if os.path.commonpath((str(target), str(member_target))) != str(target):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zf.extractall(target)
    return verify_patch_files(target)


def _splice_frames(
    production: pd.DataFrame,
    donor: pd.DataFrame,
    period: str,
    window: GapWindow,
) -> pd.DataFrame:
    time_column = _time_column(production, period)
    donor_time_column = _time_column(donor, period)
    if donor_time_column != time_column:
        donor = donor.rename(columns={donor_time_column: time_column})
    start, end = window.bounds(period)
    prod_times = production[time_column].astype("int64")
    donor_times = donor[time_column].astype("int64")
    donor_gap = donor.loc[(donor_times >= start) & (donor_times <= end)].copy()
    combined = pd.concat(
        [production.loc[prod_times < start], donor_gap, production.loc[prod_times > end]],
        ignore_index=True,
    ).sort_values(time_column, kind="stable", ignore_index=True)
    if period in BAR_PERIODS and combined[time_column].duplicated().any():
        raise ValueError("Combined bar data contains duplicate timestamps")
    return combined


def _window_from_manifest(manifest: dict[str, Any]) -> GapWindow:
    data = manifest["window"]
    return GapWindow(data["trading_date"], data["start_hhmm"], data["end_hhmm"])


def combine_patch_to_staging(
    production_root: os.PathLike[str] | str,
    patch_root: os.PathLike[str] | str,
    staging_root: os.PathLike[str] | str,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Splice manifest-listed patch files into production files in staging."""

    production = Path(production_root).resolve()
    patch = Path(patch_root).resolve()
    staging = Path(staging_root).resolve()
    _require_empty_directory(staging)
    manifest = verify_patch_files(patch)
    window = _window_from_manifest(manifest)
    runtime = runtime or _load_wt_runtime()

    merged_entries: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        production_file = production / relative
        patch_file = patch / relative
        staged_file = staging / relative
        if not production_file.is_file():
            raise FileNotFoundError(
                f"Production file is missing; refusing to substitute full donor history: {production_file}"
            )
        period = entry["period"]
        production_frame = _read_dsb(production_file, period, runtime)
        donor_frame = _read_dsb(patch_file, period, runtime)
        combined = _splice_frames(production_frame, donor_frame, period, window)
        _write_dsb(
            combined,
            staged_file,
            period,
            entry["exchange"],
            entry["contract"],
            runtime,
        )
        merged_entries.append(
            {
                **entry,
                "production_rows": int(len(production_frame)),
                "staged_rows": int(len(combined)),
                "staged_sha256": sha256_file(staged_file),
            }
        )

    merge_manifest = {
        "format": "wt-data-gap-merge",
        "version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "production_root": str(production),
        "patch_root": str(patch),
        "window": manifest["window"],
        "files": merged_entries,
    }
    (staging / MERGE_MANIFEST_NAME).write_text(
        json.dumps(merge_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return merge_manifest


def _canonical_frame(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    result = frame.copy()
    time_column = _time_column(result, period)
    result = result.sort_values(time_column, kind="stable", ignore_index=True)
    return result.reindex(sorted(result.columns), axis=1)


def _assert_same_frame(left: pd.DataFrame, right: pd.DataFrame, context: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(f"Validation failed for {context}: {exc}") from exc


def validate_staging(
    production_root: os.PathLike[str] | str,
    patch_root: os.PathLike[str] | str,
    staging_root: os.PathLike[str] | str,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that only the requested interval changed in every staged file."""

    production = Path(production_root).resolve()
    patch = Path(patch_root).resolve()
    staging = Path(staging_root).resolve()
    manifest = verify_patch_files(patch)
    merge_manifest = json.loads((staging / MERGE_MANIFEST_NAME).read_text(encoding="utf-8"))
    window = _window_from_manifest(manifest)
    runtime = runtime or _load_wt_runtime()

    if {entry["relative_path"] for entry in manifest["files"]} != {
        entry["relative_path"] for entry in merge_manifest["files"]
    }:
        raise ValueError("Patch and merge manifests list different files")

    checked: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        period = entry["period"]
        prod = _read_dsb(production / relative, period, runtime)
        donor = _read_dsb(patch / relative, period, runtime)
        staged = _read_dsb(staging / relative, period, runtime)
        expected = _splice_frames(prod, donor, period, window)
        _assert_same_frame(
            _canonical_frame(staged, period),
            _canonical_frame(expected, period),
            relative.as_posix(),
        )
        staged_gap = _filter_window(staged, period, window)
        donor_gap = _filter_window(donor, period, window)
        _assert_same_frame(
            _canonical_frame(staged_gap, period),
            _canonical_frame(donor_gap, period),
            f"{relative.as_posix()} repaired interval",
        )
        checked.append(
            {
                "relative_path": relative.as_posix(),
                "staged_rows": len(staged),
                "gap_rows": len(staged_gap),
            }
        )
    return {"valid": True, "files_checked": len(checked), "files": checked}


def install_staged_files(
    production_root: os.PathLike[str] | str,
    staging_root: os.PathLike[str] | str,
    backup_root: os.PathLike[str] | str,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Back up and atomically replace files listed by the merge manifest.

    Call ``validate_staging`` immediately before this function.  The production
    data collector must not be writing these files during installation.
    """

    production = Path(production_root).resolve()
    staging = Path(staging_root).resolve()
    backup = Path(backup_root).resolve()
    manifest = json.loads((staging / MERGE_MANIFEST_NAME).read_text(encoding="utf-8"))
    runtime = runtime or _load_wt_runtime()
    # Installation revalidates rather than trusting that a separate earlier
    # command was run or that files remained unchanged afterward.
    validation = validate_staging(
        production,
        manifest["patch_root"],
        staging,
        runtime=runtime,
    )
    _require_empty_directory(backup)
    # Complete every existence/checksum preflight before changing the first
    # production file, avoiding a preventable partial installation.
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        source = staging / relative
        destination = production / relative
        if not source.is_file() or not destination.is_file():
            raise FileNotFoundError(f"Missing staged or production file: {relative}")
        if sha256_file(source) != entry["staged_sha256"]:
            raise ValueError(f"Staged file checksum mismatch: {relative}")

    installed: list[str] = []
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        source = staging / relative
        destination = production / relative
        backup_file = backup / relative
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup_file)
        temporary = destination.with_name(destination.name + ".gap-repair-new")
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Temporary installation copy failed checksum: {relative}")
        os.replace(temporary, destination)
        installed.append(relative.as_posix())
    return {
        "installed": len(installed),
        "files": installed,
        "backup_root": str(backup),
        "validation": validation,
    }


def merge_patch(
    production_root: os.PathLike[str] | str,
    patch_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    *,
    backup_root: os.PathLike[str] | str | None = None,
    confirm_in_place: bool = False,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge and validate a patch, either to a new folder or in place.

    When ``output_root`` is the production root, a backup directory and an
    explicit confirmation are required.  A temporary staging directory is
    used so production files are never rewritten while they are being read.
    """

    production = Path(production_root).resolve()
    output = Path(output_root).resolve()
    runtime = runtime or _load_wt_runtime()
    if output != production:
        merge_manifest = combine_patch_to_staging(
            production, patch_root, output, runtime=runtime
        )
        validation = validate_staging(
            production, patch_root, output, runtime=runtime
        )
        return {
            "mode": "new_folder",
            "output_root": str(output),
            "files": len(merge_manifest["files"]),
            "validation": validation,
        }

    if not confirm_in_place:
        raise ValueError("In-place merge requires confirm_in_place=True")
    if backup_root is None:
        raise ValueError("In-place merge requires a backup_root")
    backup = Path(backup_root).resolve()
    if backup == production or production in backup.parents:
        raise ValueError("Backup directory must be outside the production store")

    production.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wt-gap-merge-", dir=str(production.parent)
    ) as temporary:
        staging = Path(temporary)
        combine_patch_to_staging(production, patch_root, staging, runtime=runtime)
        result = install_staged_files(
            production,
            staging,
            backup,
            runtime=runtime,
        )
    return {"mode": "in_place", "output_root": str(production), **result}


def _parse_hhmm(value: str) -> int:
    if len(value) != 4 or not value.isdigit():
        raise argparse.ArgumentTypeError("HHMM must contain exactly four digits")
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Extract a small gap patch from local DSBs")
    export.add_argument("--source", required=True)
    export.add_argument("--patch", required=True)
    export.add_argument("--date", required=True, type=int)
    export.add_argument("--start", default="0900", type=_parse_hhmm)
    export.add_argument("--end", default="0925", type=_parse_hhmm)
    export.add_argument("--exchanges", nargs="+", default=list(DEFAULT_EXCHANGES))
    export.add_argument("--periods", nargs="+", choices=ALL_PERIODS, default=list(ALL_PERIODS))
    export.add_argument(
        "--scan-all-bars",
        action="store_true",
        help="Scan expired bar files too; slower, but ignores file modification dates",
    )

    dump = subparsers.add_parser(
        "dump", help="Dump only the requested gap records into one small folder"
    )
    dump.add_argument("--source", required=True)
    dump.add_argument("--output", required=True)
    dump.add_argument("--date", required=True, type=int)
    dump.add_argument("--start", default="0900", type=_parse_hhmm)
    dump.add_argument("--end", default="0925", type=_parse_hhmm)
    dump.add_argument("--exchanges", nargs="+", default=list(DEFAULT_EXCHANGES))
    dump.add_argument("--periods", nargs="+", choices=ALL_PERIODS, default=list(ALL_PERIODS))
    dump.add_argument("--scan-all-bars", action="store_true")

    zip_command = subparsers.add_parser("zip", help="Verify and zip a patch directory")
    zip_command.add_argument("--patch", required=True)
    zip_command.add_argument("--output", required=True)

    unzip = subparsers.add_parser("unzip", help="Verify checksum and safely unpack a patch")
    unzip.add_argument("--archive", required=True)
    unzip.add_argument("--output", required=True)
    unzip.add_argument("--sha256")

    combine = subparsers.add_parser("combine", help="Merge a patch into a staging directory")
    combine.add_argument("--production", required=True)
    combine.add_argument("--patch", required=True)
    combine.add_argument("--staged", required=True)

    validate = subparsers.add_parser("validate", help="Validate staged data against both inputs")
    validate.add_argument("--production", required=True)
    validate.add_argument("--patch", required=True)
    validate.add_argument("--staged", required=True)

    install = subparsers.add_parser("install", help="Back up and install validated staged files")
    install.add_argument("--production", required=True)
    install.add_argument("--staged", required=True)
    install.add_argument("--backup", required=True)
    install.add_argument(
        "--confirm",
        action="store_true",
        help="Required confirmation that the collector is stopped and validation passed",
    )

    merge = subparsers.add_parser(
        "merge", help="Merge and validate a dumped patch in one operation"
    )
    merge.add_argument("--production", required=True)
    merge.add_argument("--patch", required=True)
    merge.add_argument(
        "--output",
        required=True,
        help="New result folder, or the production folder for an in-place merge",
    )
    merge.add_argument("--backup", help="Required only when output is production")
    merge.add_argument(
        "--confirm-in-place",
        action="store_true",
        help="Required when output is the production folder",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        result = export_gap_patch(
            args.source,
            args.patch,
            GapWindow(args.date, args.start, args.end),
            exchanges=args.exchanges,
            periods=args.periods,
            only_bar_files_modified_on_date=not args.scan_all_bars,
        )
    elif args.command == "dump":
        result = export_gap_patch(
            args.source,
            args.output,
            GapWindow(args.date, args.start, args.end),
            exchanges=args.exchanges,
            periods=args.periods,
            only_bar_files_modified_on_date=not args.scan_all_bars,
        )
    elif args.command == "zip":
        digest = create_patch_zip(args.patch, args.output)
        result = {"archive": str(Path(args.output).resolve()), "sha256": digest}
    elif args.command == "unzip":
        result = extract_patch_zip(args.archive, args.output, expected_sha256=args.sha256)
    elif args.command == "combine":
        result = combine_patch_to_staging(args.production, args.patch, args.staged)
    elif args.command == "validate":
        result = validate_staging(args.production, args.patch, args.staged)
    elif args.command == "install":
        if not args.confirm:
            raise SystemExit(
                "Refusing installation without --confirm. Stop the collector and run validate first."
            )
        result = install_staged_files(args.production, args.staged, args.backup)
    elif args.command == "merge":
        result = merge_patch(
            args.production,
            args.patch,
            args.output,
            backup_root=args.backup,
            confirm_in_place=args.confirm_in_place,
        )
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
