import os
import re
import shutil
from wtpy.wrapper import WtDataHelper
from pycmqlib3.utility.process_wt_data import *
import glob

# If you don't already have these in scope:
dtHelper = WtDataHelper()

def period_to_pkey(period: str) -> str:
    """Map folder period name to save_bars_to_dsb period key."""
    return 'd' if period == 'day' else (period[0] + period[-1])  # "min1"->"m1", "min5"->"m5"


def copy_to_out(infile: str, out_folder: str, period: str, exch: str, new_contract: str) -> str:
    """
    Create out dir {out_folder}/{period}/{exch}, and copy infile to {new_contract}.dsb.
    Returns destination path.
    """
    outdir = os.path.join(out_folder, period, exch)
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"{new_contract}.dsb")
    shutil.copy2(infile, outfile)
    print(f"[COPY] {os.path.basename(infile)} -> {new_contract}.dsb")
    return outfile


def process_czce_folder(src_folder: str,
                        out_folder: str,
                        periods = ('day', 'min1', 'min5'),
                        exch: str = 'CZCE'):
    """
    Walk {src_folder}/{period}/{exch}, and:
      - For 4-digit YYMM:
          * 2509–2608: drop leading '2' (becomes 3-digit) then handle as split range
          * <=1412: copy unchanged (same name) to out folder
      - For 3-digit YMM:
          * <=412  -> copy as XX2YMM
          * >=609  -> copy as XX1YMM
          * 501–608 -> READ, preprocess, SPLIT at 2020-01-01, SAVE as XX1YMM (pre-2020) and XX2YMM (>=2020)
    """
    for period in periods:
        p_key = period_to_pkey(period)
        is_day = (period == 'day')
        in_dir = os.path.join(src_folder, period, exch)
        if not os.path.isdir(in_dir):
            print(f"[WARN] Missing folder: {in_dir}")
            continue

        for fname in os.listdir(in_dir):
            if not fname.endswith('.dsb'):
                continue

            infile = os.path.join(in_dir, fname)
            contract = fname[:-4]  # strip .dsb
            m = re.match(r'^([A-Za-z]{2})(\d{3,4})$', contract)
            if not m:
                print(f"[SKIP] Unexpected name format: {fname}")
                continue

            prefix, digits = m.groups()

            # ----- 4-digit YYMM handling -----
            if len(digits) == 4:
                d4 = int(digits)
                if 2509 <= d4 <= 2608:
                    # Normalize by removing the leading '2' (e.g., 2509 -> 509)
                    digits = digits[1:]
                    # fall through to 3-digit logic below
                elif d4 <= 1412:
                    # Copy unchanged (same contract name into out folder)
                    copy_to_out(infile, out_folder, period, exch, contract)
                    continue
                else:
                    print(f"[WARN] 4-digit out-of-scope: {contract}")
                    continue

            # ----- 3-digit YMM handling (original or normalized) -----
            d3 = int(digits)

            # Case A: <= 412 -> XX2YMM (copy)
            if d3 <= 412:
                new_contract = f"{prefix}2{digits}"
                copy_to_out(infile, out_folder, period, exch, new_contract)
                continue

            # Case B: >= 609 -> XX1YMM (copy)
            if d3 >= 609:
                new_contract = f"{prefix}1{digits}"
                copy_to_out(infile, out_folder, period, exch, new_contract)
                continue

            # Case C: 501–608 -> split by date 2020-01-01
            if 501 <= d3 <= 608:
                # Read the .dsb
                print(infile)
                bar_df = dtHelper.read_dsb_bars(infile, isDay=is_day).to_df()

                # --- preprocessing before save ---
                # rename columns and adjust time
                rename_map = {"money": "turnover", "hold": "open_interest", "bartime": "time"}
                cols_to_rename = {k: v for k, v in rename_map.items() if k in bar_df.columns}
                if cols_to_rename:
                    bar_df = bar_df.rename(columns=cols_to_rename)
                if "time" in bar_df.columns:
                    if is_day:
                        bar_df["time"] = 0
                    else:
                        bar_df["time"] = bar_df["time"] - 199000000000

                # ensure integer date
                if "date" in bar_df.columns:
                    bar_df["date"] = bar_df["date"].astype("int64")
                else:
                    print(f"[WARN] No 'date' column in {contract}, skipping split.")
                    continue

                cutoff = 20200101
                pre_df  = bar_df[bar_df["date"] <  cutoff]
                post_df = bar_df[bar_df["date"] >= cutoff]

                # Build out folder once
                outdir = os.path.join(out_folder, period, exch)
                os.makedirs(outdir, exist_ok=True)

                # Save each side using your wrapper (creates {outdir}/{contract}.dsb)
                pre_contract  = f"{prefix}1{digits}"   # pre-2020 -> 201x
                post_contract = f"{prefix}2{digits}"   # 2020+  -> 202x

                # Map period string to p_key for saving
                # (We already computed p_key above)
                if not pre_df.empty:
                    save_bars_to_dsb(pre_df, pre_contract,  folder_loc=outdir, period=p_key)
                    print(f"[SPLIT] {contract} pre-2020 -> {pre_contract}.dsb ({len(pre_df)})")
                else:
                    print(f"[SPLIT] {contract} pre-2020 -> (no rows)")

                if not post_df.empty:
                    save_bars_to_dsb(post_df, post_contract, folder_loc=outdir, period=p_key)
                    print(f"[SPLIT] {contract} 2020+    -> {post_contract}.dsb ({len(post_df)})")
                else:
                    print(f"[SPLIT] {contract} 2020+    -> (no rows)")

                continue

            print(f"[WARN] 3-digit out-of-scope: {contract}")


def rename_files(start_dir):
    for dirpath, dirnames, filenames in os.walk(start_dir):
        for fname in filenames:
            if not fname.endswith('.dsb'):
                continue

            infile = os.path.join(dirpath, fname)
            contract = fname[:-4]  # strip .dsb
            m = re.match(r'^([A-Z]{2})(\d{3})$', contract)
            if not m:
                print(f"[SKIP] Unexpected name format: {fname}")
                continue

            prefix, digits = m.groups()
            new_fname = f"{prefix}2{digits}.dsb"    
            new_path = os.path.join(dirpath, new_fname)
            print(f"Renaming '{dirpath}' to '{new_path}'")            
            try:
                # Perform the actual rename operation.
                os.rename(infile, new_path)
            except OSError as e:
                print(f"Error: Could not rename file {infile}. Reason: {e}")


if __name__ == "__main__":
    src_folder = r"c:/dev/wtdev/storage/his"
    out_folder = r"c:/dev/data/his_out"
    periods = ['day', 'min1', 'min5']
    print("please check if this need to be run!")
    #process_czce_folder(src_folder, out_folder, periods, exch='CZCE')
