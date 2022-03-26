# -*- coding: utf-8 -*-
"""
yimingxiao
"""
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error


def _eval(csv):
    df = pd.read_csv(csv)
    mos = df['mos']
    mos_pred = df['mos_pred']
    pccs = np.corrcoef(mos, mos_pred)[0][1]
    rmse = np.sqrt(mean_squared_error(mos, mos_pred))
    SROCC = stats.spearmanr(mos_pred, mos)[0]
    print(round(pccs,4))
    print(round(SROCC,4))
    print(round(rmse,4))

def eval():
    from pathlib import Path
    script_dir = Path(__file__).parent
    in_dir = script_dir.joinpath("eval_input")

    for subdir in in_dir.iterdir():
        paths = list(subdir.iterdir())
        for csv_path in sorted(paths):
            print(f"Processing {csv_path}")
            _eval(str(csv_path))
            print("============================")

if __name__ == "__main__":
    eval()