import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from dataclasses import dataclass

def read_chunk(file_path: str, chunksize: int):
    cols = ["AccessionYear", "Object Begin Date"]

    for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=cols, engine="python"):
        yield chunk

def preprocess(chunk):
    for df in chunk:
        acc = pd.to_numeric(df["AccessionYear"], errors = "coerce")
        begin = pd.to_numeric(df["Object Begin Date"], errors = "coerce")

        check = acc.notna() & begin.notna()
        if not check.any():
            continue

        acc = acc[check]
        begin = begin[check]

        def reduce_mem(data):
            acmin, acmax = data.min(), data.max()
            if acmin >= np.iinfo(np.int8).min and acmax <= np.iinfo(np.int8).max:
                return data.astype(np.int8)
            elif acmin >= np.iinfo(np.int16).min and acmax <= np.iinfo(np.int16).max:
                return data.astype(np.int16)
            else:
                return data.astype(np.int32)

        acc = reduce_mem(acc)
        begin = reduce_mem(begin)

        age = reduce_mem(acc - begin)

        decade = reduce_mem((acc // 10) * 10)

        result = pd.DataFrame({"decade": decade, "age": age})

        yield result

def aggregate(frames):
    acc_df = pd.DataFrame(columns=["n", "S", "Q"])

    for df in frames:
        age = df["age"].astype(np.float64)

        df2 = df.assign( age=age, age2=age * age)

        chunk_agg = (
            df2.groupby("decade", sort=False)
               .agg(
                   n=("age", "size"),
                   S=("age", "sum"),
                   Q=("age2", "sum"),
               )
        )

        acc_df = acc_df.add(chunk_agg, fill_value=0)

    return acc_df

def finalize(df: pd.DataFrame):

    n = df["n"].astype(float)
    S = df["S"].astype(float)
    Q = df["Q"].astype(float)

    mean = S / n

    var = (Q - (S**2) / n) / (n - 1)
    var = var.where(n > 1, 0)
    var = var.clip(lower=0)

    sd = var.pow(0.5)

    t_vals = pd.Series(
        scipy.stats.t.ppf(0.975, n - 1),
        index=n.index
    ).where(n > 1, 0)

    ci_half = t_vals * sd / n.pow(0.5)
    scatter_half = t_vals * sd

    result = pd.DataFrame({
        "n": n,
        "mean_age": mean,
        "sd": sd,
        "ci_low": mean - ci_half,
        "ci_high": mean + ci_half,
        "scatter_low": mean - scatter_half,
        "scatter_high": mean + scatter_half,
    })

    result["delta_mean_age"] = result["mean_age"].diff()

    return result.reset_index()


def plot(res: pd.DataFrame):
    x = res["decade"].to_numpy()
    mean = res["mean_age"].to_numpy()
    ci_half = (res["ci_high"] - res["ci_low"]).to_numpy() / 2
    scatter_low = np.maximum(res["scatter_low"].to_numpy(), 0)
    scatter_high = res["scatter_high"].to_numpy()

    yerr_scatter = [
        mean - scatter_low,
        scatter_high - mean
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(x, mean, width=8, label="Средний возраст")
    plt.errorbar(x, mean, yerr=yerr_scatter, fmt="none", ecolor="lightgray", elinewidth=6, label="95% интервал рассеяния")
    plt.errorbar(x, mean, yerr=ci_half, fmt="none", ecolor="black", capsize=3, label="95% ДИ среднего")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="black", linewidth=1)
    plt.bar(x, res["delta_mean_age"].to_numpy(), width=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.show()

def read_chunk_categorical(file_path: str, chunksize: int, fields: list[str]):
    for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=fields, low_memory=False):
        yield chunk

def aggregate_categorical(chunks, fields: list[str]) -> pd.DataFrame:
    total_df = pd.DataFrame()

    for chunk in chunks:
        chunk_result = pd.DataFrame()

        for field in fields:
            counts = (
                chunk[field]
                .dropna()
                .astype(str)
                .value_counts()
                .rename(field)
            )

            chunk_result = pd.concat([chunk_result, counts], axis=1)

        if total_df.empty:
            total_df = chunk_result
        else:
            total_df = total_df.add(chunk_result, fill_value=0)

    return total_df.fillna(0)

def finalize_categorical(counters: pd.DataFrame) -> pd.DataFrame:
    gini_series = pd.Series(dtype=float)
    entropy_series = pd.Series(dtype=float)
    enc_norm_series = pd.Series(dtype=float)
    

    for field in counters.columns:
        counts = counters[field]
        counts = counts[counts > 0]

        total = counts.sum()

        if total == 0:
            continue
        p = counts / total
        k = len(p)

        gini = 1 - (p.pow(2).sum())

        if k > 1:
            entropy_raw = -(p * np.log2(p)).sum()
            entropy = entropy_raw / np.log2(k)
        else:
            entropy = 0

        enc = 1 / p.pow(2).sum()
        enc_norm = enc / k if k > 0 else 0

        gini_series[field] = gini
        entropy_series[field] = entropy
        enc_norm_series[field] = enc_norm

    return pd.DataFrame({
        "Gini": gini_series,
        "Entropy": entropy_series,
        "ENC_norm": enc_norm_series
    })

def plot_categorical_heatmap(metrics_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        metrics_df, 
        annot=True, 
        cmap='coolwarm' 
    )

    plt.title('Метрики качества категориальных полей')
    plt.tight_layout()
    plt.show()

def main(file_path: str, chunksize: int):
    dataframe = read_chunk(file_path, chunksize)
    frames = preprocess(dataframe)
    acc = aggregate(frames)
    res = finalize(acc)
    fields_to_analyze = ['Department', 'Culture', 'Medium', 'Classification', 'Country']
    
    categorical_chunks = read_chunk_categorical(file_path, chunksize, fields_to_analyze)
    aggregated_counts = aggregate_categorical(categorical_chunks, fields_to_analyze)
    metrics_df = finalize_categorical(aggregated_counts)

    plot(res)
    plot_categorical_heatmap(metrics_df)

if __name__ == "__main__":
    main("/Users/krovich/Desktop/6301kroviakovds/data/MetObjects.csv", 20000)

