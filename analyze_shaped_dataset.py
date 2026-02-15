import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score


def main():
    in_csv = "cyborg_rl_dataset_scenario1b_shaped.csv"

    print("Loading:", in_csv)
    df = pd.read_csv(in_csv)

    print("\n--- Basic stats ---")
    print("Rows:", len(df))
    print("Episodes:", df["episode"].nunique())
    print("Unique actions:", df["action_name"].nunique())
    print("Reward_shaped mean:", df["reward_shaped"].mean())
    print("Reward_shaped std :", df["reward_shaped"].std())
    print("Reward_shaped min/max:", df["reward_shaped"].min(), df["reward_shaped"].max())

    # ----------------------------
    # C1) Reward histogram plot
    # ----------------------------
    plt.figure()
    plt.hist(df["reward_shaped"].values, bins=40)
    plt.title("Shaped Reward Distribution (Scenario1b)")
    plt.xlabel("reward_shaped")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("shaped_reward_hist.png", dpi=160)
    plt.close()
    print("✅ Saved: shaped_reward_hist.png")

    # ----------------------------
    # C2) Episode returns plot
    # ----------------------------
    ep_returns = df.groupby("episode")["reward_shaped"].sum().reset_index(name="episode_return_shaped")

    plt.figure()
    plt.plot(ep_returns["episode"], ep_returns["episode_return_shaped"])
    plt.title("Episode Return (Shaped Reward) - Scenario1b")
    plt.xlabel("episode")
    plt.ylabel("episode_return_shaped")
    plt.tight_layout()
    plt.savefig("shaped_episode_return.png", dpi=160)
    plt.close()
    print("✅ Saved: shaped_episode_return.png")

    # ----------------------------
    # C3) Action vs Event correlation table
    # ----------------------------
    group = df.groupby("action_name").agg(
        count=("reward_shaped", "count"),
        mean_reward_shaped=("reward_shaped", "mean"),
        pct_compromise=("ev_compromise", "mean"),
        pct_credential=("ev_cred", "mean"),
        pct_exploit=("ev_exploit", "mean") if "ev_exploit" in df.columns else ("reward_shaped", lambda x: 0),
        pct_detect=("ev_detect", "mean") if "ev_detect" in df.columns else ("reward_shaped", lambda x: 0),
    ).reset_index()

    group = group.sort_values(by="pct_compromise", ascending=False)
    group.to_csv("action_vs_events.csv", index=False)
    print("✅ Saved: action_vs_events.csv")

    print("\nTop actions associated with compromise:")
    print(group[["action_name", "count", "pct_compromise", "pct_credential", "mean_reward_shaped"]].head(10))

    # ----------------------------
    # Features for baseline ML
    # ----------------------------
    # simple, safe features
    feature_cols = [
        "timestep",
        "action_name",
        "obs_success",
        "next_obs_success",
        "obs_len",
        "next_obs_len",
        "info_len",
    ]

    # keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()

    # one-hot encode action_name
    categorical = ["action_name"] if "action_name" in X.columns else []
    numeric = [c for c in feature_cols if c not in categorical]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    # ----------------------------
    # B1) Reward regression baseline
    # ----------------------------
    y_reward = df["reward_shaped"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reward, test_size=0.2, random_state=42
    )

    reward_model = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", Ridge(alpha=1.0, random_state=42))
        ]
    )

    reward_model.fit(X_train, y_train)
    y_pred = reward_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Reward Model (Ridge) ---")
    print("MAE:", mae)
    print("R2 :", r2)

    # ----------------------------
    # B2) Compromise classifier baseline
    # ----------------------------
    if "ev_compromise" in df.columns and df["ev_compromise"].nunique() >= 2:
        y_comp = df["ev_compromise"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_comp, test_size=0.2, random_state=42, stratify=y_comp
        )

        comp_model = Pipeline(
            steps=[
                ("prep", preprocess),
                ("model", LogisticRegression(max_iter=2000))
            ]
        )

        comp_model.fit(X_train, y_train)
        prob = comp_model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)

        print("\n--- Compromise Model (LogReg) ---")
        print("Accuracy:", acc)
        print("ROC-AUC :", auc)
    else:
        print("\n⚠️ Compromise classifier skipped (ev_compromise missing or only one class).")


if __name__ == "__main__":
    main()
