import pandas as pd


# ----------------------------
# Cyber Event Reward Shaping
# ----------------------------
def cyber_event_reward(obs_text, next_obs_text, info_text, step_cost=-0.1):
    """
    Reward shaping based on cybersecurity events extracted from text logs.

    Positive events: detect/alert, block/quarantine, patch
    Negative events: compromise, exfiltration, exploit, credential activity
    """

    s = f"{obs_text} {next_obs_text} {info_text}".lower()

    # --- base step cost ---
    reward = step_cost

    # --- event flags (booleans) ---
    ev_detect = ("detect" in s) or ("alert" in s) or ("alarm" in s) or ("ids" in s) or ("suspicious" in s) or ("anomal" in s)
    ev_block  = ("block" in s) or ("blocked" in s) or ("deny" in s) or ("dropped" in s) or ("firewall" in s) or ("isolat" in s) or ("quarantine" in s)
    ev_patch  = ("patch" in s) or ("patched" in s) or ("update" in s)

    ev_compromise = ("compromise" in s) or ("compromised" in s) or ("owned" in s) or ("root" in s) or ("breach" in s) or ("session" in s)
    ev_exploit    = ("exploit" in s) or ("payload" in s) or ("rce" in s) or ("execute" in s) or ("injection" in s)
    ev_exfil      = ("exfil" in s) or ("exfiltration" in s) or ("steal" in s) or ("leak" in s)
    ev_cred       = ("credential" in s) or ("password" in s) or ("hash" in s) or ("login" in s) or ("bruteforce" in s)
    
   
    # --- apply shaped reward ---
    # Defender good outcomes
    if ev_detect:
        reward += 2.0
    if ev_block:
        reward += 5.0
    if ev_patch:
        reward += 3.0

    # Attacker bad outcomes
    if ev_exploit:
        reward -= 3.0
    if ev_cred:
        reward -= 2.0
    if ev_compromise:
        reward -= 10.0
    if ev_exfil:
        reward -= 20.0

    return reward, ev_detect, ev_block, ev_patch, ev_exploit, ev_cred, ev_compromise, ev_exfil


def main():
    # ✅ input dataset (your Scenario1b file)
    in_csv = "cyborg_rl_dataset_scenario1b.csv"
    out_csv = "cyborg_rl_dataset_scenario1b_shaped.csv"

    df = pd.read_csv(in_csv)

    # Fill missing text fields safely
    df["obs_json"] = df["obs_json"].fillna("")
    df["next_obs_json"] = df["next_obs_json"].fillna("")
    df["info_json"] = df["info_json"].fillna("")

    shaped = df.apply(
        lambda row: cyber_event_reward(row["obs_json"], row["next_obs_json"], row["info_json"]),
        axis=1
    )

    # Expand tuple columns
    df["reward_shaped"] = [x[0] for x in shaped]
    df["ev_detect"] = [x[1] for x in shaped]
    df["ev_block"] = [x[2] for x in shaped]
    df["ev_patch"] = [x[3] for x in shaped]
    df["ev_exploit"] = [x[4] for x in shaped]
    df["ev_cred"] = [x[5] for x in shaped]
    df["ev_compromise"] = [x[6] for x in shaped]
    df["ev_exfil"] = [x[7] for x in shaped]

    # Save
    df.to_csv(out_csv, index=False)

    # Quick summary
    print("\n✅ Saved:", out_csv)
    print("Rows:", len(df))
    print("\n--- Event Counts ---")
    print("Detect:", int(df["ev_detect"].sum()))
    print("Block :", int(df["ev_block"].sum()))
    print("Patch :", int(df["ev_patch"].sum()))
    print("Exploit:", int(df["ev_exploit"].sum()))
    print("Credential:", int(df["ev_cred"].sum()))
    print("Compromise:", int(df["ev_compromise"].sum()))
    print("Exfil:", int(df["ev_exfil"].sum()))

    print("\n--- Reward stats ---")
    print("Original reward mean:", df["reward"].mean())
    print("Shaped reward mean  :", df["reward_shaped"].mean())
    print("Shaped reward min/max:", df["reward_shaped"].min(), df["reward_shaped"].max())


if __name__ == "__main__":
    main()
