import json
import csv
import os
import sys
sys.path.insert(0, r'D:\Vasanth\RL_CYB')
from stable_baselines3 import DQN
import eval_dqn_3x

MODEL_PATH = r'D:\Vasanth\RL_CYB\ppo_out\dqn_3x_v1.zip'

# Load Q metadata
with open('q_metadata.json') as f:
    qmeta = json.load(f)

# Load episode-layer JSONs for sightings
with open('cyborg_3x_episode_meander_20260215_142036_4.5.json') as f:
    meander_layer = json.load(f)
with open('cyborg_3x_episode_bline_20260215_142036_4.5.json') as f:
    bline_layer = json.load(f)

# Helper to get sightings by techniqueID
def sightings(layer, techid):
    for t in layer.get('techniques', []):
        if t.get('techniqueID') == techid:
            # look in metadata first
            meta = t.get('metadata', [])
            for m in meta:
                if m.get('name') == 'sightings':
                    return int(m.get('value'))
            # fallback: try comment 'Observed N'
            c = t.get('comment','')
            if 'Observed' in c:
                import re
                m = re.search(r'Observed (\d+) times', c)
                if m:
                    return int(m.group(1))
            return 0
    return 0

# Techniques of interest (observed in episodes)
techs = [
    ('T1210','Exploitation of Remote Services'),
    ('T1068','Exploitation for Privilege Escalation'),
    ('T1046','Network Service Discovery'),
    ('T1018','Software Discovery')
]

# Run small evaluation to get empirical action counts (uses eval_dqn_3x.evaluate_vs_red)
model = DQN.load(MODEL_PATH)
empirical = {}
for red in ['meander','bline']:
    m, s, type_counts, action_counts = eval_dqn_3x.evaluate_vs_red(model, red, n_episodes=10, max_steps=100, deterministic=True, verbose=False)
    # action_counts is defaultdict(int)
    # compute per-host patch/restore counts
    patch_counts = {}
    restore_counts = {}
    total_actions = sum(action_counts.values())
    for action_name, cnt in action_counts.items():
        parts = action_name.split()
        # expect format 'Patch Op_Host1' or 'Restore Op_Server0'
        if len(parts) >= 2:
            a_type = parts[0]
            host = ' '.join(parts[1:])
            if a_type == 'Patch':
                patch_counts[host] = patch_counts.get(host,0)+cnt
            if a_type == 'Restore':
                restore_counts[host] = restore_counts.get(host,0)+cnt
    empirical[red] = {'patch_counts': patch_counts, 'restore_counts': restore_counts, 'total_actions': total_actions}

# Write technique summary CSV
tech_csv = 'policy_technique_summary.csv'
with open(tech_csv,'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['techniqueID','techniqueName','sightings_meander','sightings_bline','preferred_Q_action','preferred_Q_Q','empirical_top_action','empirical_top_action_freq','layer_color'])
    # preferred Q action: extract top_actions[0] from qmeta['bline'] (representative)
    b_top = qmeta['bline']['top_actions'][0]
    empirical_top = None
    # empirical top action (from evaluation): the action with max count across action_counts for bline
    best_action = None
    best_cnt = 0
    for a,c in eval_dqn_3x.evaluate_vs_red(model,'bline',n_episodes=10,max_steps=100,deterministic=True,verbose=False)[3].items():
        if c>best_cnt:
            best_cnt=c; best_action=a
    empirical_top = f"{best_action} ({best_cnt}/{1000})" if best_action else ''

    for tid, name in techs:
        s_me = sightings(meander_layer, tid)
        s_b = sightings(bline_layer, tid)
        # find preferred Q action in qmeta top_actions list
        pref = ''
        pref_q = ''
        for entry in qmeta['bline']['top_actions']:
            if entry['action'].startswith('Patch'):
                pref = entry['action']
                pref_q = f"{entry['Q']:.3f}"
                break
        # layer color from bline layer
        color = ''
        for t in bline_layer.get('techniques',[]):
            if t.get('techniqueID')==tid:
                color = t.get('color','')
        writer.writerow([tid,name,s_me,s_b,pref,pref_q,empirical_top,color])

# Write host-level Q summary CSV (Patch/Restore Q and empirical counts)
host_csv = 'policy_host_q_summary.csv'
# Canonical ALL_HOSTS ordering
from train_dqn_3x import ALL_HOSTS
with open(host_csv,'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['host','patch_Q_bline','restore_Q_bline','patch_count_bline','restore_count_bline','patch_Q_meander','restore_Q_meander','patch_count_meander','restore_count_meander'])
    b_patch_q = qmeta['bline']['per_host_q']['Patch']
    b_restore_q = qmeta['bline']['per_host_q']['Restore']
    m_patch_q = qmeta['meander']['per_host_q']['Patch']
    m_restore_q = qmeta['meander']['per_host_q']['Restore']
    for i,host in enumerate(ALL_HOSTS):
        b_patch = b_patch_q[i]
        b_restore = b_restore_q[i]
        m_patch = m_patch_q[i]
        m_restore = m_restore_q[i]
        b_patch_cnt = empirical['bline']['patch_counts'].get(host,0)
        b_restore_cnt = empirical['bline']['restore_counts'].get(host,0)
        m_patch_cnt = empirical['meander']['patch_counts'].get(host,0)
        m_restore_cnt = empirical['meander']['restore_counts'].get(host,0)
        writer.writerow([host,f"{b_patch:.3f}",f"{b_restore:.3f}",b_patch_cnt,b_restore_cnt,f"{m_patch:.3f}",f"{m_restore:.3f}",m_patch_cnt,m_restore_cnt])

print('Wrote',tech_csv,'and',host_csv)