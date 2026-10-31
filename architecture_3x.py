"""
3× CybORG Architecture Diagram
================================
Generates a comprehensive architecture diagram showing:
  - 5-subnet network topology (20 hosts)
  - Red agent attack paths and policies
  - Blue agent defense actions and DQN policy
  - Green agent benign traffic patterns
  - NACL firewall rules between subnets
  - Reward structure
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np

fig = plt.figure(figsize=(28, 22))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
ax.set_xlim(-1, 27)
ax.set_ylim(-2, 22)
ax.set_aspect('equal')
ax.axis('off')

# ===== TITLE =====
fig.suptitle('3× Scaled CybORG Architecture — Agent Interactions & Policies',
             fontsize=22, fontweight='bold', y=0.97, color='#1a1a2e')
ax.text(13, 21.3, '20 Hosts  ·  5 Subnets  ·  3 Agent Roles  ·  120 Action Space  ·  121-dim Observation',
        ha='center', va='center', fontsize=11, color='#555',
        fontstyle='italic')

# ===== COLORS =====
SUBNET_COLORS = {
    'User':       ('#e3f2fd', '#1565c0'),  # light blue bg, dark blue border
    'Enterprise': ('#fff3e0', '#e65100'),  # light orange bg, dark orange border
    'Operational':('#fce4ec', '#b71c1c'),  # light red bg, dark red border
    'DMZ':        ('#e8f5e9', '#2e7d32'),  # light green bg, dark green border
    'Research':   ('#f3e5f5', '#6a1b9a'),  # light purple bg, dark purple border
}
RED_COLOR = '#d32f2f'
BLUE_COLOR = '#1565c0'
GREEN_COLOR = '#2e7d32'

# ===== HELPER FUNCTIONS =====
def draw_subnet_box(ax, x, y, w, h, name, bg, border, hosts, host_images):
    """Draw a subnet box with hosts inside."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=bg, edgecolor=border, linewidth=2.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.35, name, ha='center', va='center',
            fontsize=13, fontweight='bold', color=border,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor=border))

    # Draw hosts
    positions = []
    n = len(hosts)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    host_w, host_h = 1.6, 0.7
    start_x = x + (w - cols * (host_w + 0.25)) / 2 + 0.1
    start_y = y + h - 1.3

    for i, (host, img) in enumerate(zip(hosts, host_images)):
        row = i // cols
        col = i % cols
        hx = start_x + col * (host_w + 0.25)
        hy = start_y - row * (host_h + 0.3)

        # Host box
        os_color = '#e3f2fd' if 'windows' in img.lower() or 'win' in img.lower() else '#e8f5e9'
        if 'server' in img.lower() or 'internal' in img.lower() or 'gateway' in img.lower():
            os_color = '#fff8e1'
        if 'velociraptor' in img.lower():
            os_color = '#fce4ec'

        hbox = FancyBboxPatch((hx, hy), host_w, host_h, boxstyle="round,pad=0.08",
                              facecolor=os_color, edgecolor='#555', linewidth=1)
        ax.add_patch(hbox)

        # Host name
        display_name = host.replace('_', '\n') if len(host) > 10 else host
        ax.text(hx + host_w/2, hy + host_h/2 + 0.08, display_name, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='#333')

        # OS indicator
        os_label = 'LNX' if 'linux' in img.lower() else 'WIN' if 'windows' in img.lower() else 'SRV'
        os_lbl_color = '#388e3c' if 'linux' in img.lower() else '#1565c0' if 'windows' in img.lower() else '#e65100'
        if 'server' in img.lower() or 'op_server' in img.lower():
            os_label = 'SRV'
            os_lbl_color = '#e65100'
        if 'velociraptor' in img.lower():
            os_label = 'MON'
            os_lbl_color = '#6a1b9a'
        if 'gateway' in img.lower():
            os_label = 'GW'
            os_lbl_color = '#00695c'
        ax.text(hx + host_w - 0.25, hy + 0.12, os_label, ha='center', va='center',
                fontsize=5.5, fontweight='bold', color=os_lbl_color)

        positions.append((hx + host_w/2, hy + host_h/2))

    return positions

def draw_agent_box(ax, x, y, title, color, items, w=5.5, h=None):
    """Draw an agent info box."""
    if h is None:
        h = 1.0 + len(items) * 0.4
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                         facecolor='white', edgecolor=color, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, edgecolor=color))
    for i, item in enumerate(items):
        ax.text(x + 0.3, y + h - 0.9 - i * 0.4, item, ha='left', va='center',
                fontsize=8.5, color='#333', fontstyle='italic' if item.startswith('→') else 'normal')


# ===== DRAW SUBNETS =====

# User Subnet (bottom-left)
user_hosts = ['User0', 'User1', 'User2', 'User3', 'User4', 'User5', 'User6']
user_images = ['windows_user', 'windows_user', 'windows_user', 'linux_user',
               'linux_user', 'windows_user', 'linux_user']
user_pos = draw_subnet_box(ax, 0.5, 3.5, 8, 4, 'USER SUBNET  (7 hosts)',
                           *SUBNET_COLORS['User'], user_hosts, user_images)

# Enterprise Subnet (center)
ent_hosts = ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender']
ent_images = ['Gateway', 'Internal', 'Internal', 'Velociraptor_Server']
ent_pos = draw_subnet_box(ax, 9.5, 3.5, 7.5, 4, 'ENTERPRISE SUBNET  (4 hosts)',
                          *SUBNET_COLORS['Enterprise'], ent_hosts, ent_images)

# Operational Subnet (top-right)
op_hosts = ['Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0']
op_images = ['Gateway', 'Gateway', 'Gateway', 'OP_Server']
op_pos = draw_subnet_box(ax, 18, 7.5, 7.5, 4, 'OPERATIONAL SUBNET  (4 hosts)',
                         *SUBNET_COLORS['Operational'], op_hosts, op_images)

# DMZ Subnet (top-left)
dmz_hosts = ['DMZ_Server0', 'DMZ_Server1']
dmz_images = ['Internal', 'Internal']
dmz_pos = draw_subnet_box(ax, 0.5, 10, 4.5, 3, 'DMZ  (2 hosts)',
                          *SUBNET_COLORS['DMZ'], dmz_hosts, dmz_images)

# Research Subnet (top-center)
res_hosts = ['Research0', 'Research1', 'Research2']
res_images = ['linux_user', 'linux_user', 'windows_user']
res_pos = draw_subnet_box(ax, 6, 10, 6, 3, 'RESEARCH SUBNET  (3 hosts)',
                          *SUBNET_COLORS['Research'], res_hosts, res_images)

# ===== NETWORK CONNECTIONS (inter-subnet) =====
def draw_connection(ax, x1, y1, x2, y2, color='#aaa', style='-', lw=1.5, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               linestyle=style, connectionstyle='arc3,rad=0.1'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='center', fontsize=7,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor=color))

# User <-> Enterprise (bidirectional)
ax.annotate('', xy=(9.5, 5.5), xytext=(8.5, 5.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=2))
ax.text(9.0, 5.85, '<->', ha='center', va='center', fontsize=8, color='#666', fontweight='bold')

# Enterprise <-> Operational
ax.annotate('', xy=(18, 9.0), xytext=(17, 5.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=2,
                            connectionstyle='arc3,rad=0.15'))
ax.text(17.8, 7.5, '<->', ha='center', va='center', fontsize=8, color='#666', fontweight='bold')

# Enterprise <-> DMZ
ax.annotate('', xy=(5, 10), xytext=(9.5, 7.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=2,
                            connectionstyle='arc3,rad=-0.15'))

# Enterprise <-> Research
ax.annotate('', xy=(9, 10), xytext=(13, 7.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=2,
                            connectionstyle='arc3,rad=0.15'))

# Research <-> Operational
ax.annotate('', xy=(18, 9.5), xytext=(12, 11.5),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=1.5,
                            connectionstyle='arc3,rad=-0.1'))

# ===== NACL FIREWALL INDICATORS =====
# Draw firewall symbols
def draw_firewall(ax, x, y, label):
    ax.text(x, y, 'X', ha='center', va='center', fontsize=14, fontweight='bold', color='#c62828')
    ax.text(x, y - 0.35, label, ha='center', va='center', fontsize=6.5,
            color='#c62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.08', facecolor='#ffebee', edgecolor='#c62828', alpha=0.9))

draw_firewall(ax, 17.5, 6.5, 'User→Op\nBLOCKED')
draw_firewall(ax, 4.5, 8.5, 'DMZ→Op\nBLOCKED')
draw_firewall(ax, 14.5, 12.5, 'User→Res\nBLOCKED')

# ===== RED AGENT BOX =====
draw_agent_box(ax, 0.5, 15, 'RED AGENT  (Attacker)', RED_COLOR, [
    'Policy: Random per episode (1 of 3):',
    '   [1] B_lineAgent  -  Fixed kill chain:',
    '      User0 > User1 > Ent1 > Ent2 > Op_Server0 > Impact',
    '   [2] RedMeanderAgent  -  Adaptive/opportunistic:',
    '      Scans all subnets > Exploits all > PrivEsc > Impact',
    '   [3] SleepAgent  -  No-op (baseline)',
    '',
    'Actions: DiscoverRemoteSystems, DiscoverNetworkServices,',
    '   ExploitRemoteService, PrivilegeEscalate, Impact',
    'Goal: Compromise Op_Server0 OTService (-10 reward)',
    'Start: User0 with SYSTEM access (phishing entry)',
    'Reward: HybridImpactPwn (maximize disruption)',
], w=12.5, h=6)

# ===== BLUE AGENT BOX =====
draw_agent_box(ax, 14, 15, 'BLUE AGENT  (Defender  -  DQN)', BLUE_COLOR, [
    'Policy: DQN [512, 256, 128] learned via eps-greedy',
    'Action space: 6 types x 20 hosts = Discrete(120)',
    '   [1] Monitor  -  Network-wide scan',
    '   [2] Analyse  -  Deep inspect specific host',
    '   [3] Remove   -  Kill malicious processes/sessions',
    '   [4] Restore  -  Full host restore (costly)',
    '   [5] Isolate  -  Block lateral movement (meta-action)',
    '   [6] Patch    -  Fix vulnerability proactively (meta-action)',
    '',
    'Observation: 121-dim vector (6 features x 20 hosts + delay)',
    '   [scan_activity, exploit_activity, compromised_unknown,',
    '    compromised_level, is_isolated, is_patched] per host',
    '1-step observation delay (partial observability)',
    'Reward: HybridAvailabilityConfidentiality (negate Red)',
], w=12.5, h=7.2)

# ===== GREEN AGENT BOX =====
draw_agent_box(ax, 6.5, 0, 'GREEN AGENT  (Benign Users)', GREEN_COLOR, [
    'Policy: Random (scripted, non-learning)',
    'Actions: GreenPingSweep, GreenPortScan, GreenConnection',
    'Active on: User0-6, Op_Host0-2, Research0-2 (13 hosts)',
    'Purpose: Simulates legitimate user traffic / noise',
    '> Blue must avoid disrupting Green (Restore penalty)',
    '> Green traffic masks Red reconnaissance scans',
], w=12.5, h=3.2)

# ===== RED ATTACK PATH ARROWS =====
# Red starts at User0
ax.annotate('RED\nSTART', xy=(1.5, 4.8), fontsize=8, fontweight='bold',
            color=RED_COLOR, ha='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffebee', edgecolor=RED_COLOR))

# B_line kill chain arrows
bline_path = [
    ((2.0, 4.0), (3.5, 4.0), 'Scan'),           # User0 → User1
    ((4.3, 3.5), (10.5, 4.5), 'Exploit'),         # User1 → Enterprise1
    ((12, 3.5), (13.5, 4.5), 'Pivot'),             # Ent1 → Ent2
    ((14.5, 4.5), (19, 8.5), 'Exploit'),          # Ent2 → Op_Server0
]

for (x1,y1), (x2,y2), label in bline_path:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=RED_COLOR, lw=2,
                               linestyle='--', connectionstyle='arc3,rad=0.15'))

# Impact star on Op_Server0
ax.text(23.5, 10.5, '** IMPACT **\n(OTService)', ha='center', va='center',
        fontsize=9, fontweight='bold', color=RED_COLOR,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffcdd2', edgecolor=RED_COLOR, linewidth=2))
ax.annotate('', xy=(23.5, 10.0), xytext=(22.0, 9.5),
            arrowprops=dict(arrowstyle='->', color=RED_COLOR, lw=2.5))

# ===== BLUE DEFENSE ARROWS =====
# Blue monitors all subnets
for (sx, sy), label in [((4.5, 7.5), 'Monitor\nAll'), ((13, 7.5), 'Analyse\nRemove'),
                        ((22, 7.5), 'Restore\nPatch')]:
    ax.annotate('', xy=(sx, sy + 0.3), xytext=(sx, sy + 1.5),
                arrowprops=dict(arrowstyle='->', color=BLUE_COLOR, lw=2, alpha=0.6))
    ax.text(sx, sy + 2.0, label, ha='center', va='center', fontsize=7.5,
            color=BLUE_COLOR, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.08', facecolor='#e3f2fd', edgecolor=BLUE_COLOR, alpha=0.8))

# ===== Velociraptor monitoring indicator =====
ax.text(15.5, 6.2, '[Velociraptor]\n   (Defender)', ha='center', va='center',
        fontsize=7.5, color='#6a1b9a', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#f3e5f5', edgecolor='#6a1b9a', alpha=0.8))

# ===== REWARD STRUCTURE BOX =====
reward_x, reward_y = 18, 0
rw, rh = 8.5, 3.2
rbox = FancyBboxPatch((reward_x, reward_y), rw, rh, boxstyle="round,pad=0.2",
                      facecolor='#fafafa', edgecolor='#333', linewidth=2)
ax.add_patch(rbox)
ax.text(reward_x + rw/2, reward_y + rh - 0.3, 'REWARD STRUCTURE',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#333')

rewards = [
    ('Compromised Enterprise host:', '-1.0 / step', RED_COLOR),
    ('Compromised Op_Server0:', '-1.0 / step', RED_COLOR),
    ('Op_Server0 OT Disrupted:', '-10.0 / step', RED_COLOR),
    ('Successful Patch:', '+0.5 bonus', BLUE_COLOR),
    ('Isolation cost:', '-0.5 penalty', '#ff8f00'),
    ('Compromised User host:', '0.0 (no value)', '#888'),
]
for i, (desc, val, color) in enumerate(rewards):
    ax.text(reward_x + 0.4, reward_y + rh - 0.85 - i * 0.37, desc,
            ha='left', va='center', fontsize=8, color='#333')
    ax.text(reward_x + rw - 0.4, reward_y + rh - 0.85 - i * 0.37, val,
            ha='right', va='center', fontsize=8, fontweight='bold', color=color)

# ===== DQN TRAINING INFO BOX =====
dqn_x, dqn_y = 0.5, 0
dw, dh = 5.5, 3.2
dbox = FancyBboxPatch((dqn_x, dqn_y), dw, dh, boxstyle="round,pad=0.2",
                      facecolor='#e8eaf6', edgecolor='#283593', linewidth=2)
ax.add_patch(dbox)
ax.text(dqn_x + dw/2, dqn_y + dh - 0.3, 'DQN TRAINING CONFIG',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#283593')
dqn_items = [
    'Network: MLP [512 → 256 → 128]',
    'Buffer: 200K  |  Batch: 128',
    'ε: 1.0 → 0.05 over 30%',
    'γ = 0.99  |  τ = 0.005',
    'Total steps: 750,000',
    'GPU: RTX 5090 (CUDA 12.8)',
]
for i, item in enumerate(dqn_items):
    ax.text(dqn_x + 0.3, dqn_y + dh - 0.8 - i * 0.37, item,
            ha='left', va='center', fontsize=8, color='#283593')

# ===== LEGEND =====
legend_elements = [
    mpatches.Patch(facecolor='#ffcdd2', edgecolor=RED_COLOR, label='Red Attack Path (B_line kill chain)'),
    mpatches.Patch(facecolor='#e3f2fd', edgecolor=BLUE_COLOR, label='Blue Defense Coverage (all 20 hosts)'),
    mpatches.Patch(facecolor='#e8f5e9', edgecolor=GREEN_COLOR, label='Green Benign Traffic (13 hosts)'),
    mpatches.Patch(facecolor='#ffebee', edgecolor='#c62828', label='NACL Firewall (blocked inbound)'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=4,
          fontsize=9, framealpha=0.9, edgecolor='#ccc',
          bbox_to_anchor=(0.5, -0.01))

plt.savefig(r'D:\Vasanth\RL_CYB\architecture_3x.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Saved: architecture_3x.png")
plt.close()
