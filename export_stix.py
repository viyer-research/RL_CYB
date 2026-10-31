"""
STIX 2.1 Export for CybORG 3× Simulation
==========================================
Maps CybORG simulation episodes to STIX 2.1 bundles, enabling:
  - Comparison with real-world STIX threat intelligence feeds
  - ATT&CK technique coverage analysis
  - Incident timeline reconstruction in STIX-compatible tools
  - Integration with SIEM / threat intel platforms (MISP, OpenCTI, etc.)

CybORG Action → MITRE ATT&CK → STIX 2.1 Mapping:
  Red:
    DiscoverRemoteSystems   → T1018  (Remote System Discovery)
    DiscoverNetworkServices → T1046  (Network Service Scanning)
    ExploitRemoteService    → T1210  (Exploitation of Remote Services)
      - EternalBlue           → CVE-2017-0144 (MS17-010)
      - BlueKeep              → CVE-2019-0708
      - HTTPRFI               → T1190 (Exploit Public-Facing App)
      - SSHBruteForce         → T1110.001 (Password Guessing)
    PrivilegeEscalate       → TA0004 (Privilege Escalation)
      - JuicyPotato           → T1134.001 (Token Impersonation)
      - V4L2KernelExploit     → T1068 (Exploitation for Priv Esc)
    Impact                  → T1489  (Service Stop)

  Blue:
    Monitor                 → Detection / TA0043 (Reconnaissance Detection)
    Analyse                 → Analysis / Investigation
    Remove                  → Mitigation / Containment
    Restore                 → Recovery / TA0040 (Impact Mitigation)
    Isolate                 → Network Segmentation (M1030)
    Patch                   → Vulnerability Management (M1051)
"""

import os
import sys
import json
import uuid
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# Add CybORG to path
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

# ============================================================
#  ATT&CK Mapping Tables
# ============================================================

# Red CybORG actions → STIX attack-pattern references
RED_ACTION_ATTACK_MAP = {
    'DiscoverRemoteSystems': {
        'technique_id': 'T1018',
        'technique_name': 'Remote System Discovery',
        'tactic': 'Discovery',
        'tactic_id': 'TA0007',
        'description': 'Adversary scans subnet for live hosts via ICMP ping sweep',
        'kill_chain_phase': 'discovery',
    },
    'DiscoverNetworkServices': {
        'technique_id': 'T1046',
        'technique_name': 'Network Service Scanning',
        'tactic': 'Discovery',
        'tactic_id': 'TA0007',
        'description': 'Adversary enumerates open ports and running services on target host',
        'kill_chain_phase': 'discovery',
    },
    'ExploitRemoteService': {
        'technique_id': 'T1210',
        'technique_name': 'Exploitation of Remote Services',
        'tactic': 'Lateral Movement',
        'tactic_id': 'TA0008',
        'description': 'Adversary exploits remote service vulnerability for code execution',
        'kill_chain_phase': 'lateral-movement',
        'sub_techniques': {
            'EternalBlue': {
                'technique_id': 'T1210',
                'cve': 'CVE-2017-0144',
                'name': 'EternalBlue (MS17-010 SMB)',
                'port': 139,
            },
            'BlueKeep': {
                'technique_id': 'T1210',
                'cve': 'CVE-2019-0708',
                'name': 'BlueKeep (RDP)',
                'port': 3389,
            },
            'HTTPRFI': {
                'technique_id': 'T1190',
                'cve': None,
                'name': 'HTTP Remote File Inclusion',
                'port': 80,
            },
            'HTTPSRFI': {
                'technique_id': 'T1190',
                'cve': None,
                'name': 'HTTPS Remote File Inclusion',
                'port': 443,
            },
            'SSHBruteForce': {
                'technique_id': 'T1110.001',
                'cve': None,
                'name': 'SSH Brute Force (Password Guessing)',
                'port': 22,
            },
        },
    },
    'PrivilegeEscalate': {
        'technique_id': 'T1068',
        'technique_name': 'Exploitation for Privilege Escalation',
        'tactic': 'Privilege Escalation',
        'tactic_id': 'TA0004',
        'description': 'Adversary escalates to SYSTEM/root on compromised host',
        'kill_chain_phase': 'privilege-escalation',
        'sub_techniques': {
            'JuicyPotato': {
                'technique_id': 'T1134.001',
                'name': 'Token Impersonation/Theft (JuicyPotato)',
                'os': 'windows',
            },
            'V4L2KernelExploit': {
                'technique_id': 'T1068',
                'name': 'Exploitation for Privilege Escalation (V4L2 Kernel)',
                'os': 'linux',
            },
        },
    },
    'Impact': {
        'technique_id': 'T1489',
        'technique_name': 'Service Stop',
        'tactic': 'Impact',
        'tactic_id': 'TA0040',
        'description': 'Adversary disrupts OTService on operational server',
        'kill_chain_phase': 'impact',
    },
}

# Blue CybORG actions → STIX course-of-action references
BLUE_ACTION_MITIGATION_MAP = {
    'Monitor': {
        'mitigation_name': 'Network Monitoring',
        'description': 'Blue agent performs network-wide monitoring to detect anomalous activity',
        'type': 'detection',
    },
    'Analyse': {
        'mitigation_name': 'Host Analysis',
        'description': 'Blue agent performs deep analysis on specific host for compromise indicators',
        'type': 'detection',
    },
    'Remove': {
        'mitigation_name': 'Malware Removal',
        'description': 'Blue agent kills malicious processes and removes adversary sessions',
        'type': 'containment',
    },
    'Restore': {
        'mitigation_name': 'System Restore',
        'description': 'Blue agent performs full host restoration to clean state',
        'type': 'recovery',
    },
    'Isolate': {
        'mitigation_name': 'Network Segmentation (M1030)',
        'mitigation_id': 'M1030',
        'description': 'Blue agent isolates host from network to prevent lateral movement',
        'type': 'containment',
    },
    'Patch': {
        'mitigation_name': 'Update Software (M1051)',
        'mitigation_id': 'M1051',
        'description': 'Blue agent patches vulnerability on host proactively',
        'type': 'prevention',
    },
}

# Host → infrastructure mapping for STIX
HOST_INFRASTRUCTURE = {
    'User0': {'type': 'workstation', 'os': 'windows', 'subnet': 'User', 'criticality': 'low'},
    'User1': {'type': 'workstation', 'os': 'windows', 'subnet': 'User', 'criticality': 'low'},
    'User2': {'type': 'workstation', 'os': 'windows', 'subnet': 'User', 'criticality': 'low'},
    'User3': {'type': 'workstation', 'os': 'linux', 'subnet': 'User', 'criticality': 'low'},
    'User4': {'type': 'workstation', 'os': 'linux', 'subnet': 'User', 'criticality': 'low'},
    'User5': {'type': 'workstation', 'os': 'windows', 'subnet': 'User', 'criticality': 'low'},
    'User6': {'type': 'workstation', 'os': 'linux', 'subnet': 'User', 'criticality': 'low'},
    'Enterprise0': {'type': 'gateway', 'os': 'linux', 'subnet': 'Enterprise', 'criticality': 'medium'},
    'Enterprise1': {'type': 'server', 'os': 'linux', 'subnet': 'Enterprise', 'criticality': 'medium'},
    'Enterprise2': {'type': 'server', 'os': 'linux', 'subnet': 'Enterprise', 'criticality': 'medium'},
    'Defender': {'type': 'monitoring', 'os': 'linux', 'subnet': 'Enterprise', 'criticality': 'high'},
    'Op_Host0': {'type': 'gateway', 'os': 'linux', 'subnet': 'Operational', 'criticality': 'medium'},
    'Op_Host1': {'type': 'gateway', 'os': 'linux', 'subnet': 'Operational', 'criticality': 'medium'},
    'Op_Host2': {'type': 'gateway', 'os': 'linux', 'subnet': 'Operational', 'criticality': 'medium'},
    'Op_Server0': {'type': 'ot-server', 'os': 'linux', 'subnet': 'Operational', 'criticality': 'critical'},
    'DMZ_Server0': {'type': 'server', 'os': 'linux', 'subnet': 'DMZ', 'criticality': 'medium'},
    'DMZ_Server1': {'type': 'server', 'os': 'linux', 'subnet': 'DMZ', 'criticality': 'medium'},
    'Research0': {'type': 'workstation', 'os': 'linux', 'subnet': 'Research', 'criticality': 'medium'},
    'Research1': {'type': 'workstation', 'os': 'linux', 'subnet': 'Research', 'criticality': 'medium'},
    'Research2': {'type': 'workstation', 'os': 'windows', 'subnet': 'Research', 'criticality': 'medium'},
}


# ============================================================
#  STIX 2.1 Object Generators (manual, no stix2 library needed)
# ============================================================

def stix_id(stype):
    """Generate a STIX 2.1 compliant ID."""
    return f"{stype}--{uuid.uuid4()}"


def make_attack_pattern(technique_info):
    """Create STIX attack-pattern from ATT&CK technique info."""
    ap = {
        'type': 'attack-pattern',
        'spec_version': '2.1',
        'id': stix_id('attack-pattern'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'name': technique_info['technique_name'],
        'description': technique_info['description'],
        'external_references': [
            {
                'source_name': 'mitre-attack',
                'external_id': technique_info['technique_id'],
                'url': f"https://attack.mitre.org/techniques/{technique_info['technique_id'].replace('.', '/')}/",
            }
        ],
        'kill_chain_phases': [
            {
                'kill_chain_name': 'mitre-attack',
                'phase_name': technique_info['kill_chain_phase'],
            }
        ],
    }
    return ap


def make_infrastructure(hostname, host_info):
    """Create STIX infrastructure object for a network host."""
    return {
        'type': 'infrastructure',
        'spec_version': '2.1',
        'id': stix_id('infrastructure'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'name': hostname,
        'description': f"{host_info['type']} ({host_info['os']}) in {host_info['subnet']} subnet",
        'infrastructure_types': [host_info['type']],
        'x_subnet': host_info['subnet'],
        'x_os': host_info['os'],
        'x_criticality': host_info['criticality'],
    }


def make_threat_actor(red_agent_name):
    """Create STIX threat-actor for a Red agent profile."""
    descriptions = {
        'meander': 'Adaptive Red agent that opportunistically scans all subnets, '
                   'exploits all reachable hosts, and escalates privileges before '
                   'attempting Impact on operational infrastructure.',
        'bline': 'Deterministic Red agent following a fixed kill chain: '
                 'User0→User1→Enterprise1→Enterprise2→Op_Server0→Impact. '
                 'Models a targeted APT with predetermined lateral movement path.',
        'sleep': 'Inactive Red agent (no-op baseline). Models periods of no adversary activity.',
    }
    return {
        'type': 'threat-actor',
        'spec_version': '2.1',
        'id': stix_id('threat-actor'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'name': f'CybORG_{red_agent_name}',
        'description': descriptions.get(red_agent_name, 'Simulated adversary'),
        'threat_actor_types': ['unknown'],
        'sophistication': 'expert' if red_agent_name == 'bline' else
                          'advanced' if red_agent_name == 'meander' else 'none',
        'resource_level': 'organization',
        'primary_motivation': 'organizational-gain',
        'x_simulation_agent': red_agent_name,
    }


def make_course_of_action(blue_action_info):
    """Create STIX course-of-action for a Blue agent defense action."""
    coa = {
        'type': 'course-of-action',
        'spec_version': '2.1',
        'id': stix_id('course-of-action'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'name': blue_action_info['mitigation_name'],
        'description': blue_action_info['description'],
        'x_action_type': blue_action_info['type'],
    }
    if 'mitigation_id' in blue_action_info:
        coa['external_references'] = [{
            'source_name': 'mitre-attack',
            'external_id': blue_action_info['mitigation_id'],
            'url': f"https://attack.mitre.org/mitigations/{blue_action_info['mitigation_id']}/",
        }]
    return coa


def make_observed_data(step, timestamp, red_action, blue_action,
                       target_host, reward, obs_summary):
    """Create STIX observed-data for a simulation step."""
    return {
        'type': 'observed-data',
        'spec_version': '2.1',
        'id': stix_id('observed-data'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'first_observed': timestamp,
        'last_observed': timestamp,
        'number_observed': 1,
        'x_simulation_step': step,
        'x_red_action': red_action,
        'x_blue_action': blue_action,
        'x_target_host': target_host,
        'x_reward': reward,
        'x_observation_summary': obs_summary,
    }


def make_sighting(attack_pattern_id, target_infra_id, timestamp, step):
    """Create STIX sighting linking observed attack to attack pattern."""
    return {
        'type': 'sighting',
        'spec_version': '2.1',
        'id': stix_id('sighting'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'first_seen': timestamp,
        'last_seen': timestamp,
        'count': 1,
        'sighting_of_ref': attack_pattern_id,
        'where_sighted_refs': [target_infra_id],
        'x_simulation_step': step,
    }


def make_relationship(source_id, rel_type, target_id, description=''):
    """Create STIX relationship."""
    return {
        'type': 'relationship',
        'spec_version': '2.1',
        'id': stix_id('relationship'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'relationship_type': rel_type,
        'source_ref': source_id,
        'target_ref': target_id,
        'description': description,
    }


def make_indicator(technique_info, confidence=85):
    """Create STIX indicator for detecting a specific technique."""
    return {
        'type': 'indicator',
        'spec_version': '2.1',
        'id': stix_id('indicator'),
        'created': datetime.now(timezone.utc).isoformat(),
        'modified': datetime.now(timezone.utc).isoformat(),
        'name': f"Detection for {technique_info['technique_name']}",
        'description': f"Indicator pattern for detecting {technique_info['technique_id']} "
                       f"({technique_info['technique_name']}) in simulated environment",
        'indicator_types': ['anomalous-activity'],
        'pattern': f"[network-traffic:extensions.'tcp'.dst_port = 0]",
        'pattern_type': 'stix',
        'valid_from': datetime.now(timezone.utc).isoformat(),
        'confidence': confidence,
        'external_references': [{
            'source_name': 'mitre-attack',
            'external_id': technique_info['technique_id'],
        }],
    }


# ============================================================
#  Episode Simulation & STIX Export
# ============================================================

def run_episode_and_export(model_path, red_agent_name='meander',
                           n_steps=100, output_dir=None):
    """
    Run one episode with the trained DQN model and export as STIX 2.1 bundle.
    """
    from train_dqn_3x import CybORG3xEnv, ALL_HOSTS, ACTION_TYPES
    from stable_baselines3 import DQN

    if output_dir is None:
        output_dir = r'D:\Vasanth\RL_CYB\stix_export'
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    # Create environment
    env = CybORG3xEnv(max_steps=n_steps, red_agent=red_agent_name)

    # ── Build static STIX objects ──
    stix_objects = []

    # Infrastructure (all 20 hosts)
    infra_map = {}  # hostname -> STIX ID
    for hostname, info in HOST_INFRASTRUCTURE.items():
        infra_obj = make_infrastructure(hostname, info)
        stix_objects.append(infra_obj)
        infra_map[hostname] = infra_obj['id']

    # Attack patterns (all Red techniques)
    attack_pattern_map = {}  # technique_id -> STIX ID
    for action_name, tech_info in RED_ACTION_ATTACK_MAP.items():
        ap = make_attack_pattern(tech_info)
        stix_objects.append(ap)
        attack_pattern_map[tech_info['technique_id']] = ap['id']

    # Threat actor
    threat_actor = make_threat_actor(red_agent_name)
    stix_objects.append(threat_actor)

    # Link threat actor → uses → attack patterns
    for tech_id, ap_id in attack_pattern_map.items():
        rel = make_relationship(threat_actor['id'], 'uses', ap_id,
                                f'Threat actor uses {tech_id}')
        stix_objects.append(rel)

    # Courses of action (Blue defenses)
    coa_map = {}  # action_name -> STIX ID
    for action_name, mit_info in BLUE_ACTION_MITIGATION_MAP.items():
        coa = make_course_of_action(mit_info)
        stix_objects.append(coa)
        coa_map[action_name] = coa['id']

    # Link CoA → mitigates → attack patterns
    mitigation_links = [
        ('Monitor', 'T1018'), ('Monitor', 'T1046'),
        ('Analyse', 'T1210'), ('Analyse', 'T1068'),
        ('Remove', 'T1210'), ('Remove', 'T1068'),
        ('Restore', 'T1489'), ('Restore', 'T1210'),
        ('Isolate', 'T1210'), ('Patch', 'T1210'),
    ]
    for action_name, tech_id in mitigation_links:
        if action_name in coa_map and tech_id in attack_pattern_map:
            rel = make_relationship(coa_map[action_name], 'mitigates',
                                    attack_pattern_map[tech_id])
            stix_objects.append(rel)

    # Indicators for detection
    for action_name, tech_info in RED_ACTION_ATTACK_MAP.items():
        ind = make_indicator(tech_info)
        stix_objects.append(ind)

    # ── Run episode ──
    print(f"\nRunning episode vs {red_agent_name} ({n_steps} steps)...")
    obs, info = env.reset()

    # ── Build IP → hostname mapping from CybORG (AFTER reset, IPs randomised) ──
    ip_to_host = {}
    try:
        true_state = env._cyborg.environment_controller.state
        for hostname in ALL_HOSTS:
            if hostname in true_state.hosts:
                host_obj = true_state.hosts[hostname]
                for iface in host_obj.interfaces:
                    ip_str = str(iface.ip_address)
                    if ip_str != '127.0.0.1':
                        ip_to_host[ip_str] = hostname
    except Exception:
        pass
    if ip_to_host:
        print(f"  Resolved {len(ip_to_host)} IP→Host mappings")

    base_time = datetime(2026, 2, 15, 10, 0, 0, tzinfo=timezone.utc)

    episode_events = []
    sightings = []
    total_reward = 0
    technique_counts = defaultdict(int)
    blue_action_counts = defaultdict(int)
    compromised_hosts = set()

    for step in range(n_steps):
        # Blue action
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        type_idx = action // len(ALL_HOSTS)
        host_idx = action % len(ALL_HOSTS)
        blue_action_type = ACTION_TYPES[type_idx]
        blue_target = ALL_HOSTS[host_idx]

        # Step
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        # Timestamp (1 minute per step)
        timestamp = (base_time + timedelta(minutes=step)).isoformat()

        # ── Get Red's ACTUAL action via CybORG API ──
        red_action_obj = env._cyborg.get_last_action('Red')
        red_action_str = str(red_action_obj) if red_action_obj else 'Sleep'
        red_action_class = type(red_action_obj).__name__ if red_action_obj else 'Sleep'

        # Parse Red target host from action string (try hostname, then IP)
        red_target_host = None
        for host in ALL_HOSTS:
            if host in red_action_str:
                red_target_host = host
                break
        if red_target_host is None:
            # Try resolving IP addresses in action string
            import re
            ip_matches = re.findall(r'\d+\.\d+\.\d+\.\d+', red_action_str)
            for ip in ip_matches:
                if ip in ip_to_host:
                    red_target_host = ip_to_host[ip]
                    break

        # Map Red action class → ATT&CK technique
        red_technique_id = None
        red_technique_name = None
        for action_name, tech_info in RED_ACTION_ATTACK_MAP.items():
            if action_name in red_action_class:
                red_technique_id = tech_info['technique_id']
                red_technique_name = tech_info['technique_name']
                technique_counts[red_technique_id] += 1

                # Check for sub-techniques (specific exploit variants)
                if 'sub_techniques' in tech_info:
                    for sub_name, sub_info in tech_info['sub_techniques'].items():
                        if sub_name.lower() in red_action_str.lower():
                            red_technique_id = sub_info['technique_id']
                            red_technique_name = sub_info.get('name', sub_name)
                            technique_counts[red_technique_id] += 1
                            break
                break

        # Track compromised hosts from observation
        obs_reshaped = obs[:-1].reshape(len(ALL_HOSTS), 6)
        for h_idx, host in enumerate(ALL_HOSTS):
            if obs_reshaped[h_idx, 1] > 0 or obs_reshaped[h_idx, 3] > 0:
                compromised_hosts.add(host)
        if red_target_host and 'Exploit' in red_action_class:
            compromised_hosts.add(red_target_host)

        # Create observed-data with REAL Red action
        obs_data = make_observed_data(
            step=step,
            timestamp=timestamp,
            red_action=f"{red_action_class}" + (f" ({red_technique_id})" if red_technique_id else ""),
            blue_action=f"{blue_action_type} {blue_target}",
            target_host=blue_target,
            reward=float(reward),
            obs_summary={
                'red_action_raw': red_action_str,
                'red_target_host': red_target_host,
                'red_technique_id': red_technique_id,
                'red_technique_name': red_technique_name,
                'compromised_hosts': list(compromised_hosts),
                'isolated': int(obs_reshaped[:, 4].sum()),
                'patched': int(obs_reshaped[:, 5].sum()),
            },
        )
        stix_objects.append(obs_data)
        blue_action_counts[blue_action_type] += 1

        # Create sighting when Red uses a known technique
        if red_technique_id and red_technique_id in attack_pattern_map:
            target_infra = infra_map.get(red_target_host, list(infra_map.values())[0])
            sighting = make_sighting(
                attack_pattern_map[red_technique_id],
                target_infra,
                timestamp, step
            )
            stix_objects.append(sighting)
            sightings.append(sighting)

        if done:
            break

    # ── Build STIX Bundle ──
    bundle = {
        'type': 'bundle',
        'id': stix_id('bundle'),
        'objects': stix_objects,
    }

    # ── Save ──
    filename = f"cyborg_3x_episode_{red_agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(bundle, f, indent=2, default=str)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  STIX 2.1 Export Summary")
    print(f"{'='*70}")
    print(f"  Red agent:          {red_agent_name}")
    print(f"  Episode steps:      {step + 1}")
    print(f"  Total reward:       {total_reward:.1f}")
    print(f"  STIX objects:       {len(stix_objects)}")
    print(f"    - Infrastructure: {len(infra_map)}")
    print(f"    - Attack patterns:{len(attack_pattern_map)}")
    print(f"    - Sightings:      {len(sightings)}")
    print(f"    - Observed data:  {step + 1}")
    print(f"    - Relationships:  {sum(1 for o in stix_objects if o['type'] == 'relationship')}")
    print(f"    - Indicators:     {sum(1 for o in stix_objects if o['type'] == 'indicator')}")
    print(f"    - CoA (defenses): {len(coa_map)}")
    print(f"\n  ATT&CK Techniques Observed:")
    for tech_id, count in sorted(technique_counts.items(), key=lambda x: -x[1]):
        name = next((v['technique_name'] for v in RED_ACTION_ATTACK_MAP.values()
                     if v['technique_id'] == tech_id), tech_id)
        print(f"    {tech_id:8s} {name:40s} ×{count}")
    print(f"\n  Blue Action Distribution:")
    total_blue = sum(blue_action_counts.values())
    for act, count in sorted(blue_action_counts.items(), key=lambda x: -x[1]):
        print(f"    {act:10s}: {count:4d} ({count/total_blue*100:.0f}%)")
    print(f"\n  Compromised hosts:  {len(compromised_hosts)} / {len(ALL_HOSTS)}")
    print(f"    {', '.join(sorted(compromised_hosts)) if compromised_hosts else 'None'}")
    print(f"\n  Saved to: {filepath}")
    print(f"{'='*70}")

    return filepath, bundle


# Technique → tactic mapping for Navigator layer pinning
TECHNIQUE_TACTIC_MAP = {
    'T1595': 'reconnaissance',
    'T1595.001': 'reconnaissance',
    'T1018': 'discovery',
    'T1046': 'discovery',
    'T1210': 'lateral-movement',
    'T1190': 'initial-access',
    'T1110.001': 'credential-access',
    'T1068': 'privilege-escalation',
    'T1134.001': 'privilege-escalation',
    'T1489': 'impact',
}

# Techniques mitigated by Blue actions (for Navigator annotations)
BLUE_TECHNIQUE_ANNOTATIONS = {
    'T1210': 'Blue: Analyse/Remove/Restore/Isolate/Patch',
    'T1068': 'Blue: Analyse/Remove',
    'T1018': 'Blue: Monitor',
    'T1046': 'Blue: Monitor',
    'T1489': 'Blue: Restore',
}


def export_attck_coverage_matrix(bundle, output_path):
    """
    Generate an ATT&CK coverage heatmap from STIX bundle sightings.
    Produces a JSON compatible with ATT&CK Navigator layer format (v4.5).

    Only valid technique IDs (T****) are included; tactic (TA****) and
    mitigation (M****) IDs are filtered out as Navigator rejects them.
    """
    # Count sightings per technique
    technique_scores = defaultdict(int)
    for obj in bundle['objects']:
        if obj['type'] == 'sighting':
            ap_ref = obj['sighting_of_ref']
            for ap in bundle['objects']:
                if ap.get('id') == ap_ref and ap['type'] == 'attack-pattern':
                    for ref in ap.get('external_references', []):
                        if ref.get('source_name') == 'mitre-attack':
                            ext_id = ref['external_id']
                            # Only accept technique IDs (T****)
                            if ext_id.startswith('T'):
                                technique_scores[ext_id] += 1

    max_score = max(technique_scores.values()) if technique_scores else 1
    techniques = []
    for tech_id, score in technique_scores.items():
        entry = {
            'techniqueID': tech_id,
            'score': score,
            'color': '',
            'comment': f'Observed {score} times in simulation',
            'enabled': True,
            'showSubtechniques': '.' in tech_id,
        }
        # Pin to specific tactic if known
        if tech_id in TECHNIQUE_TACTIC_MAP:
            entry['tactic'] = TECHNIQUE_TACTIC_MAP[tech_id]
        # Append blue defense annotation
        if tech_id in BLUE_TECHNIQUE_ANNOTATIONS:
            entry['comment'] += f' | {BLUE_TECHNIQUE_ANNOTATIONS[tech_id]}'
        techniques.append(entry)

    layer = {
        'name': 'CybORG 3x Simulation Coverage',
        'versions': {
            'attack': '16',
            'navigator': '5.1',
            'layer': '4.5',
        },
        'domain': 'enterprise-attack',
        'description': 'ATT&CK technique coverage from CybORG 3x DQN autonomous defense simulation',
        'sorting': 3,
        'layout': {
            'layout': 'side',
            'aggregateFunction': 'average',
            'showID': True,
            'showName': True,
        },
        'gradient': {
            'colors': ['#ffffff', '#ff6666', '#ff0000'],
            'minValue': 0,
            'maxValue': max_score,
        },
        'legendItems': [
            {'label': 'Red technique (scored by sighting count)', 'color': '#ff6666'},
            {'label': 'Blue defense: Patch (M1051 - Update Software)', 'color': '#1565c0'},
            {'label': 'Blue defense: Isolate (M1030 - Network Segmentation)', 'color': '#1565c0'},
            {'label': 'Blue defense: Analyse/Remove/Restore', 'color': '#43a047'},
        ],
        'techniques': techniques,
    }

    with open(output_path, 'w') as f:
        json.dump(layer, f, indent=2)

    print(f"\n  ATT&CK Navigator layer saved to: {output_path}")
    print(f"    Techniques: {len(techniques)}, Max sightings: {max_score}")
    return layer


# ============================================================
#  Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export CybORG 3x episodes as STIX 2.1')
    parser.add_argument('--model', type=str,
                        default=r'D:\Vasanth\RL_CYB\ppo_out\dqn_3x_v1.zip',
                        help='Path to trained DQN model')
    parser.add_argument('--red', type=str, default='all',
                        choices=['meander', 'bline', 'sleep', 'all'],
                        help='Red agent to simulate')
    parser.add_argument('--steps', type=int, default=100,
                        help='Steps per episode')
    parser.add_argument('--output', type=str,
                        default=r'D:\Vasanth\RL_CYB\stix_export',
                        help='Output directory')
    args = parser.parse_args()

    print("=" * 70)
    print("  CybORG 3× → STIX 2.1 Exporter")
    print("=" * 70)
    print(f"  Model:  {args.model}")
    print(f"  Red:    {args.red}")
    print(f"  Steps:  {args.steps}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    red_agents = ['meander', 'bline', 'sleep'] if args.red == 'all' else [args.red]

    all_bundles = []
    for red_name in red_agents:
        filepath, bundle = run_episode_and_export(
            model_path=args.model,
            red_agent_name=red_name,
            n_steps=args.steps,
            output_dir=args.output,
        )
        all_bundles.append(bundle)

        # Export ATT&CK Navigator layer
        nav_path = os.path.join(args.output,
                                f"attck_navigator_{red_name}.json")
        export_attck_coverage_matrix(bundle, nav_path)

    print(f"\n{'='*70}")
    print(f"  All exports complete!")
    print(f"  Files in: {args.output}")
    print(f"\n  Usage:")
    print(f"    - Import .json into MISP, OpenCTI, or any STIX 2.1 consumer")
    print(f"    - Load attck_navigator_*.json into ATT&CK Navigator")
    print(f"      (https://mitre-attack.github.io/attack-navigator/)")
    print(f"    - Compare sightings against real-world CTI feeds")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
