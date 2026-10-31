"""
Fix STIX 2.1 bundle files: replace all TA0004 (tactic ID) with T1068 (technique ID)
across ALL object types (attack-pattern, observed-data, sighting, relationship, indicator).
"""
import json
import os


def fix_recursive(obj, replacements):
    """Recursively walk a JSON structure and apply string replacements."""
    count = 0
    if isinstance(obj, dict):
        for key in obj:
            if isinstance(obj[key], str):
                for old, new in replacements:
                    if old in obj[key]:
                        obj[key] = obj[key].replace(old, new)
                        count += 1
            else:
                count += fix_recursive(obj[key], replacements)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                for old, new in replacements:
                    if old in item:
                        obj[i] = item.replace(old, new)
                        count += 1
            else:
                count += fix_recursive(item, replacements)
    return count


REPLACEMENTS = [
    # Tactic ID → Technique ID
    ('TA0004', 'T1068'),
    # Fix the ATT&CK URL (tactics use /tactics/ path, techniques use /techniques/)
    ('https://attack.mitre.org/tactics/T1068', 'https://attack.mitre.org/techniques/T1068'),
    # Fix name where it was the tactic name
    ('"Privilege Escalation"', '"Exploitation for Privilege Escalation"'),
]

# String-level replacements (simpler and catches everything)
STR_REPLACEMENTS = [
    ('TA0004', 'T1068'),
    ('attack.mitre.org/tactics/T1068', 'attack.mitre.org/techniques/T1068'),
]

stix_dir = os.path.join(os.path.dirname(__file__), 'stix_export')

for fname in os.listdir(stix_dir):
    if not fname.startswith('cyborg_3x_episode_') or not fname.endswith('.json'):
        continue

    filepath = os.path.join(stix_dir, fname)
    print(f"\nProcessing: {fname}")

    # Read as raw text for reliable replacement
    with open(filepath, 'r') as f:
        content = f.read()

    before_count = content.count('TA0004')
    print(f"  TA0004 occurrences before: {before_count}")

    for old, new in STR_REPLACEMENTS:
        content = content.replace(old, new)

    after_count = content.count('TA0004')
    print(f"  TA0004 occurrences after:  {after_count}")

    # Validate JSON is still valid
    bundle = json.loads(content)

    # Verify all attack-patterns have T-prefixed IDs
    for obj in bundle['objects']:
        if obj['type'] == 'attack-pattern':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    eid = ref['external_id']
                    assert eid.startswith('T'), f"ERROR: attack-pattern still has non-T ID: {eid}"

    # Count object types
    types = {}
    for obj in bundle['objects']:
        types[obj['type']] = types.get(obj['type'], 0) + 1
    print(f"  Objects: {types}")
    print(f"  Total objects: {len(bundle['objects'])}")

    # Write back (pretty-printed)
    with open(filepath, 'w') as f:
        json.dump(bundle, f, indent=2)

    print(f"  ✓ Fixed and saved")

print("\n=== All STIX bundles fixed! ===")
