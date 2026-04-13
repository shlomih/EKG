"""Patch multilabel_merged_colab.ipynb to add Gmail completion notification."""
import json, re

NB_PATH = r'c:\Users\osnat\Documents\Shlomi\EKG\multilabel_merged_colab.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 1: add gmail.send scope + build _gmail_svc ────────────────────────
c1 = ''.join(nb['cells'][1]['source'])

OLD_SCOPE = (
    "_creds, _ = google.auth.default(\n"
    "    scopes=['https://www.googleapis.com/auth/drive.readonly']\n"
    ")"
)
NEW_SCOPE = (
    "_creds, _ = google.auth.default(\n"
    "    scopes=[\n"
    "        'https://www.googleapis.com/auth/drive.readonly',\n"
    "        'https://www.googleapis.com/auth/gmail.send',\n"
    "    ]\n"
    ")"
)
assert OLD_SCOPE in c1, 'scope pattern not found'
c1 = c1.replace(OLD_SCOPE, NEW_SCOPE, 1)

OLD_SVC = "_svc = _build('drive', 'v3', credentials=_creds, cache_discovery=False)"
NEW_SVC = (
    "_svc = _build('drive', 'v3', credentials=_creds, cache_discovery=False)\n"
    "_gmail_svc = _build('gmail', 'v1', credentials=_creds, cache_discovery=False)\n"
    "_my_email = _gmail_svc.users().getProfile(userId='me').execute()['emailAddress']\n"
    "print(f'Gmail ready: will notify {_my_email}')"
)
assert OLD_SVC in c1, 'svc pattern not found'
c1 = c1.replace(OLD_SVC, NEW_SVC, 1)

nb['cells'][1]['source'] = c1
print(f'Cell 1 patched: {len(c1)} chars')

# ── Cell 2: capture output + send email ────────────────────────────────────
c2 = ''.join(nb['cells'][2]['source'])

# 2a. Add `re` to imports
OLD_IMP = "import time, os, sys, subprocess, shutil"
assert OLD_IMP in c2, 're import: original import not found'
c2 = c2.replace(OLD_IMP, "import time, os, sys, subprocess, shutil, re", 1)

# 2b. Capture lines while streaming
OLD_STREAM = (
    "for line in proc.stdout:\n"
    "    print(line, end='', flush=True)\n"
    "proc.wait()"
)
NEW_STREAM = (
    "_captured = []\n"
    "for line in proc.stdout:\n"
    "    print(line, end='', flush=True)\n"
    "    _captured.append(line.rstrip())\n"
    "proc.wait()"
)
assert OLD_STREAM in c2, 'stream pattern not found'
c2 = c2.replace(OLD_STREAM, NEW_STREAM, 1)

# 2c. Insert email block just before the final print line
# Find the last print statement by looking for the known suffix
SUFFIX = "s)')\n"   # unique end of: print(f'\nCell 2 done ({time.time()-t0:.0f}s)')
assert c2.endswith(SUFFIX), f'expected suffix not found; end: {repr(c2[-30:])}'
final_print_line = c2[c2.rfind("\nprint(f'") + 1:]   # everything from last print
assert final_print_line.startswith("print(f'"), f'unexpected: {repr(final_print_line[:40])}'

EMAIL_BLOCK = (
    "# -- Send completion email -----------------------------------------------\n"
    "try:\n"
    "    import base64\n"
    "    from email.mime.text import MIMEText\n"
    "\n"
    "    # Collect key summary lines from training output\n"
    "    _kw = ('MacroAUROC', 'MacroF1', 'MicroF1', 'Early stop',\n"
    "           'Checkpoint saved', '  Ep ')\n"
    "    _summary = [l for l in _captured if any(k in l for k in _kw)]\n"
    "    # Per-class breakdown lines (4-space indent + label + F1=)\n"
    r"    _summary += [l for l in _captured if re.match(r'    \w', l) and 'F1=' in l]"
    "\n"
    "    _summary = list(dict.fromkeys(_summary))  # dedup, preserve order\n"
    "\n"
    "    # Extract MacroAUROC for subject line\n"
    "    _mac = next((l for l in reversed(_captured) if 'MacroAUROC' in l), '')\n"
    "    _av = re.search(r'[\\d.]{5,}', _mac)\n"
    "    _auroc_str = _av.group() if _av else '?'\n"
    "\n"
    "    _elapsed = f'{(time.time()-t0)/60:.0f} min'\n"
    "    _body = (\n"
    "        f'Training completed in {_elapsed} on {accel}.\\n\\n'\n"
    "        + '\\n'.join(_summary[-60:])\n"
    "        + f'\\n\\nFull run: {len(_captured)} lines.\\n'\n"
    "    )\n"
    "    _msg = MIMEText(_body)\n"
    "    _msg['To'] = _my_email\n"
    "    _msg['From'] = _my_email\n"
    "    _msg['Subject'] = f'[EKG] Training complete - MacroAUROC={_auroc_str}'\n"
    "    _raw = base64.urlsafe_b64encode(_msg.as_bytes()).decode()\n"
    "    _creds.refresh(_GReq())\n"
    "    _gmail_svc.users().messages().send(userId='me', body={'raw': _raw}).execute()\n"
    "    print(f'Email sent to {_my_email}')\n"
    "except Exception as _e:\n"
    "    print(f'Email notification failed (non-fatal): {_e}')\n"
    "\n"
)

# Insert email block before the final print line
insert_pos = len(c2) - len(final_print_line)
c2 = c2[:insert_pos] + EMAIL_BLOCK + final_print_line

nb['cells'][2]['source'] = c2
print(f'Cell 2 patched: {len(c2)} chars')

# Quick sanity checks
assert '_gmail_svc' in c1
assert 'gmail.send' in c1
assert '_my_email' in c1
assert '_captured' in c2
assert 'MIMEText' in c2
assert '_gmail_svc.users().messages().send' in c2
print('All assertions passed')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook saved OK')
