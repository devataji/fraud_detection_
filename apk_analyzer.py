#!/usr/bin/env python3
"""
APK Analyzer
------------
Extracts manifest metadata (permissions, activities, receivers, services) and
estimates Binder usage from DEX bytecode. Optionally parses syscall and binder
trace logs (from dynamic runs like strace/bindertrace) to compute frequencies.

USAGE
-----
python apk_analyzer.py path/to/app.apk \
    [--strace strace.txt] [--binder bindertrace.txt] [--out report.json] [--csv frequencies.csv]

OUTPUT
------
Prints a JSON object to stdout and (optionally) writes it to --out and CSV.

DEPENDENCIES
------------
- Python 3.9+
- androguard >= 3.4.0 (pip install androguard)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Optional, Tuple

try:
    from androguard.misc import AnalyzeAPK
except Exception as e:
    AnalyzeAPK = None  # type: ignore


def extract_manifest_info(apk_path: str) -> Dict:
    if AnalyzeAPK is None:
        raise RuntimeError(
            "androguard is required. Install with: pip install androguard"
        )
    a, d, dx = AnalyzeAPK(apk_path)
    
    # Fixed: Handle both string and object types for activities, receivers, services, providers
    def safe_get_name(item):
        if isinstance(item, str):
            return item
        elif hasattr(item, 'get_name'):
            return item.get_name()
        else:
            return str(item)
    
    return {
        "package": a.get_package(),
        "version_name": a.get_androidversion_name(),
        "version_code": a.get_androidversion_code(),
        "min_sdk": a.get_min_sdk_version(),
        "target_sdk": a.get_target_sdk_version(),
        "permissions": sorted(set(p for p in a.get_permissions() if p)),
        "activities": sorted(set(safe_get_name(x) for x in a.get_activities())),
        "receivers": sorted(set(safe_get_name(x) for x in a.get_receivers())),
        "services": sorted(set(safe_get_name(x) for x in a.get_services())),
        "providers": sorted(set(safe_get_name(x) for x in a.get_providers())),
    }

BINDER_CLASS_PREFIXES = (
    "Landroid/os/IBinder;",
    "Landroid/os/Binder;",
    "Landroid/os/Parcel;",
    "Landroid/os/ServiceManager;",
    "Landroid/content/ContentProvider;",
    "Landroid/app/IActivityManager;",
    "Landroid/app/ActivityManager;",
)

BINDER_METHOD_KEYWORDS = (
    "transact",
    "onTransact",
    "asInterface",
    "queryLocalInterface",
    "getService",
    "checkService",
    "addService",
    "obtain",
    "write*",
    "read*",
)

INVOKE_PREFIXES = ("INVOKE-", "INVOKE_")


def count_binder_calls(apk_path: str) -> Dict:
    if AnalyzeAPK is None:
        raise RuntimeError(
            "androguard is required. Install with: pip install androguard"
        )
    a, d, dx = AnalyzeAPK(apk_path)
    calls_by_method: Counter = Counter()
    total_invoke = 0
    
    # Handle both single DEX and list of DEX files
    dex_files = d if isinstance(d, list) else [d]
    
    for dex in dex_files:
        for current_class in dex.get_classes():
         for method in current_class.get_methods():
            code = method.get_code()
            if code is None:
                continue
            bytecode = code.get_bc()
            for ins in bytecode.get_instructions():
                op = ins.get_name() or ""
                if not op.startswith(INVOKE_PREFIXES):
                    continue
                total_invoke += 1
                try:
                    called_method, called_class = _resolve_invoked(ins, current_class)
                except Exception:
                    continue
                if called_method is None:
                    continue
                if _is_binder_like(called_method, called_class):
                    calls_by_method[called_method] += 1
    
    calls_by_class: Counter = Counter()
    for method_sig, cnt in calls_by_method.items():
        class_name = method_sig.split("->", 1)[0]
        matched = None
        for pref in BINDER_CLASS_PREFIXES:
            if class_name.startswith(pref):
                matched = pref
                break
        calls_by_class[matched or class_name] += cnt
    
    return {
        "calls_by_method": dict(calls_by_method.most_common()),
        "calls_by_class": dict(calls_by_class.most_common()),
        "total_invoke": total_invoke,
        "binder_like_invoke": int(sum(calls_by_method.values())),
    }


def _resolve_invoked(ins, current_class) -> Tuple[Optional[str], Optional[str]]:
    out = ins.get_output() or ""
    m = re.search(r"(L[^;]+;)->([^(]+)\(([^)]*)\).*", out)
    if not m:
        return None, None
    class_name, name, proto = m.group(1), m.group(2), m.group(3)
    return f"{class_name}->{name}:({proto})", class_name


def _is_binder_like(method_sig: str, class_name: Optional[str]) -> bool:
    if class_name and class_name.startswith(BINDER_CLASS_PREFIXES):
        return True
    m = re.search(r"->([^:]+):\(", method_sig)
    name = m.group(1) if m else method_sig
    low = name.lower()
    if any(k.lower().rstrip('*') in low for k in BINDER_METHOD_KEYWORDS if not k.endswith('*')):
        return True
    if any(low.startswith(k[:-1].lower()) for k in BINDER_METHOD_KEYWORDS if k.endswith('*')):
        return True
    return False

SYSCALL_RE = re.compile(r"^\s*([a-zA-Z0-9_]+)\(")
BINDER_EVENT_RE = re.compile(r"\bbinder_[a-z_]+\b")


def parse_strace(path: str) -> Dict[str, int]:
    counts: Counter = Counter()
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = SYSCALL_RE.search(line)
            if m:
                counts[m.group(1)] += 1
    return dict(counts.most_common())


def parse_bindertrace(path: str) -> Dict[str, int]:
    counts: Counter = Counter()
    with open(path, "r", errors="ignore") as f:
        for line in f:
            for ev in BINDER_EVENT_RE.findall(line):
                counts[ev] += 1
    return dict(counts.most_common())


def normalize_frequencies(counts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    total = sum(counts.values()) or 1
    freq = {k: v / total for k, v in counts.items()}
    return {
        "counts": counts,
        "total": total,
        "frequency": freq,
    }


def export_csv(data: Dict[str, Dict], csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Key", "Count", "Frequency"])
        for category, val in data.items():
            if not isinstance(val, dict) or "counts" not in val:
                continue
            counts = val["counts"]
            freqs = val["frequency"]
            for key, cnt in counts.items():
                writer.writerow([category, key, cnt, freqs.get(key, 0.0)])


def main():
    p = argparse.ArgumentParser(description="APK → manifest + binder + syscall/binder frequencies + CSV export")
    p.add_argument("apk", help="Path to APK file")
    p.add_argument("--strace", dest="strace", help="Path to strace log (optional)")
    p.add_argument("--binder", dest="binder", help="Path to binder trace log (optional)")
    p.add_argument("--out", dest="out", help="Write full JSON report to this file")
    p.add_argument("--csv", dest="csv", help="Write frequencies to CSV file")
    args = p.parse_args()

    if not os.path.isfile(args.apk):
        print(f"[!] APK not found: {args.apk}", file=sys.stderr)
        sys.exit(2)

    report = {
        "apk": os.path.abspath(args.apk),
        "manifest": extract_manifest_info(args.apk),
        "binder_static": count_binder_calls(args.apk),
        "strace": None,
        "binder_trace": None,
    }

    if args.strace:
        if not os.path.isfile(args.strace):
            print(f"[!] strace log not found: {args.strace}", file=sys.stderr)
            sys.exit(2)
        sc_counts = parse_strace(args.strace)
        report["strace"] = normalize_frequencies(sc_counts)

    if args.binder:
        if not os.path.isfile(args.binder):
            print(f"[!] binder trace log not found: {args.binder}", file=sys.stderr)
            sys.exit(2)
        b_counts = parse_bindertrace(args.binder)
        report["binder_trace"] = normalize_frequencies(b_counts)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[+] Report saved to {args.out}")

    if args.csv:
        export_csv(report, args.csv)
        print(f"[+] Frequency CSV saved to {args.csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)