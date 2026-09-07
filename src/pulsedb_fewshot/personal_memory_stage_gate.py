"""Predeclared exploratory go/no-go rule; not the model-promotion rule."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cache_dirs, output):
    modes = {"random_disjoint", "chronological_blocked"}
    hashes, evidence = {}, []
    for directory in cache_dirs:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        mode = manifest["split_mode"]
        if (mode not in modes or mode in hashes or manifest["status"] != "complete"
                or manifest["protocol_id"] != "development-calbased-analogue-v1"
                or manifest.get("source_parent_split") != "meta_train"
                or manifest["heldout_test_accessed"] is not False
                or manifest["read_roles"] != ["train", "internal_validation"]):
            raise ValueError("invalid/duplicate cache protocol")
        summary_path = directory / "probe_summary.json"
        entry = manifest.get("files", {}).get("probe_summary.json", {})
        if entry.get("path") != "probe_summary.json" or summary_path.is_symlink() or sha(summary_path) != entry.get("sha256"):
            raise ValueError("diagnostic evidence hash differs from bound cache manifest")
        summary = json.loads(summary_path.read_text())
        if (summary["status"] != "complete" or summary["split_mode"] != mode
                or summary.get("protocol_id") != manifest["protocol_id"]
                or summary.get("heldout_test_accessed") is not False):
            raise ValueError("incomplete diagnostic evidence")
        hashes[mode] = sha(manifest_path)
        baseline = float(summary["overall"]["D0"]["Overall"]["mean_mae"])
        for method in ["D1", "D2"]:
            gain = baseline - float(summary["overall"][method]["Overall"]["mean_mae"])
            if gain >= 0.02:
                evidence.append({"split_mode": mode, "method": method, "scope": "Overall", "gain_mmhg": gain})
        for subgroup in summary["subgroups"]:
            if (subgroup["method"] in {"D1", "D2"}
                    and subgroup["group_type"] == "bp_deviation"
                    and subgroup["scope"] in {"MIMIC", "VitalDB"}
                    and subgroup["n_participants"] >= 100
                    and subgroup["n_events"] >= 500
                    and subgroup["gain_mmhg"] >= 0.15):
                evidence.append({"split_mode": mode, **subgroup})
    if set(hashes) != modes:
        raise ValueError("both split modes are required before the joint go/no-go decision")
    decision = "proceed_to_train" if evidence else "stop_no_complementarity"
    result = {"status": decision, "decision": decision, "screen_id": "personal-memory-v1",
              "protocol_id": "development-calbased-analogue-v1", "heldout_test_accessed": False,
              "cache_manifest_sha256": hashes, "evidence": evidence,
              "selection_role": "internal_validation",
              "gate_is_exploratory_not_confirmation": True,
              "rule": "D1/D2 overall mean gain >=0.02 in either mode OR preregistered source/BP-deviation cell gain>=0.15 with>=100 persons and>=500 queries; no test access, no parameters fitted here",
              "promotion_rule_unchanged": "overall mean>=0.15 in BOTH modes AND positive improvement in each source/mode"}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dirs", nargs=2, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.cache_dirs, args.output), indent=2))


if __name__ == "__main__":
    main()
