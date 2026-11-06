"""Audit-first ChEMBL collection. No cross-assay arithmetic or invented measurements."""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import molecular_record
from .utils import sha256_file, utc_now, write_json

API_BASE = "https://www.ebi.ac.uk/chembl/api/data/"
API_DOCUMENTATION = "https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services"
ACTIVITY_TYPES = "IC50,CC50,EC50,SI,Selectivity index,Selectivity Index,Ratio CC50/IC50,Ratio CC50/EC50"


class ChEMBLClient:
    def __init__(self, cache_dir, offline=False, timeout=30):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline, self.timeout = offline, timeout
        self.requests = []

    def get(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "www.ebi.ac.uk":
            raise ValueError("Unexpected ChEMBL pagination URL.")
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["url"] != url:
                raise ValueError("Cache URL mismatch.")
        elif self.offline:
            raise FileNotFoundError(f"Offline cache miss: {url}")
        else:
            request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "vacv-selectivity/0.1 research-data-audit"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        payload = json.load(response)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise RuntimeError(f"ChEMBL HTTP {error.code}: {url}; {error.read().decode()[:400]}") from error
                    time.sleep(1 + attempt)
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            record = {"url": url, "retrieved_at": utc_now(), "payload": payload}
            write_json(path, record)
        self.requests.append({"url": url, "retrieved_at": record["retrieved_at"],
                              "cache_file": path.name, "sha256": sha256_file(path)})
        return record["payload"]

    def pages(self, resource, params, collection_key):
        query = dict(params, limit=1000)
        url = API_BASE + resource + ".json?" + urllib.parse.urlencode(sorted(query.items()))
        rows, visited = [], set()
        while url:
            if url in visited:
                raise RuntimeError("ChEMBL returned a pagination cycle.")
            visited.add(url)
            payload = self.get(url)
            if collection_key not in payload:
                raise ValueError(f"Missing API collection {collection_key}: {url}")
            rows.extend(payload[collection_key])
            next_page = payload.get("page_meta", {}).get("next")
            url = urllib.parse.urljoin(API_BASE, next_page) if next_page else None
        return rows


def batches(values, size=40):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def to_micromolar(value, unit):
    factors = {"M": 1e6, "mM": 1e3, "uM": 1.0, "µM": 1.0, "μM": 1.0,
               "nM": 1e-3, "pM": 1e-6, "fM": 1e-9}
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0 or unit not in factors:
        return None
    return numeric * factors[unit]


def normalize_activity(activity, assay, document):
    kind = (activity.get("standard_type") or "").strip()
    relation = activity.get("standard_relation") or ""
    description = assay.get("description") or activity.get("assay_description") or ""
    raw_value = activity.get("standard_value")
    try:
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0:
            value = None
    except (TypeError, ValueError):
        value = None
    molar = to_micromolar(raw_value, activity.get("standard_units"))
    lower_kind = kind.lower()
    ratio_type = lower_kind in {"si", "selectivity index", "ratio cc50/ic50", "ratio cc50/ec50"}
    # An SI field alone is insufficient: many publications use CC50/EC50.
    ic50_definition = "cc50/ic50" in lower_kind or bool(re.search(r"cc50\s*/\s*ic50", description, flags=re.I))
    ec50_definition = "cc50/ec50" in lower_kind or bool(re.search(r"cc50\s*/\s*ec50", description, flags=re.I))
    if ic50_definition and ec50_definition:
        definition = "ambiguous"
    elif ic50_definition:
        definition = "CC50/IC50"
    elif ec50_definition:
        definition = "CC50/EC50"
    else:
        definition = "unverified"
    smiles = activity.get("canonical_smiles") or ""
    try:
        canonical = molecular_record(smiles)["canonical_smiles"]
        valid_smiles = True
    except (ValueError, RuntimeError):
        canonical, valid_smiles = "", False
    organism = activity.get("target_organism") or ""
    validity = activity.get("data_validity_comment")
    exact_ratio = (ratio_type and definition == "CC50/IC50" and relation == "=" and value is not None
                   and "vaccinia" in organism.lower() and valid_smiles
                   and activity.get("standard_units") in (None, "", "dimensionless", "ratio")
                   and activity.get("potential_duplicate") != 1
                   and validity in (None, "", "Manually validated"))
    return {
        "smiles": smiles, "IC50": molar if kind == "IC50" else None,
        "CC50": molar if kind == "CC50" else None, "SI": value if ratio_type else None,
        "doi": document.get("doi") or "", "concentration_unit": "uM",
        "measurement_type": kind, "relation": relation, "raw_value": raw_value,
        "raw_unit": activity.get("standard_units"), "si_definition": definition,
        "canonical_smiles": canonical, "valid_smiles": valid_smiles,
        "eligible_for_training": exact_ratio, "data_validity_comment": validity,
        "potential_duplicate": activity.get("potential_duplicate"),
        "activity_id": activity.get("activity_id"), "assay_id": activity.get("assay_chembl_id"),
        "document_id": activity.get("document_chembl_id"), "molecule_id": activity.get("molecule_chembl_id"),
        "target_id": activity.get("target_chembl_id"), "target_organism": organism,
        "assay_description": description, "assay_type": assay.get("assay_type"),
        "assay_organism": assay.get("assay_organism"), "cell_type": assay.get("assay_cell_type"),
        "strain": assay.get("assay_strain"),
        "source_url": API_BASE + f"activity/{activity.get('activity_id')}.json",
        "source": "ChEMBL", "pairing_status": "single_reported_measurement;not_cross_assay_paired",
    }


def collect_chembl(output_dir, offline=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = ChEMBLClient(output / "cache", offline=offline)
    targets = client.pages("target", {"organism__icontains": "vaccinia"}, "targets")
    # Whole-virus activity only; do not conflate isolated viral-protein inhibition with cellular efficacy.
    targets = [t for t in targets if t["target_type"] == "ORGANISM"]
    if not targets:
        raise ValueError("No whole-organism VACV targets were returned.")
    ids = sorted(t["target_chembl_id"] for t in targets)
    activities = client.pages("activity", {"target_chembl_id__in": ",".join(ids), "standard_type__in": ACTIVITY_TYPES}, "activities")
    print(f"ChEMBL: {len(activities)} VACV activity records; fetching related cytotoxicity and provenance.", flush=True)
    document_ids = sorted({a["document_chembl_id"] for a in activities if a.get("document_chembl_id")})
    # CC50 may be assigned to a mammalian cell target rather than the virus.
    # Retrieve it from the SAME publication, retaining all assay context and never pairing on molecule ID alone.
    for chunk in batches(document_ids):
        activities.extend(client.pages("activity", {"document_chembl_id__in": ",".join(chunk), "standard_type": "CC50"}, "activities"))
    activities = list({a["activity_id"]: a for a in activities}.values())
    activities.sort(key=lambda row: row["activity_id"])
    documents = []
    for chunk in batches(document_ids):
        documents.extend(client.pages("document", {"document_chembl_id__in": ",".join(chunk)}, "documents"))
    assay_ids = sorted({a["assay_chembl_id"] for a in activities if a.get("assay_chembl_id")})
    assays = []
    for chunk in batches(assay_ids):
        assays.extend(client.pages("assay", {"assay_chembl_id__in": ",".join(chunk)}, "assays"))
    document_map = {d["document_chembl_id"]: d for d in documents}
    assay_map = {a["assay_chembl_id"]: a for a in assays}
    rows = [normalize_activity(a, assay_map.get(a.get("assay_chembl_id"), {}),
                               document_map.get(a.get("document_chembl_id"), {})) for a in activities]
    if not rows:
        raise ValueError("No activity records returned; do not present an empty file as a collected dataset.")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "external_candidates.csv", index=False)
    verified = frame[frame["eligible_for_training"]].copy()
    verified.to_csv(output / "external_verified.csv", index=False)
    write_json(output / "raw_activities.json", activities)
    write_json(output / "raw_assays.json", assays)
    write_json(output / "raw_documents.json", documents)
    write_json(output / "targets.json", targets)
    manifest = {"created_at": utc_now(), "source": "ChEMBL REST API", "documentation": API_DOCUMENTATION,
                "target_ids": ids, "records": len(frame), "documents": len(documents),
                "assays": len(assays), "explicit_ic50_based_si_rows": len(verified),
                "type_counts": {str(k): int(v) for k, v in frame["measurement_type"].value_counts().items()},
                "doi_rows": int(frame["doi"].ne("").sum()), "requests": client.requests,
                "files": {name: sha256_file(output / name) for name in
                          ("external_candidates.csv", "external_verified.csv", "raw_activities.json", "raw_assays.json", "raw_documents.json", "targets.json")},
                "policy": "No automatic CC50/IC50 joins across assays; no EC50→IC50 substitution; no automatic training augmentation.",
                "limitations": "Per-record wide schema is sparse: IC50, CC50 and SI are not necessarily measured together. Review papers and matched assay context to construct paired rows."}
    write_json(output / "collection_manifest.json", manifest)
    print(f"Collected {len(frame)} records, {len(verified)} explicitly IC50-based exact SI records; saved {output}.", flush=True)
    return manifest
