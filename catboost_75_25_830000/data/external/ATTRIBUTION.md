# External data attribution

The external measurements are ChEMBL-deposited records obtained from the official NCBI PubChem BioAssay service, https://pubchem.ncbi.nlm.nih.gov/, on 2026-11-07. The direct EMBL-EBI ChEMBL API returned HTTP 500 during retrieval, so its official PubChem deposition was used.

ChEMBL data are provided by EMBL-EBI, https://www.ebi.ac.uk/chembl/, under Creative Commons Attribution-ShareAlike 3.0 Unported, https://creativecommons.org/licenses/by-sa/3.0/. The cleaned external datasets and adaptations distributed here retain this license. ChEMBL licensing documentation: https://chembl.gitbook.io/chembl-interface-documentation/about. This statement applies to the external ChEMBL data; the user's competition files are retained separately under data/local.

Each retained measurement includes its PubChem AID, ChEMBL assay ID, DOI where available, assay description/cell metadata, units and source URL. Exact downloaded payloads and their URLs, retrieval timestamps and SHA256 digests are under data/external/raw. A specific ChEMBL release number is not supplied by these PubChem assay responses; no unsupported release assignment is made. Individual assay versions are preserved in the raw descriptions.

Services:
- https://pubchem.ncbi.nlm.nih.gov/docs/bioassays
- https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- https://www.ncbi.nlm.nih.gov/books/NBK25499/ (Entrez utilities)

No commercial calculated ChEMBL properties were extracted. Molecular descriptors and fingerprints were computed with RDKit from structures. All 320 original local train molecules and all 597 test molecules were excluded from external training by neutral parent, achiral, canonical-tautomer identity. Test concentrations and missingness are never predictor inputs.

## Scope of this v8 distribution

The paragraphs above document the original collection. This v8 package contains a derived subset and the copied collection-level `audit.json`, not the complete original download or its `raw/` payload directory. The original source project is `outputs/chembl_transfer`; it is not a dependency for using the saved v8 model. Local competition files are under `data/` in this package, rather than `data/local/`.

No new network retrieval was performed for v8. `task_si_vacv.csv` preserves the previously aggregated 64-structure dataset. `strict_ic50_measurements.csv` contains the 17 measurements retained by the stricter CC50/IC50 endpoint qualification, corresponding to 16 structures. `prepared_si.csv` retains the original structure alignment and adds qualified targets in `strict_si`; an empty `strict_si` excludes that structure from the active external training experiment. The selected production model does not use external labels. The qualification and rejected mixed-endpoint trial are documented in `REPORT.md` and `reports/external_qualification.json`.
