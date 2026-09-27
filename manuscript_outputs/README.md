# Manuscript-output provenance

`MANUSCRIPT_OUTPUT_MAP.csv` maps every displayed main-text figure/table and every table in the v0.22 online appendix to the computational family that supports it.

`MANUSCRIPT_OUTPUT_COVERAGE.csv` checks the map against the actual release candidate.

The repository deliberately does not duplicate the manuscript PDFs or full historical HANDbacks here. Exact full-archive hashes remain in `../provenance/EXTERNAL_ARCHIVE_MANIFEST.csv`.

A `PASS` row means the release retains both readable code and compact evidence for the relevant computational family. The mapping is family-level provenance; it does not claim that every displayed PDF was regenerated during this finalisation step.
