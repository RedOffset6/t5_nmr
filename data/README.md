# Dataset Manifest

The full datasets are not stored in Git because of their size and possible redistribution restrictions.

## Directories

- `alberts_2d/` contains paired source and target files for train, validation, and test splits.
- `nmr_expt_data/` contains paired source and target files for train, validation, and test splits, plus `prd-test.txt`.

`manifest.tsv` records each file path, byte size, line count, and SHA-256 checksum.
Paired source and target files must retain matching line order.

## Verification

Run `sha256sum <file>` and compare the result with `manifest.tsv`.

Dataset provenance, licensing, and access instructions must be documented before public release.
