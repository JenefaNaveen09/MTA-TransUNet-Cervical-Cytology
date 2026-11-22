# Split manifests

The partitions of both Roboflow exports were used exactly as provided, so the splits are defined by the public exports (see `../README.md`).

Run `python scripts/make_split_manifest.py` to write, for each export, a JSON file listing every image per split with its source filename stem, size, box counts, and SHA-256 checksum:

- `cric_roboflow_v1.json`: train 147, valid 148, test 104 images
- `mendeley_roboflow_v1.json`: train 671, valid 191, test 97 images (pooled for external evaluation)

Commit the generated files with the release, so readers can confirm they hold the same images.
