# AirPods capture quality

Open **Devices → Open Fit Test** for a same-person signal check. Collect the reference, freeze it before removing the AirPods, then reseat both and record the second session. The app scores new takes against the frozen reference. Use the controlled fit comparison if you also need an unchanged-fit block under matched conditions.

Use the local cohort audit for paired practice recordings. For each participant, collect a first visit, a fresh repeat with the AirPods left in place, and a fresh visit after reseating them. Use the same prompts and headset for those three visits. Assign each person one anonymous code across all visits. Record the visit IDs from the `visits` array in the participant’s `sentence-training.json` and use copies of the stores for analysis. Wait for spoken checks to finish before copying them.

Create a manifest such as the one below. Paths are relative to the manifest. Keep the manifest local because it points to the original recordings.

```json
{
  "schema_version": 1,
  "sessions": [
    {"speaker": "P01", "phase": "first", "store": "p01/sentence-training.json", "visit_id": "00000000-0000-4000-8000-000000000001"},
    {"speaker": "P01", "phase": "repeat", "store": "p01/sentence-training.json", "visit_id": "00000000-0000-4000-8000-000000000002"},
    {"speaker": "P01", "phase": "reseated", "store": "p01/sentence-training.json", "visit_id": "00000000-0000-4000-8000-000000000003"}
  ]
}
```

From the repository root:

```sh
uv run --project ml python -m subvox_ml.airpods_audit /path/to/manifest.json /path/to/report.json
```

The report shows the first three saved pairs per participant, motion eligibility for spoken and silent takes, usable silent seconds, and comparisons over prompts present in both visits. Earlier attempts discarded before saving cannot be counted. The report includes anonymous codes and hashed pair IDs. It omits raw text, paths, sensor samples, audio, and wall-clock timestamps.

To prepare held-out evaluations, add a `splits` object with `train`, `development`, and `test` entries. Each entry has a `speakers` array of anonymous codes and a `families` array. Use the community corpus `source_id` for published reading material, `literature:<pieceID>` for library pieces, and `prompt_families` to map other prompt IDs to their source families. Keep related text in the same family, including translations and variants. Freeze the manifest before looking at test results. Give every partition a separate set of speakers and families.

Only pairs with matching speaker and family assignments enter a split. The audit quarantines crossings, shared prompts, ineligible pairs, and unclassified material. A report with `splits.ready` set to `false` has an empty partition. The split IDs are hashes of the JSON array `["airpods-audit-v1", speakerCode, lowercasePairUUID]` using the canonical JSON encoding in `subvox_ml.data.digest`. The audit measures capture quality and split readiness. Decoder accuracy requires a trained model evaluated on the held-out partitions.
