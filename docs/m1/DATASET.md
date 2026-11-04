# Dataset provenance and locked evaluation protocol

## Identity and access

- Base: CrisisMMD v2.0, humanitarian image-text classification. The selected resource is a custom five-class agreed-label derivative, not the full eight-class QCRI configuration.
- Primary QCRI revision: `10a5626ba112ab9c50369c473a2062bcf205d708`. [Pinned primary source](https://huggingface.co/datasets/QCRI/CrisisMMD/tree/10a5626ba112ab9c50369c473a2062bcf205d708).
- Actual mirror revision: `a935fda3cee8a5250746a8d1d6278214d615cc89`. [Pinned third-party source](https://huggingface.co/datasets/pranitchawla/crisismmd-mteb/tree/a935fda3cee8a5250746a8d1d6278214d615cc89). Exact four Parquet names, sizes, SHA256 values and URLs are in `A3-M1/provenance/mirror_downloads.json`.
- Upstream license: **CC BY-NC-SA 4.0**. The primary license text and card are retained under `A3-M1/provenance/`. Attribute Alam, Ofli and Imran; identify filtering changes; observe applicable noncommercial/share-alike conditions. No new source-content permissions are inferred.
- Original agreed-label archive: `https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip` returned **403 in the original audit**. It was not downloaded or retried for publication. Archive equivalence remains unverified; 245/245 prior image-byte matches cover only that checked subset.

Dataset version: `crisismmd5-mirror-remediation-20261104-v1`. Primary: **filtered_original**, **7,938 pairs = 6,001 train + 986 dev + 951 test**. Backup repartition is documented historically and is not the primary evaluation set.

## Label mapping and measured counts

| ID | Class | Train | Dev | Test | Total |
|---|---|---:|---:|---:|---:|
| 0 | affected_individuals | 71 | 9 | 9 | 89 |
| 1 | infrastructure_and_utility_damage | 601 | 80 | 81 | 762 |
| 2 | rescue_volunteering_or_donation_effort | 901 | 148 | 126 | 1175 |
| 3 | other_relevant_information | 1247 | 235 | 235 | 1717 |
| 4 | not_humanitarian | 3181 | 514 | 500 | 4195 |
| | Total | 6001 | 986 | 951 | 7938 |

Injured/dead and missing/found people merge into affected individuals; vehicle damage merges into infrastructure/utility damage. Primary modality agreement is checked before merging; remediation does not relabel targets. Source class-count CSVs retain explicit protocol labels, including historical original/backup counts.

## Alignment and exclusions

Each record links an attached image to its tweet ID, source path and event. Multiple images from one tweet are separate records, grouped together. Pair linkage does not establish every location/date claim or semantic entailment.

| Exclusion | Train | Dev | Test | Total |
|---|---:|---:|---:|---:|
| Primary annotation discrepancy | 7 | 0 | 0 | 7 |
| Verified IFTTT missing-image placeholder | 4 | 1 | 4 | 9 |
| Lower-priority duplicate/leakage component record | 114 | 11 | 0 | 125 |
| Total | 125 | 12 | 4 | 141 |

The seven training targets agree with QCRI humanitarian JSON/image labels (`not_humanitarian`), but primary event TSV text labels are `other_relevant_information`. IDs/text/paths matched in the original investigation. Cause remains unresolved; quarantine is not silent relabelling. `annotation_resolution.csv` retains the primary record locators and confidence fields without tweet text.

## Split procedure

1. Before quarantine, build transitive components of tweet ID, identical image bytes, identical decoded pixels, raw text hash and normalized text hash, plus 81 visually confirmed duplicate/shared-asset candidate edges.
2. Normalization: HTML unescape, NFKC, casefold, remove URLs, leading RT mention prefix and mentions, join Unicode word tokens. Equal cleaned inputs are conservatively grouped; they are not necessarily the same source post.
3. Quarantine 7 discrepant rows and 9 confirmed placeholders. For a component spanning eligible splits, retain original **test > dev > train** rows in that order; exclude lower-priority members. No retained record moves split. No primary split seed is used.
4. Lock row IDs and hashes in `A3-M1/remediation/manifests/filtered_original/`. Training seeds 42/43/44 are plans, not executed runs. Backup repartition seed 20261104 is historical and backup only.

All 17 original exact-image groups/54 records are traced. Of the 134 reviewed cross-split pHash candidates: 70 duplicates, 11 shared-asset variants, 38 distinct observations and 15 placeholder edges (9 rows). Different forecast advisory observations are not merged merely for similar appearance. Additional 164 within-original-split candidates are conservative guards only in the backup protocol, not visually adjudicated duplicates.

The **951-pair test is custom filtered**, so published scores are contextual, not directly comparable. Events overlap across splits; no unseen-disaster generalization is established. Nine affected-individual dev/test records give weak rare-class support. Undetected semantic/edited duplicates can remain.

## Verified EDA and its populations

The original 8,079-pair pool had 0 empty texts, 0 corrupt/missing files and unique successful primary joins. Decodable placeholders were separately excluded. Original whitespace word median/p95 are 14/20 in each split; these are not subtoken statistics. Original pool flags include 652 heuristic encoding candidates and 24 images with a side below 64 pixels; filtered flag counts were not recomputed. All plots are copied from the verified audit; no new dataset audit was run for publication.

- `assets/m1/remediation_counts.png`: retained counts and final primary test distribution.
- `assets/m1/source_input_profiles.png`, `source_event_distribution.png`: **pre-filter 8,079-pair** source diagnostics.
- CSV evidence and prior check results are under `A3-M1/tables/` and `A3-M1/remediation/`.

## Planned preprocessing and models

Frozen ResNet-18 IMAGENET1K_V1: RGB, short-side resize 256, center crop 224, official ImageNet normalization. Frozen `distilbert/distilbert-base-uncased`: audited text normalization, tokenizer/mask/padding, proposed 128-subtoken limit (actual truncation unmeasured). Cache features keyed by immutable row/image IDs. No fitted preprocessing uses test labels.

Four models are planned: image only, text only, 128-dimensional concatenation fusion, and scalar gated intermediate fusion. The gate ablation fixes it to 0.5. Macro-F1 is primary; accuracy, per-class scores/confusion matrix, operational modality disagreement with paired review, error analysis and end-to-end cost are planned. Frozen encoders and cached heads target modest compute; no hardware allocation or training results are asserted.

No additional subset is used for reported experiments. Any optional training-only smoke subset must keep whole groups, class coverage, seed/IDs and be excluded from main reported results. Full primary dev/test remain fixed. See the proposal for planned optimizer settings, resource estimates and M2/M3 milestones.

## Primary references

- [Alam, Ofli and Imran, CrisisMMD (ICWSM 2018)](https://arxiv.org/abs/1805.00713).
- [Ofli, Alam and Imran, multimodal disaster response (ISCRAM 2020)](https://arxiv.org/abs/2004.11838).
- [Arevalo et al., Gated Multimodal Units (2017)](https://arxiv.org/abs/1702.01992).
- [Official ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Official DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).
- Course Project Handbook, CO3133 Semester 261, Part I, Sections 7.6 and 24-32; [course LMS](https://lms.hcmut.edu.vn/course/view.php?id=142848). The group reports prior approval was waived; no formal approval is claimed.
