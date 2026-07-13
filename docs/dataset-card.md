# Dataset Card: CFPB Complaint Narratives

## Source

- **Publisher:** United States Consumer Financial Protection Bureau
- **Dataset:** [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **API:** [Official API documentation](https://cfpb.github.io/api/ccdb/api.html)
- **API license:** CC0 / United States Government work

The CFPB database generally updates daily. Complaint narratives are published only when
the consumer opts to share them and after the CFPB takes steps to remove personal
information. The CFPB does not verify each narrative and warns that the database is not
a statistical sample of all consumer experiences.

## Fields used

| Project field | CFPB API field | Purpose |
|---|---|---|
| `complaint_id` | `complaint_id` | Stable record identity and reproducibility checks |
| `date_received` | `date_received` | Approved source-period filtering |
| `text` | `complaint_what_happened` | NLP input after normalization |
| `label` | `product` | Six-class routing target |

No company, location, demographic, response, issue, or outcome field is used as a model
feature.

## Collection contract

The downloader requests only records with public narratives across five bounded periods
from 1 August 2023 through 31 December 2025. It first attempts a chronological candidate
pool through the official API. If the CFPB edge service denies API access, it reads
deterministically selected, non-overlapping byte ranges from the official uncompressed
CSV export and filters the same period and labels. The download manifest records the
source mode, ranges, counts, and checksum. The processed dataset then uses deterministic
sampling so reruns against the same source records select the same data.

The CSV fallback collects at least 4,250 candidates per class, providing a 1,250-record
cleaning buffer above the final 3,000-class target. Candidate records and completed
range offsets are checkpointed locally so an interrupted, target-expanded, or
range-expanded run can resume.

## Preparation

1. Retain only the six registered product labels.
2. Normalize HTML entities, Unicode form, whitespace, and repeated redaction markers.
3. Retain natural language without stemming, lemmatization, or stop-word removal.
4. Remove missing narratives and normalized text shorter than 20 characters.
5. Limit model input to the first 2,000 normalized characters.
6. Remove exact duplicate narratives.
7. Remove every narrative whose normalized text appears under conflicting labels.
8. Sample 3,000 records per class with seed 42.
9. Split each class into 70% training, 15% validation, and 15% sealed test data.

The manifest contains class/split counts and SHA-256 checksums. Narrative CSV files are
excluded from Git.

The realized processed dataset contains 18,000 records: 12,600 training, 2,700
validation, and 2,700 sealed-test records. Every class contributes exactly 3,000
records. Its SHA-256 checksum is
`8727affded9c28e0e3886264c34a53a8a7705d81216bd5d744f943a6cf3d7719`.

## Known limitations

- Publication is opt-in, so narratives are subject to selection bias.
- The data represents submitted US consumer-finance complaints, not all consumers or
  financial-service interactions.
- Narratives are consumer accounts and are not independently verified by the CFPB.
- Product taxonomy and complaint behavior can change over time.
- English-only modeling may behave unpredictably on other languages.
- Truncating long narratives can remove relevant information.
- Explicit product words can inflate routing performance relative to ambiguous messages.
- CFPB scrubbing reduces but cannot make all re-identification risk impossible.

## Ethical use

Use the dataset only for aggregate analysis and routing research. Do not publish raw
narratives in reports, use predictions to judge consumers or companies, or infer facts
not represented by the routing label. Keep downloaded copies local and delete them when
they are no longer required.
