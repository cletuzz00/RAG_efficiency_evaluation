Place your evaluation data here.

Expected files:

- `business_corpus.csv` with columns:
  - `id` (string or int)
  - `title` (string)
  - `text` (string)
  - optional: `tags` (string)

- `queries_answers.csv` with columns:
  - `query_id` (string or int)
  - `query` (string)
  - `expected_answer` (string)
  - optional: `relevant_doc_ids` (comma-separated list of document IDs)

These files are consumed by `src/dataset.py` and the experiment runner.

