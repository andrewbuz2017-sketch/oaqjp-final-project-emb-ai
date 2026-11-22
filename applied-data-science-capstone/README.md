# Applied Data Science Capstone: SpaceX landing prediction

Andrii Buz. Historical IBM course data.

## Reproduce

```bash
python -m pip install -r requirements.txt
python spacex_analysis.py
python spacex_dashboard.py
```

The dashboard serves port 8050. For a hosted reverse proxy, set `DASH_REQUESTS_PREFIX` to the public application prefix. The app contains launch-site and payload filters, success pies, payload/outcome scatter plots, a Folium launch map, expanded launch records, and proximity maps.

## Analysis

- 90 Falcon 9 launches for visual EDA and classification, including 60 supplied Class=1 outcomes.
- 101 historical records for SQLite exercises.
- 56 geographic records for maps and dashboard figures. These populations are kept separate.
- Pipeline-based imputation, scaling and categorical encoding fitted within training folds.
- Stratified 72/18 train/test split, seed 2. Five-fold stratified training CV, seed 42.
- Logistic regression, SVM, decision tree and KNN hyperparameter comparisons.

The selected decision tree achieved 90.3% mean training CV accuracy and 83.3% holdout accuracy. Its confusion matrix is `[[3,3],[0,12]]`. The 18-launch holdout contains only six failures, of which three were incorrectly predicted as successful. Balanced accuracy is 75%. A chronological 72/18 sensitivity split gave 88.9% accuracy against an 83.3% later-period majority baseline.

## Source handling and limitations

All inputs use public frozen IBM course snapshots. The archived Wikipedia request returned HTTP 403 in this environment. The script records the actual status and uses the supplied historical CSV for downstream SQL. Its HTML parser is also validated on a rendering of that supplied table. No successful archived-page scrape is claimed for this run.

The supplied target includes controlled ocean landings; landing does not imply recovery or reuse. Core reuse attributes may be retrospective and need an as-of audit. Do not treat these small-sample results as operational launch forecasts or price estimates. Proximity distances use explicit illustrative reference points, rather than an authoritative nearest-network GIS analysis.

## Outputs

`spacex_output/` contains source snapshots, query results, `queries.sql`, descriptive charts, model scores, confusion matrix, Folium HTML maps and Plotly figure exports. The executed notebook documents the same code in seven sections. The final presentation is submitted separately as a PDF.
