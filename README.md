## Environment

Create and activate the conda environment:

```bash
conda env create -f DIDRF.yml
conda activate DIDRF
```

Alternatively, install the Python dependencies with:

```bash
pip install -r requirements-windows.txt
```

## Data

Download the LETOR datasets and place them under the relative paths configured
in `LTRlocal_dataset_info.txt`, for example:

```text
data/MQ2008/Fold1/
data/MSLR-WEB30K/Fold1/
data/istella-s/sample/
```

Dataset links:

- MQ2008: download the supervised LETOR 4.0 MQ2008 package from the Microsoft
  LETOR page: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
  The original direct package URL is:
  http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2008.rar
- MSLR-WEB30K: download from the Microsoft Learning to Rank Datasets page:
  https://www.microsoft.com/en-us/research/project/mslr/
  Use the `MSLR-WEB30K` link after accepting the dataset agreement.
- Istella-S LETOR: download from the official Istella LETOR page:
  https://istella.ai/datasets/letor-dataset/
  Use the `download Istella-S LETOR here` link.

After extraction, the directory names should match `LTRlocal_dataset_info.txt`.
For example, the MQ2008 archive should contain `Fold1` to `Fold5`; place those
fold directories under `data/MQ2008/`. For MSLR-WEB30K, place `Fold1` to
`Fold5` under `data/MSLR-WEB30K/`. For Istella-S, place the sample files under
`data/istella-s/sample/` or update the path in `LTRlocal_dataset_info.txt`.

If your data is stored elsewhere, edit `LTRlocal_dataset_info.txt` before
running experiments.

## Income Simulation Data

The two real-data-calibrated income simulation scenarios used by the experiments
are generated as trajectory banks:

```text
data/income_trajectories/criteo_cva.npz
data/income_trajectories/yoochoose_rpv.npz
```

These `.npz` files are not included in this anonymous code package. They can be
rebuilt from the raw Criteo and YOOCHOOSE files using the script below.

The simulation code is in `utils/simulation.py`. At runtime, `f_d` automatically
loads the corresponding trajectory bank when `--income_simulation=criteo_cva`
or `--income_simulation=yoochoose_rpv` is used. If a bank file is missing, the
same code falls back to deterministic semi-synthetic curves with the same
scenario assumptions.

The trajectory bank schema is:

```text
trajectories: float array with shape [n_entities, n_time_bins]
entity_ids: optional string array with shape [n_entities]
```

The raw source datasets are not included because they are large and should be
downloaded under their own data terms:

- Criteo-CVA scenario: calibrated from the Criteo Attribution Modeling for
  Bidding dataset. Dataset page:
  https://huggingface.co/datasets/criteo/criteo-attribution-dataset
  The Criteo AI Lab dataset index is:
  https://ailab.criteo.com/ressources/
- YOOCHOOSE-RPV scenario: calibrated from the RecSys Challenge 2015 YOOCHOOSE
  click/buy dataset. Challenge page:
  https://recsys.acm.org/recsys15/challenge/
  A public archive mirror is available at:
  https://zenodo.org/records/7412233

To rebuild the derived trajectory banks from raw files, place the raw files
under `data/raw_income/` and run:

```bash
python scripts/build_income_trajectories.py criteo_cva --input data/raw_income/criteo_attribution_dataset.tsv.gz --sep "\t" --has_header
python scripts/build_income_trajectories.py yoochoose_rpv --clicks data/raw_income/yoochoose-clicks.dat --buys data/raw_income/yoochoose-buys.dat
```

To check whether the runtime will use generated banks or fallback curves:

```bash
python scripts/check_income_trajectory_banks.py
```

## Running DIDRF

Offline relevance setting:

```bash
python main.py --progressbar=false --rankListLength=5 --query_least_size=5 --n_iteration=10000 --queryMaximumLength=10000000000 --relvance_strategy=TrueAverage --positionBiasSeverity=1 --dataset_name=MQ2008 --fairness_strategy=didrf --income_simulation=criteo_cva --fairness_tradeoff_param=1000 --exploration_tradeoff_param=0 --random_seed=0 --log_dir=localOutput/
```

Online relevance-estimation setting:

```bash
python main.py --progressbar=false --rankListLength=5 --query_least_size=5 --n_iteration=10000 --queryMaximumLength=10000000000 --relvance_strategy=EstimatedAverage --positionBiasSeverity=1 --dataset_name=MQ2008 --fairness_strategy=didrf --income_simulation=criteo_cva --fairness_tradeoff_param=1000 --exploration_tradeoff_param=0 --random_seed=0 --log_dir=localOutput/
```

Available ranking strategies include `didrf`, `MCFair`, `FairCo`, `FairK`,
`Randomk`, `ExploreK`, `Topk`, `FARA`, `MMF`, `PLFair`, `TaxRankIncome`, `ILP`,
and `LP`.








