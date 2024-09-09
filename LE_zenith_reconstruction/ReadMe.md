# Zenith angle regression LE
***
## The data
path to database:

    "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"

path to selection files:
- general: 
    "/groups/icecube/cjb924/workspace/work/reconstruction/train/selections/only_neutrinos.parquet"
- train:
    [:800_000]

- validation: 
    [800_000:1_000_000]

- test: 
    [1_000_000:1_200_000]

## Training parameters
- model: DynEdge
- loss: VonMisesFisher2DLoss
- lr scheduler: PiecewiseLinear
- lr: 1e-3
- eps: 1e-3
- early stopping: 5

## Performance
- Very similar to Results of Frederik
    - Good overal performance, but errors near the edges (zenith = 0 or pi)
    - Residuals follow expected distribution
    - No Z-score analysis has been done (yet)


