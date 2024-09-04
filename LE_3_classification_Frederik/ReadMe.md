# Multi classification of Neutrino's, Muons and Noise in LE data
***
## The data
path to database:

    "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"

path to selection files:

- train:
     '/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_train_event_no.csv'

- validation: 
    '/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_val_event_no.csv'

- test: 
    '/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_test_event_no.csv'

## Training parameters
- model: DynEdge
- loss: CrossEntropy
- lr scheduler: PiecewiseLinear
- lr: 1e-4
- eps: 1e-3
- early stopping: 5
- Inference transform: Softmax

## Performance
- Performance only evaluated on the validation set, due to balancing problems with the test set. 
    - 96.39% accuracy on classifying Neutrinos
    - 95.90% on Muons
    - 98.40% on Noise
- Logit(p) values deviate from results by Frederik
- Overall comparable results to previous research


