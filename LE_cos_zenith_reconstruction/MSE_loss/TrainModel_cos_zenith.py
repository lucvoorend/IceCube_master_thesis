print('Start importing')
from distutils.log import debug
import logging
import os
import sys
sys.path.append('/groups/icecube/luc/Workspace/GraphNet/graphnet/src')

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.data.batch import Batch

from graphnet.training.loss_functions import CrossEntropyLoss, VonMisesFisher2DLoss, LogCoshLoss, MSELoss, VonMisesFisher3DLoss
from graphnet.data.constants import FEATURES, TRUTH
# from graphnet.data.sqlite.sqlite_selection import (
#     get_desired_event_numbers,
#     get_equal_proportion_neutrino_indices,
# )
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
#changed 09/10-2023
#from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.graphs import KNNGraph

from graphnet.data.dataset.sqlite import SQLiteDataset
from graphnet.training.labels import Label


#Which type of classification do we wanna use (MulticlassClassificationTask or BinaryClassificationTask)
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa
from graphnet.models.task.reconstruction import CosZenithReconstruction
from graphnet.models.task.reconstruction import AzimuthReconstructionWithKappa

from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
    make_dataloader
)
from graphnet.models.graphs import GraphDefinition
from graphnet.models.detector.icecube  import  IceCube86

import numpy as np
import pandas as pd
import csv
import sqlite3

torch.set_float32_matmul_precision('medium')

print('All is imported')

aimingDirPath = "/groups/icecube/luc/playground/Results_full_data/"
# oscNewMuonsDirPath = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/"
# if not os.path.exists(aimingDirPath):
#     os.makedirs(aimingDirPath)


#logger = get_logger()
# set increased verbose information when debugging.
#logger.setLevel(logging.DEBUG)
# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]
#truth.append('cos_zenith')

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run

#wandb_logger = WandbLogger(
#    project="cjb924",
#    #entity="frederik-icecube",
#    name='Peter_work_script_2',
#    save_dir=WANDB_DIR,
#    #log_model=True,
#    log_model=False,
#)

import argparse
#print('WandB initialized')

#Trainingfiles, validation and test
'''
dataDirPath = '/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/'
trainDataFile = dataDirPath + 'Multiclassification_train_event_no.csv'
valDataFile = dataDirPath + 'Multiclassification_val_event_no.csv'
testDataFile = dataDirPath + 'Multiclassification_test_event_no.csv'

train_selection = pd.read_csv(trainDataFile).reset_index(drop = True)['event_no'].ravel().tolist()
validation_selection = pd.read_csv(valDataFile).reset_index(drop = True)['event_no'].ravel().tolist()
test_selection = pd.read_csv(testDataFile).reset_index(drop = True)['event_no'].ravel().tolist()
# train_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_train_event_no.csv').reset_index(drop = True)['event_no'].ravel().tolist()
# validation_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_val_event_no.csv').reset_index(drop = True)['event_no'].ravel().tolist()
# test_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_test_event_no.csv').reset_index(drop = True)['event_no'].ravel().tolist()

#Test script. Only takes the first 1000/100/100 events to check if script works.
train_selection = train_selection[:1000]
validation_selection = validation_selection[:100]
test_selection = test_selection[:100]
'''

selection_nu = pd.read_parquet("/groups/icecube/cjb924/workspace/work/reconstruction/train/selections/only_neutrinos.parquet").sample(frac=1, replace=False, random_state=1).reset_index(drop = True)['event_no'].ravel().tolist()
train_selection = selection_nu[:800000]
validation_selection = selection_nu[800000:1000000]
test_selection = selection_nu[1000000:1200000]

"""
train_selection = train_selection[:1000]
validation_selection = validation_selection[:100]
test_selection = test_selection[:100]
"""

train_selection = np.array(train_selection)
validation_selection = np.array(validation_selection)
test_selection = np.array(test_selection)

print('Length of training data:', len(train_selection))
print('Length of validation data:', len(validation_selection))
print('Length of test data:', len(test_selection))


parser = argparse.ArgumentParser(
    description="plotting the predicted zenith and azimuth vs truth."
)

#Give the path to the database here
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_db",
    type=str,
    help="<required> path(s) to database [list]",
    #Database!!! Path
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db",
    # required=True,
)

#Give the output directory here
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="<required> the output path [str]",
    #output directory
    default=aimingDirPath,
    # required=True,
)

#Pulsemap (should almost always be SplitInIcePulses)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="<required> the pulsemap to use. [str]",
    default="SplitInIcePulses",
    # required=True,
)

#Below is unnecessary (has been overwritten below)
parser.add_argument(
    "-n",
    "--event_numbers",
    dest="event_numbers",
    type=int,
    help="the number of muons to train on; if too high will take all available. [int]",
    default=int(7500000*3),
)
#Choice of GPU /comment out for cpu-usage
parser.add_argument(
    "-g",
    "--gpu",
    dest="gpu",
    type=int,
    help="<required> the name for the model. [str]",
    #GPU
    default=1# required=True,
)
parser.add_argument(
    "-b",
    "--batch_size",
    dest="batch_size",
    type=int,
    help="<required> the name for the model. [str]",
    default=512,
    # required=True,
)
parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    type=int,
    help="<required> the name for the model. [str]",
    default=150,
    # required=True,
)
parser.add_argument(
    "-w",
    "--workers",
    dest="workers",
    type=int,
    help="<required> the number of cpu's to use. [str]",
    default=15,
    # required=True,
)

#Give the name of the file you wanna create
parser.add_argument(
    "-r",
    "--run_name",
    dest="run_name",
    type=str,
    help="<required> the name for the model. [str]",
    #New name
    # default='Frederik_multiclass_test_of_db'#"last_one_lvl3MC_SplitInIcePulses_21.5_mill_equal_frac"
    default='Luc_logcosh_cosZenith_reconstruction_full_db'
    # required=True,
)
parser.add_argument(
    "-a",
    "--accelerator",
    dest="accelerator",
    type=str,
    help="<required> the name for the model. [str]",
    #cpu change
    default="gpu"
    #default="gpu"
    # required=True,
)

args = parser.parse_args()

print('Argparse done, defining main loop')
# Main function definition
def main():

    #logger.info(f"features: {features}")
    #logger.info(f"truth: {truth}")

    # Configuration
    config = {
        "db": args.path_to_db,
        "pulsemap": args.pulsemap,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "accelerator": args.accelerator,
        #Change for cpu-usage
        "devices": [args.gpu],#1
        "target": "cos_zenith",
        "n_epochs": args.epochs,
        "patience": 5, #Early stopping, how many epochs without improvements
    }
    config["archive"] = args.output
    config["run_name"] = "dynedge_{}_".format(config["target"]) + args.run_name
    print('before logs to wand')
    # Log configuration to W&B
    #change
    #wandb_logger.experiment.config.update(config)
    print('after logs to wand')


    #graph_definition has replaced detector
    graph_definition = KNNGraph(
        detector = IceCubeDeepCore(),
        nb_nearest_neighbours=8,
        input_feature_names=features
    )

    #Manually change dataloader to train and validate on cos(zenith)
    train_set = SQLiteDataset(
        path=config["db"],
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        graph_definition=graph_definition,
        selection=train_selection
    )

    validation_set = SQLiteDataset(
        path=config["db"],
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        graph_definition=graph_definition,
        selection=validation_selection
    )

    '''
    print("Train set truth columns:", train_set.truth)
    truth_update = truth.append('cos_zenith')
    
    validation_set.truth = truth_update
    train_set.truth = truth_update

    print("Train set truth columns:", train_set.truth)
    '''

    # Load the db to pd dataframe
    con = sqlite3.connect(config["db"])
    query_train = "SELECT event_no, zenith FROM truth WHERE event_no IN ({})".format(
        ",".join([str(i) for i in train_selection])
    )
    query_val = "SELECT event_no, zenith FROM truth WHERE event_no IN ({})".format(
        ",".join([str(i) for i in validation_selection])
    )
    train_df = pd.read_sql(query_train, con)
    val_df = pd.read_sql(query_val, con)
    con.close()

    zenith_train_np = train_df["zenith"].to_numpy()
    zenith_val_np = val_df["zenith"].to_numpy()

    event_no_train = train_df["event_no"].to_numpy()
    event_no_val = val_df["event_no"].to_numpy()

    event_no_dict_train = {event_no_train[i]: i for i in range(len(event_no_train))}
    event_no_dict_val = {event_no_val[i]: i for i in range(len(event_no_val))}

    train_indices = np.array([event_no_dict_train[i] for i in train_selection])
    val_indices = np.array([event_no_dict_val[i] for i in validation_selection])

    event_no_train = event_no_train[train_indices]
    zenith_train = zenith_train_np[train_indices]
    zenith_val = zenith_val_np[val_indices]

    cos_zenith_train = np.cos(zenith_train)
    cos_zenith_val = np.cos(zenith_val)
    
    print('Event_no train:', event_no_train[0])
    print('Zenith train:', zenith_train[0])
    print('Cos zenith train:', cos_zenith_train[0])

    # For the training events:
    print('Adding custom cos(zenith) label to events')
    class DataTypeLabel_coszenith_train(Label):
        """Class for producing the 'cos_zenith' label"""
        def __init__(self):
            """Construct `DataTypeLabel`."""
            # Base class constructor
            super().__init__(key="cos_zenith")
        def __call__(self, graph: Data) -> torch.tensor:
            """Compute the 'cos_zentih' label for `graph`"""
            event_no = int(graph.event_no)
            index = np.where(train_selection == event_no)[0][0]
            cos_zenith = cos_zenith_train[index]
            label = torch.tensor([cos_zenith], dtype=torch.float64)
            return label

    # For the validation events:
    print('Adding custom cos(zenith) label to events')
    class DataTypeLabel_coszenith_val(Label):
        """Class for producing the 'cos_zenith' label"""
        def __init__(self):
            """Construct `DataTypeLabel`."""
            # Base class constructor
            super().__init__(key="cos_zenith")
        def __call__(self, graph: Data) -> torch.tensor:
            """Compute the 'cos_zentih' label for `graph`"""
            event_no = int(graph.event_no)
            index = np.where(validation_selection == event_no)[0][0]
            cos_zenith = cos_zenith_val[index]
            label = torch.tensor([cos_zenith], dtype=torch.float64)
            return label
    
    # Add the custom label to the training set and validation set
    cos_zenith_train_label = DataTypeLabel_coszenith_train()
    cos_zenith_val_label = DataTypeLabel_coszenith_val()

    train_set.add_label(cos_zenith_train_label)
    validation_set.add_label(cos_zenith_val_label)

    training_dataloaders = DataLoader(
        dataset=train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        collate_fn=Batch.from_data_list,
        #drop_last=True,
    )
    print(f"Length of train dataloader: {len(training_dataloaders)}")
    validation_dataloaders = DataLoader(
        dataset=validation_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        collate_fn=Batch.from_data_list,
    )
    print(f"Length of valid dataloader {len(validation_dataloaders)}")

    #Dataloader for test
    '''
    test_dataloader = make_dataloader(db = config['db'],
                                            selection = test_selection, #config["test_selection"],
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            #Newly changed
                                            graph_definition = graph_definition,
                                            shuffle = False)
    '''

    ## Building model
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    #Multiclassification
    '''
    task = MulticlassClassificationTask(
        nb_outputs = 3,
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=CrossEntropyLoss(options={1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}),
        #loss_function=CrossEntropyLoss(),
        transform_inference=lambda x: softmax(x,dim=-1),
        )
    '''

    task = CosZenithReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=MSELoss(),
    )

    #print(task.target_labels)

    #Model, can be optimized depending on the task
    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        #optimizer_kwargs={"lr": 1e-04, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloaders) / 2,
                len(training_dataloaders) * config["n_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        #changed
        #logger=wandb_logger,
        logger = None,
    )

    try:
        trainer.fit(model, training_dataloaders, validation_dataloaders)
        #trainer.fit(model, training_dataloader, validation_dataloader, ckpt_path='/groups/icecube/luc/playground/lightning_logs/version_23/checkpoints/epoch=4-step=7815.ckpt')
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Predict on Validation Set and save results to file
    print('Saving validation results')
    results_val = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = validation_dataloaders,
        prediction_columns =[config["target"] + "_pred"],
        additional_attributes=[config["target"], "event_no"],
    )
    save_results(config["db"], config["run_name"] + '_validation_set', results_val, config["archive"], model)
    '''
    print('Saving test results')
    results_test = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = test_dataloader,
        prediction_columns =[config["target"] + "_noise_pred", config["target"] + "_muon_pred", config["target"]+ "_neutrino_pred"],
        additional_attributes=[config["target"], "event_no"],
    )
    save_results(config["db"], config["run_name"] + '_test_set', results_test, config["archive"], model, 'test')
    '''
    print('Saving config')
    model.save_config(aimingDirPath)

    print('Luc wants to print this')

    print('Done')
# Main function call
if __name__ == "__main__":
    print('Before main loop')
    main()