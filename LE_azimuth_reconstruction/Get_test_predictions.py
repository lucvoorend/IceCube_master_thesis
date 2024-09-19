print('Starting importing all the necessary packages')
from distutils.log import debug
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax


from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge

from graphnet.models.graphs import KNNGraph

#Which type of classification do we wanna use (MulticlassClassificationTask or BinaryClassificationTask)
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa

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

print('All is imported')

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]



def main(
    input_path: str,
    output_path: str,
    model_path: str,
):

    #Event_no selection. If you don't need this just outcomment i
    test_selection = pd.read_parquet("/groups/icecube/cjb924/workspace/work/reconstruction/train/selections/only_neutrinos.parquet").sample(frac=1, replace=False, random_state=1).reset_index(drop = True)['event_no'].ravel().tolist()

    #Testing selection to check that the code is running properly. Always do this before doing the full prediction.
    test_selection = test_selection[1_000_000:1_200_000]

    # Print the length of the test selection
    print('Length of test selection:', len(test_selection))

    # Configuration. Same setup as your model
    config = {
        "db": input_path,
        "pulsemap": "SplitInIcePulses",
        "batch_size": 512,
        "num_workers": 15,
        "accelerator": "gpu",
        "devices": [1], #Which GPU you wanna use. Always take the ones that is not in use.
        "target": "zenith",
        "n_epochs": 1,
        "patience": 1, #Patience is not needed as we are nt training.
    }

    archive = output_path

    # Name of the run you are doing. An example is given below.
    run_name = "model_trained_on_{}__more_neccesary_information".format(
        config["target"]
    )

    # Module that defines what the GNN sees. Essential for running graphnet (still under development so keep that in mind if it doesn't work with your current graphnet).
    graph_definition = KNNGraph(
        detector = IceCubeDeepCore(),
        nb_nearest_neighbours=8,
        input_feature_names=features
    )

    prediction_dataloader_RD = make_dataloader(
        db = config["db"],
        pulsemaps = config["pulsemap"],
        features = features,
        truth = truth,
        selection = test_selection,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        graph_definition = graph_definition
    )

    #Buil of the model
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    #The task you have done.
    task = ZenithReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=VonMisesFisher2DLoss(),
    )

    #Define the model.
    model = StandardModel(
        #detector=detector,
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
    )

    #Using early stopping.
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    #Define your trainer. Here we use pytorch Trainer.
    trainer = Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger=wandb_logger,#If not using WandB change this to None.
        logger = None,
    )

    # Load your existing model
    model.load_state_dict(model_path)

    print("Starting prediction on test set")

    # predict and save predictions to file
    resultsRD = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = prediction_dataloader_RD,
        # State what you wanna predict. Example below is for multiclassification for "pid".
        prediction_columns =[config["target"] + "_pred", config["target"] + "_kappa"],
        additional_attributes=[config["target"], "event_no"],
    )
    #Saving the model. Here you also give the name
    resultsRD.to_csv(
        output_folder + "/results_test.csv"
    )

    print("Predictions saved to file")

#Run your main function. Here you also give the input db, output folder and model path (important that it ends with state_dict.pth)
if __name__ == "__main__":
    # Input database path
    input_db = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"
    # Output folder path
    output_folder = "/groups/icecube/luc/Workspace/GraphNet/work/LE_zenith_reconstruction/Result_CSVs"
    model_path = "/groups/icecube/luc/Workspace/GraphNet/work/LE_zenith_reconstruction/Trained_model/dynedge_zenith_Luc_regression_LE_state_dict.pth"


    main(input_db, output_folder, model_path)