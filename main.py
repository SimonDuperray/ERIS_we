from Merger import Merger
from RNN import RNNModel
from Analyzer import Analyzer
from ReportWriter import ReportWriter
from os import listdir
from os.path import join
import tensorflow as tf
import json

def average_list(li):
    return sum(li)/len(li)

# TODO: fitting observations
# TODO: confidence percentage of prediction

# ========================
#      ITERATIVE WAY
# ========================

# datasets = [
#     (
#         './datasets/bibtex/3000e/cor_in_3000e.bibtex',
#         './datasets/bibtex/3000e/incor_a_in_3000e.bibtex',
#         './datasets/bibtex/3000e/incor_b_in_3000e.bibtex'
#     ), (
#         './datasets/bibtex/10000e/cor_in_10000e.bibtex',
#         './datasets/bibtex/10000e/incor_a_in_10000e.bibtex',
#         './datasets/bibtex/10000e/incor_b_in_10000e.bibtex'
#     )
# ]

# small datasets to test algo
datasets = [
    (
        './datasets/bibtex//correct.bibtex',
        './datasets/bibtex//not_correct_a.bibtex',
        './datasets/bibtex//not_correct_b.bibtex'
    )
]

# gru_neurons = [32, 64, 128, 256, 512]
gru_neurons = [128]

# epochs = [i for i in range(10, 110, 10)]
epochs = [2, 4]

for dataset in datasets:
    # create dataset
    merger = Merger()
    merger.run(
        path_1=dataset[0],
        path_2=dataset[1],
        path_3=dataset[2]
    )

    # instanciate new ReportWriter
    report_writer = ReportWriter()

    # report title
    report_title = "Experimental Report for dt_lgth:" + str(merger.dflgth)
    filename = "experimental_report_"+str(merger.dflgth)+".md"

    # write the first page of the report
    report_writer.write_first_page(
        filename=filename,
        title=report_title,
        iter_data={"gru_neurons": gru_neurons, "epochs": epochs},
        params={
            "batch_size ": 300,
            "validation_split ": 0.2,
            "dataset_path ": "./datasets/",
            "learning_rate ": 1e-3,
            "dropout_value ": 0.5,
            "dense_neurons ": 128,
            "loss_function ": "sparse_categorical_crossentropy",
            "dataset_length ": str(merger.dflgth),
            "percentage True Negatives ": str(merger.percentage),
            "percentage names only ": str(merger.percentage_names_only),
            "percentage other tags ": str(merger.other_tags)
        },
        dtlgth=str(merger.dflgth)
    )

    # train models
    for gru_neuron in gru_neurons:
        for epoch in epochs:
            # instanciate new model
            rnn_model = RNNModel(df_length=str(merger.dflgth))

            # then train it
            rnn_model.run(
                nb_epochs=epoch,
                batch_size=300,
                validation_split=0.2,
                dataset_path=merger.get_dataset_location(),
                gru_neurons=gru_neuron,
                learning_rate=1e-3,
                dropout_value=0.5,
                dense_neurons=256,
                loss_function=None,
                gru_neuron=gru_neuron
            )

            # instanciate new Analyzer()
            analyzer = Analyzer(epochs=epoch)  # OR EPOCH ?

            # get current folder's name
            directories = listdir("./histories/"+str(merger.dflgth))
            for directory in directories:
                if int(str(directory[2])) == epoch:
                    dir=directory
                    files = listdir(join("./histories/"+str(merger.dflgth)+"/", directory))
                else:
                    try:
                        if int(str(directory[2])+str(directory[3])) == epoch:
                            dir=directory
                            files = listdir(join("./histories/"+str(merger.dflgth)+"/", directory))
                        else:
                            pass
                    except ValueError as e:
                        print(f"Error: {e}")

                file = files[0] if files[0][-4:] == "json" else files[1]

            # get path for acc and loss
            path_acc = "./analysis/compar_epochs/"+str(merger.dflgth)+"/acc_valacc_"+str(epoch)+"_"+str(gru_neuron)+".png"
            path_loss = "./analysis/compar_epochs/"+str(merger.dflgth)+"/loss_valloss_"+str(epoch)+"_"+str(gru_neuron)+".png"

            # analyze metrics and save model and json files
            analyzer.analyze_for_report(
                dir=dir,
                file=file,
                dtlgth=merger.dflgth,
                path_acc=path_acc,
                path_loss=path_loss
            )
            
            # get data from file
            with open("./histories/"+str(merger.dflgth)+"/"+dir+"/"+file, 'r') as outfile:
                data = json.load(outfile)

            curves_data = {}
            curves_data['accuracy'] = {
                "tr_min": min(data['accuracy']),
                "tr_max": max(data['accuracy']),
                "tr_mean": average_list(data['accuracy']),
                "val_min": min(data['val_accuracy']),
                "val_max": max(data['val_accuracy']),
                "val_mean": average_list(data['val_accuracy'])
            }
            curves_data['loss'] = {
                "tr_min": min(data['loss']),
                "tr_max": max(data['loss']),
                "tr_mean": average_list(data['loss']),
                "val_min": min(data['val_loss']),
                "val_max": max(data['val_loss']),
                "val_mean":average_list(data['val_loss']) 
            }

            fitting_observations = {
                "under": "Yes",
                "over": "No",
                "good": "Partially"
            }

            predictions = {
                'to_predict': data['init_seq'],
                'expected': data['to_predict'],
                'predicted': data['prediction']
            }

            print("\n\n\n======\n"+str(rnn_model.get_to_predict().split())+"\n======\n\n\n")
            print("\n\n\n======\n"+str(rnn_model.get_predicted().split())+"\n======\n\n\n")

            bilan = {
                "gap_acc": max(data['accuracy']),
                "gap_loss": min(data['loss']),
                "fit_status": "underfitted",
                "pred_percent": analyzer.confidence_percentage(
                    li=[data['to_predict'].split(), data['prediction'].split()]
                )
            }

            # expected=rnn_model.get_to_predict().split(),
            # predicted=rnn_model.get_predicted().split()

            # append new page to report
            report_writer.write_unique_epoch_page(
                gru_neuron=gru_neuron,
                epoch=epoch,
                acc_valacc_absolute_path="."+path_acc,
                loss_valloss_absolute_path="."+path_loss,
                curves_data=curves_data,
                fitting_observations=fitting_observations,
                predictions=predictions,
                bilan=bilan
            )

            bilan = {}

        report_writer.write_last_page(
            dtlgth=str(merger.dflgth),
            gru_neuron=gru_neuron
        )
