from Merger import Merger
from RNN import RNNModel
from Analyzer import Analyzer
from ReportWriter import ReportWriter
import keras.losses


# ====================
#       MERGER
# ====================
# merger = Merger()
# merger.run(
#     path_1='./datasets/bibtex/correct.bibtex',
#     path_2="./datasets/bibtex/not_correct_a.bibtex",
#     path_3='./datasets/bibtex/not_correct_b.bibtex'
# )

# ==================== 
#        PP+RNN
# ====================
# rnn_model = RNNModel()
# rnn_model.set_percent(merger.get_percent())

# ==================== 
#  ITERATIVE TRANINGS
# ====================
# hyperparameters
# epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# epochs = [2, 4, 6]

# for epo in epochs:
#     rnn_model.run(
#         nb_epochs=epo,
#         batch_size=300,
#         validation_split=0.2,
#         dataset_path=merger.get_dataset_location()
#     )

# ==================== 
#       ANALYZIS
# ====================
# analyzer = Analyzer(epochs=epochs)
#
# analyzer.analyze_epochs()
# # analyzer.analyze_predictions()
# analyzer.create_plots_figure(
#     length_epochs_list=len(epochs)
# )


# ====================
#   CREATE REPORTS
# ====================
def generate_header(epochs_list, batch_size, vocab_infos):
    header = "# Experimental Report v.0\n"
    # hyperparameters
    header += "## Hyperparameters\n"
    header += "* epochs list: " + str(epochs_list) + "\n"
    header += "* batch_size: " + str(batch_size) + "\n"
    header += "* dataset_length: " + str(merger.get_i_o_length()) + "\n"
    header += "* percent of TN: " + str(merger.get_percent()) + "%\n"
    header += "* percent of unique names: " + str("Unknown") + "\n"
    # vocab infos
    header += "## Vocab informations\n"
    # input
    header += "### Input vocabulary\n"
    header += "* Total words: " + str(vocab_infos[0]['total_words']) + "\n"
    header += "* Unique words: " + str(vocab_infos[0]['unique_words']) + "\n"
    header += "* Max sequence length: " + str(vocab_infos[0]['max_io_sentence_length']) + "\n"
    header += "* Vocab size: " + str(vocab_infos[0]['io_voc_size']) + "\n"
    # output
    header += "### Output vocabulary\n"
    header += "* Total words: " + str(vocab_infos[1]['total_words']) + "\n"
    header += "* Unique words: " + str(vocab_infos[1]['unique_words']) + "\n"
    header += "* Max sequence length: " + str(vocab_infos[1]['max_io_sentence_length']) + "\n"
    header += "* Vocab size: " + str(vocab_infos[1]['io_voc_size']) + "\n"

    # iter epochs
    header += "## Analysis of metrics for different number of epochs\n"
    header += "Accuracy                   |  Loss\n"
    header += ":-------------------------:|:-------------------------:\n"
    header += "![Accuracy](/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/4350/acc.png)|  " \
              "![Loss](/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/4350/loss.png)\n"

    # for epoch in epochs_list:
    header += "### " + str(epochs_list) + " epochs\n"
    header += "![test](/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/4350/comp_acc_valacc_" + str(
        epochs_list) + ".png)"
    header += "![test](/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/4350/com_loss_valloss_" + str(
        epochs_list) + ".png)\n"
    header += "\nPrediction: blablabla\n"
    header += "#### Observations:\n"
    header += "Voici mes observations\n"

    # to_return
    return header


# for epo in epochs:
#     path = "./experimental_reports/"
#     filename = str(epo)+"_epo.md"
#     with open(path+filename, "w") as md_file:
#         md_file.write(generate_header(
#             epochs_list=epo,
#             batch_size=300,
#             vocab_infos=rnn_model.get_infos_obj()
#         ))

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
epochs = [2, 5, 10]

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

    # write the first page of the report
    report_writer.write_first_page(
        title=report_title,
        iter_data=[gru_neurons, epochs],
        params={
            "batch_size": 300,
            "validation_split": 0.2,
            "dataset_path": "./datasets/",
            "learning_rate": 1e-3,
            "dropout_value": 0.5,
            "dense_neurons": 128,
            "loss_function": "sparse_categorical_crossentropy",
            "dataset_length: ": str(merger.dflgth),
            "percentage True Negatives: ": str(merger.percentage),
            "percentage names only": str(merger.percentage_names_only),
            "percentage other tags: ": str(merger.other_tags)
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
                loss_function=keras.losses.sparse_categorical_crossentropy
            )
    #
    #         # instanciate new Analyzer()
    #         analyzer = Analyzer(epochs=epochs)  # OR EPOCH ?
    #
    #         # analyze metrics and save model and json files
    #         analyzer.analyze_epochs(subfolder=str(merger.dflgth))
    #
    #         # create matplotlib graphs based on json files
    #         # don't need it for the moment, but keep unique graph template to send it to the report
    #         # analyzer.create_plots_figure(length_epochs_list=len(epochs))
    #
    #         # append new page to report
    #         report_writer.write_unique_epoch_page(
    #             gru_neuron=gru_neuron,
    #             epoch=epoch,
    #             acc_valacc_figure=None,
    #             loss_valloss_figure=None,
    #             fitting_observations=analyzer.fit_obs(),
    #             predictions=analyzer.analyze_predictions(),
    #             bilan=analyzer.bilan()
    #         )

    report_writer.write_last_page(str(merger.dflgth))
