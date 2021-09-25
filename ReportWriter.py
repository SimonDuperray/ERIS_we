class ReportWriter:
    def __init__(self):
        print("=== ReportWriter Initialized! ===")
        self.path = "./experimental_reports"
        self.filename = ""

    def write_first_page(self, filename, title, iter_data, params, dtlgth):
        self.filename = filename
        fp = ""
        fp += "# "+title+"\n"
        fp += "\n---\n\n<center>\n\nTest on:\n"
        fp += "\nepochs: "+str(iter_data[0])+"\n"
        fp += "\ngru_neurons: "+str(iter_data[1])+"\n"
        fp += "\n</center>\n\n---\n\n"
        fp += "# Parameters\n"
        for key in params:
            buffer = "\n* "+str(key)+": "+str(params[key])+"\n"
            fp += buffer
        fp += "\n<br>\n"

        with open(self.path+"/"+self.filename, 'w') as md_file:
            md_file.write(fp)
            print("=====================================")
            print(f"{self.filename} saved in {self.path}")
            print("=====================================")

    def write_unique_epoch_page(self, gru_neuron, epoch,
                                acc_valacc_absolute_path, loss_valloss_absolute_path,
                                curves_data,
                                fitting_observations, predictions, bilan):
        
        content = ""
        content += "\n\n# GRU: "+str(gru_neuron)+" - Epochs: "+str(epoch)

        # learning curves
        content += "\n\n## Learning curves"
        # acc
        content += "\n\n![accuracy plot]("+str(acc_valacc_absolute_path)+")\n"
        content += "\n\n<center>\n"
        content += "\n|          |      Training data      |  Validation data |"
        content += "\n|:--------:|:-----------------------:|:----------------:|"
        content += "\n| minimum  | "+str(curves_data['accuracy']['tr_min'])+" | "+str(curves_data['accuracy']['val_min'])+" |"
        content += "\n| maximum  | "+str(curves_data['accuracy']['tr_max'])+" | "+str(curves_data['accuracy']['val_max'])+" |"
        content += "\n|   mean   | "+str(curves_data['accuracy']['tr_mean'])+" | "+str(curves_data['accuracy']['val_mean'])+" |\n"
        content += "\n\n</center>\n\n"
        # loss
        content += "\n![loss plot]("+str(loss_valloss_absolute_path)+")\n"
        content += "\n\n<center>\n\n"
        content += "\n|          |      Training data      |  Validation data |"
        content += "\n|:--------:|:-----------------------:|:----------------:|"
        content += "\n| minimum  | "+str(curves_data['loss']['tr_min'])+" | "+str(curves_data['loss']['val_min'])+" |"
        content += "\n| maximum  | "+str(curves_data['loss']['tr_max'])+" | "+str(curves_data['loss']['val_max'])+" |"
        content += "\n|   mean   | "+str(curves_data['loss']['tr_mean'])+" | "+str(curves_data['loss']['val_mean'])+" |\n"
        content += "\n\n</center>\n\n"

        # fitting osbervations
        content += "\n## Fitting observations"
        content += "\n\n<center>\n"
        content += "\n|          | Undefitting | Good fitting | Overfitting |"
        content += "\n|:--------:|:-----------:|:------------:|:-----------:|"
        content += "\n|  Status  | "+str(fitting_observations['under'])+" | "+str(fitting_observations['good'])+" | "+str(fitting_observations['over'])+" |\n"
        content += "\n\n</center>\n\n"

        # predictions
        content += "\n## Predictions"
        content += "\n\n<center>\n"
        content += "\n|          | Seq. to predict | Expected seq. | Predicted seq. |"
        content += "\n|:--------:|:---------------:|:-------------:|:--------------:|"
        content += "\n|  Seq.  | "+str(predictions['to_predict'])+" | "+str(predictions['expected'])+" | "+str(predictions['predicted'])+" |\n"
        content += "\n\n</center>\n\n"

        # bilan
        content += "\n## Bilan"
        content += "\nFor "+str(gru_neuron)+" neurons in GRU and "+str(epoch)+" epochs, the training accuracy is gapped at "+str(bilan['gap_acc'])+", the training loss at "+str(bilan['gap_loss'])+". The model is "+str(bilan['fit_status'])+" and the prediction is correct at "+str(bilan['pred_percent'])+"%.\n"

        with open(self.path+"/"+self.filename, 'a') as md_file:
            md_file.write(content)

    def write_last_page(self, dtlgth):
        lp = ""

        # add model summary
        with open("./analysis/compar_epochs/" + str(dtlgth) + "/model_" + str(dtlgth) + ".h5", "r") as model_file:
            model = [line.strip('\n') for line in model_file]

        model_str = ""
        for mod in model:
            if "=" not in mod:
                model_str += mod
                model_str += "\n"

        lp += "\n# Model Summary\n"
        lp += model_str

        # add model plot
        lp += "\n# Plotted Model"
        # TODO
        absolute_model_plot_path = "D:\\ERIS\dev_ia\\analysis\\compar_epochs\\"+str(dtlgth)+"\\model_"+str(dtlgth)+".png"
        # model_plot_path = "/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/"+str(dtlgth)+"/model_"+str(dtlgth)+".png"
        lp += "\n![plot_model]("+absolute_model_plot_path+")"

        with open(self.path+"/"+self.filename, "a") as md_file:
            md_file.write(lp)
