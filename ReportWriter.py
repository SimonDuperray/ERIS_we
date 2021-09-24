class ReportWriter:
    def __init__(self):
        print("=== ReportWriter Initialized! ===")
        self.path = "./experimental_reports"
        self.filename = ""

    def write_first_page(self, title, iter_data, params, dtlgth):
        self.filename = title+".md"
        fp = ""
        fp += "# "+title+"\n"
        fp += "\n---\n\n<center>\n\nTest on:\n"
        fp += "\nepochs: "+str(iter_data[0])+"\n"
        fp += "\ngru_neurons: "+str(iter_data[1])+"\n"
        fp += "</center>\n\n---\n\n"
        fp += "# Parameters\n"
        for key in params:

            buffer = "\n* "+str(key)+": "+str(params[key])+"\n"
            fp += buffer
        fp += "<br>\n"

        with open(self.path+"/"+self.filename, 'w') as md_file:
            md_file.write(fp)
            print("=====================================")
            print(f"{self.filename} saved in {self.path}")
            print("=====================================")

    def write_unique_epoch_page(self, gru_neuron, epoch,
                                acc_valacc_figure, loss_valloss_figure,
                                fitting_observations, predictions, bilan):
        with open(self.path+"/"+self.filename, 'a') as md_file:
            pass

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
        model_plot_path = "/home/ing-angers/duperrsi/Documents/idm-ml/dev_ia/analysis/compar_epochs/"+str(dtlgth)+"/model_"+str(dtlgth)+".png"
        lp += "\n![plot_model]("+model_plot_path+")"

        with open(self.path+"/"+self.filename, "a") as md_file:
            md_file.write(lp)
