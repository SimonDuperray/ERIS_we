def merger(path_1, path_2, path_3):
	pass
	
class ReportWriter:
	def write_first_page(
		title,
		iter_data,
		params
	):
		pass
	
	def write_unique_epoch_page(
		gru_neurons,
		epochs,
		acc_valacc_figure,
		loss_valloss_figure,
		fitting_observations,
		predictions,
		bilan
	):
		pass

datasets = [
			('cor_in_4350.bibtex', 'incor_a_in_4350.bibtex', 'incor_b_in_4350.bibtex'),
			('cor_in_87774.bibtex', 'incor_a_in_87774.bibtex', 'incor_b_in_87774.bibtex')
		]
		
gru_neurons = [32, 64, 128, 256, 512]

epochs = [i for i in range(10, 110, 10)]

for dataset in datasets:
	# bibtex files to csv shuffled dataset
	merger(
			path_1 = dataset[0], 
			path_2 = dataset[1], 
			path_3 = dataset[2]
	)
	
	report_title = "Experimental report for "+str(merger.get_df_length())+" length dataset"
	
	# write header of the report
	report_writer = ReportWriter()
	report_writer.write_first_page(
		title=report_title,
		iter_data=[gru_neurons, epochs],
		params=[300, 0.2, "./blablabla", 1e-3, 0.5, 128, 'sparse_categorical_crossentropy']
	)
	
	# train models
	for gru_neuron in gru_neurons:
		for epoch in epochs:
			# train model
			rnn_model.run(
				nb_epochs = epoch,
				batch_size = 300,
				validation_split = 0.2,
				dataset_path = merger.get_dataset_location(),
				gru_neurons = gru_neuron,
				learning_rate = 1e-3,
				dropout_value = 0.5,
				dense_neurons = 256,
				loss_function = keras.losses.sparse_categorical_crossentropy
			)
			
			# instanciate new Analyzer
			analyzer = Analyzer()

			# analyze metrics and save model and json files
			analyzer.analyze_epochs()
			
			# create graphs with json files
			analyzer.analyzer.create_plots_figure(
				length_epochs_list=len(epochs)
			)
			
			# append new page (GRU=fixed; epochs=iter) to current report
			report_writer.write_unique_epoch_page(
				gru_neurons=gru_neuron,
				epochs=epoch,
				acc_valacc_figure=acc_valacc_fig,
				loss_valloss_figure=loss_valloss_fig,
				fitting_observations=analyzer.fit_obs(to_define_plt_or_json),
				predictions=analyzer.get_predictions(),
				bilan=analyzer.bilan()
			)
