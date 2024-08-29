# gulp2p
Tools for defining regions of interests (ROIs), calculating changes in fluorescence over time (DF/F) for two-photon calcium imaging experiments and analyzing drosophila behavior.

## Table of Contents:
- [Table of Contents:](#table-of-contents)
- [Install](#install)
- [Organization](#organization)

## Install
1. Clone repo, navigate into folder
1. Create conda environment:  
 `conda create -n gulp2p -c conda-forge -c anaconda caiman=1.10 python=3.10`
1. Activate environment:  
 `conda activate gulp2p`
1. Install gulp2p package:  
   `pip install -e .`

Now you should be able to run the example notebooks in the **scripts** folder without problems. 

## Organization
The gulp2p package contains the following submodules:
```
gulp2p
	preproc
		behavior.py
		draw.py
		experiment.py
		imaging.py
		tiff.py
		trial.py
		utils.py
	viz
		plotting.py
		video.py
		utils.py
	analysis
		head_direction.py
		utils.py
		stats.py
	config.py
```

In addition the `scripts` folder contains notebooks that illustrate how to use functions in this module. There is also a settings file `settings.yaml.example` which can be modified to point to data directories and change image processing settings. Importantly, it contains the path to `user_data_table.yaml` which contains a list of lab members and their data directories. This information is used to categorize data by its procurer when plotting and generate video summaries of the trials (not yet implemented).

