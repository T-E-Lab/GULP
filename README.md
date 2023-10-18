# gulp2p
Tools for defining regions of interests (ROIs) and calculating changes in fluorescence over time (DF/F) for two-photon calcium imaging experiments

### Organization:
The gulp2p package contains the following submodules:
* **DataClass**
* **ImagingPreProc**
* **PlottingFunctions**
* **ROIs**
* **utils**

In addition the **scripts** folder contains notebooks that illustrate how to use functions in this module


### Install:
I recommend using poetry to setup a custom conda environment. A helpful introduction can be found [here](https://ealizadeh.com/blog/guide-to-python-env-pkg-dependency-using-conda-poetry).

0. Clone repo, navigate into folder
1. If you don't already have poetry, [install poetry](https://python-poetry.org/docs/#installation). You may need to close command window and open a new one.
2. Create conda environment:  
 `conda create --name gulp2p python=3.9`
4. Activate environment:  
 `conda activate gulp2p`
6. Make sure you are in the top folder of the cloned repo, then install dependencies:  
 `poetry install`
8. Setup the new environment as an ipython kernel:  
    `conda install -c anaconda ipykernel`  
    then  
    `python -m ipykernel install --user --name=gulp2p`
    
    If you get an error when opening the example notebooks, run:
    `conda install nbconvert==5.4.1`
    
    If you get an error importing modules, run:
    `conda install -c conda-forge charset-normalizer=2.1.0`
Now you should be able to run the example notebooks in the **scripts** folder without problems. 
