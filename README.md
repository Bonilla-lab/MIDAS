**Introduction**

![Overhead Figure](https://github.com/user-attachments/assets/7b275f2f-25fb-4b52-85bc-0cd0e0eac081)

This repository contains the Jupyter Notebook & Google Collab link to an image analysis workflow to segment and quantify droplets in microscopy images (MIcroscopic Droplet AnalysiS-MIDAS). It is written for Python 3.10 and based on the existing segmentation algorithm Cellpose. The notebook segments and then extracts data from the images. It has been specifically adapted to segment droplets with large variations in size within the same image. 

**The Google Collab can be accessed through** https://colab.research.google.com/drive/1E9HyJT9hR5GUlWsP921ijaXZRIa2Ma5X#scrollTo=O98bFwVe_9mR

Please make sure to create a copy in your own Google Drive under File -> "Save a copy in Drive"

**For more information and use-cases please see the following pre-print paper** 
-https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4941219

If you apply the workflow, please cite as 

_"Saalbrink, Jens and Loo, Tricia Y. J. and Mertesdorf, Julia and Xu, Peidi and Pedersen, Mie T. and Clausen, Mathias Porsmose and Bonilla, Jose C., Quantifying Microscopic Droplets in Colloidal Systems Through Machine Learning-based Image Analysis Available at SSRN: https://ssrn.com/abstract=4941219 or http://dx.doi.org/10.2139/ssrn.4941219"_


**The repository includes:**
-
- A link to the Google Collab (https://colab.research.google.com/drive/1E9HyJT9hR5GUlWsP921ijaXZRIa2Ma5X#scrollTo=O98bFwVe_9mR)
- A Jupyter Notebook of the google collab for offline use. This can be done through installing Jupyter Notebook on Anaconda or other Python distributions, for this please make sure to download and incorporate the two .py files included in the repository. For more information for offline installation and how Jupyter Notebooks run, please consult **https://docs.jupyter.org/en/latest/**
- pystatistics.py & midassegmentation.py for the loops and operations
- A post-trained Cellpose model using images from food colloids, specifically emulsion and foam droplets to enhance the native capabilities of the Cellpose algorithm
- A folder of examplary images of emulsion droplets through Confocal, Brightfield and Coherent Anti-stokes Raman Scattering Microscopy you are free to test this or other image analysis algortihms in case you dont have your own.

Python Libraries used:

- Simple-ITK, numpy, pandas & Skimage: To integrate image analysis operations directly into the workflow
- CV2: For contour extraction of Regions-Of-Interest
- Tabulate, MatplotLib & Csv: For Data Visualization
- Cellpose: The original machine-learning algorithm
- google.collab: For native intergration with Google Collab runtimes & Drive

