# PAT-Pipeline-for-Analysis-of-Periderm
# PAT 1.0

Welcome to the PAT 1.0 repository. This README and the repository code are continuously being updated as we prepare for publication. Please check back regularly for new features and documentation!

## Installing PAT 1.0

To get started with PAT 1.0, first download the code. You can do this either as a zip file from this page or by cloning the git repository (recommended):

```bash
git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
```
After cloning the repository, install the required dependencies. For Linux users, use the following commands:
```bash
cd PAT-Pipeline-for-Analysis-of-Periderm
conda env create -f environment.yml
```
For users on other operating systems, we recommend using Anaconda due to the complexity of library support. Install Anaconda and then create a new environment using the provided yml dependencies file:

```bash
conda env create -f environment.yml
conda activate PAT-Pipeline-for-Analysis-of-Periderm
```
This will download and set up all the necessary libraries, including a Python 3.8 installation.

## Using the Tool

The majority of users will want to run PAT 1.0 on new images, in which case all the code you need is in the XXX folder. You can find more instructions in the inference README.We developed a GUI as following which you can run through:
```bash
python PAT_GUI.py
```
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/start.png)

You could load images in tif, png, xpm, jpg, bmp formats. After loading images, you can run Pre-process to convert images to png format. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/loading.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/pre_processing.png)


## Training new models
Training code may be found in the training folder. Instructions on training models are given in the training README. If you would like to collaborate on the development of new models for PAT 1.0, please contact us.

## Contact
PAT 1.0 is published in Plant Phenomics. For enquiries please contact wbusch@salk.edu, gvillarino@salk.edu, linzhang@salk.edu.

## License

[MIT](https://choosealicense.com/licenses/mit/)

