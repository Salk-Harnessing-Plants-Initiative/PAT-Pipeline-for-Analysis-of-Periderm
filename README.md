# PAT-Pipeline-for-Analysis-of-Periderm
# PAT 1.0

Welcome to the PAT 1.0 repository. This README and the repository code are continuously being updated as we prepare for publication. Please check back regularly for new features and documentation!
This section explains the data structure used in Salk HPI Project

## Data Structure in PAT

Here's an overview of the top-level data structure in PAT:

```plaintext
├── resources
│   ├── images
│   └── readme  
├── src
│   ├── scripts
├── models
│   ├── pre-trained models
│   └── class data
├── PAT_GUI.py
├── PAT_GUI_mac_cpu.py
├── PAT_GUI_win_cpu.py
├── README.md
├── environment.yml
├── environment_mac.yml 
└── environment_mac.yml

```

## Installing PAT 1.0

To get started with PAT 1.0, first download the code. You can do this either as a zip file from this page or by cloning the git repository (recommended):

```bash
git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
```
After cloning the repository, install the required dependencies. For Linux users, use the following commands:
```bash
cd PAT-Pipeline-for-Analysis-of-Periderm
```

For users on other operating systems, we recommend using Anaconda due to the complexity of library support. Install Anaconda and then create a new environment using the provided yml dependencies file:

```bash
conda env create -f environment.yml
```

For windows users, please create the environment using the following command:
```bash
conda env create -f environment_win.yml
```

Activate the environment using the following command:
```bash
conda activate PAT-Pipeline-for-Analysis-of-Periderm
```

This will download and set up all the necessary libraries, including a Python 3.8 installation.

<strong><em>If environment.yml doesn't include all libraries you need, please use "pip install XXXX" to install them or contact me.</em></strong>

## Using the Tool in Ubuntu (Recommend with NVIDIA GPU with memory => 6GB)

The majority of users will want to run PAT 1.0 on new images, in which case all the code you need is in the XXX folder. You can find more instructions in the inference README.We developed a GUI as following which you can run through:
```bash
python PAT_GUI.py
```

For windows user, please use the following script to start the PAT GUI:
```bash
python PAT_GUI_win_cpu.py
```

<span style="color: red">
    <strong>Please download pre-trained models and sample images from the following link:</strong> 
    <a href="https://drive.google.com/drive/folders/13F_TSJNYKEM3DVrvaFU56FUzt8BJ9m7X?usp=sharing">click here to download models and sample images</a>
</span>

<span style="color: red">
    <strong>Please put the `models` folder in the `PAT-Pipeline-for-Analysis-of-Periderm` folder.</strong>
</span>

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/start.png)

You could load images in tif, png, xpm, jpg, bmp formats. After loading images, you can run Pre-process to convert images to png format. If the images you loaded are in png format, you could skip Pre-process step. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/loading.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/loading_folders.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/pre_processing.png)

And then, you could click Segment button to segment whole roots and periderm only. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/segmentation.png)

Then, you could click "Quality Control" button to do post-procession which can fill the gaps (the gaps were caused by dark or blur) based on the context information from both sides. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/post_processing.png)

We designed a Qulity Control GUI which you could use to quickly select high quality segmentation results. In order to make easy to visualize and compare the segmentation results, especially junction parts between periderm and endodermis, original images and corresponding segmentated images of junction parts between periderm and endodermis are vertically concatenated and images for QC are saved in output/for_QC. <strong><em>Since it takes time to generate concatenated images for QC, please be patient and wait for the QC GUI to show up. :hourglass_flowing_sand:</em></strong>


![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/qc1.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/qc2.png)

Please select `./output/for_QC` folder as QC input folder.

In QC GUI, you could click left arrow to previous image or right arrow to next image; when segmentation pass the QC, please click Select buton on GUI. Selected the images which pass QC will be saved in "selected_image_names.txt". 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/qc_window_3.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/QC_output.png)

When you click Phenotyping button, a popup window will ask if you want to measure the whole root length: if you choose Yes, the both whole root lengths and periderm lengths will be measured and save to "whole_root_length.csv" (<strong><em>It will take time to measure the whole root lengths. :hourglass_flowing_sand:</em></strong>); If you choose No, only periderm lengths will be measured and save to "periderm_length.csv". 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/whole_root_measure.png)

After Phenotyping, a popup window will ask if you want to save phenotyping data after QC: if choose Yes, "periderm_length_after_QC.csv" will be generated. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/if_save_QC.png)

When Phenotyping done, you could click Visualization button to get a quick view the the periderm and/or whole root lengths in boxplot. 

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/boxplot.png)

When you done and try to close the GUI, you will be asked if you want exit PAT, delete temporary folders, save results to new folder. 

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/save_results.png)

## Using the Tool in Mac (Not Recommend since slow in Segmentation step! It is OK if you would like to try the pipeline for few images)
As of last update in 2023, using CUDA for deep learning on a Mac can be challenging due to hardware and software compatibility issues: 1) Apple has not included NVIDIA GPUs in its Mac lineup for several years; 2) Macs don't come with NVIDIA GPUs, they cannot natively support CUDA;3) The last version of macOS to support CUDA was macOS Mojave (10.14). NVIDIA has not released CUDA drivers for macOS versions beyond Mojave. So CPU need to be used if you want to run PAT in Mac. 
```bash
git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
```
```bash
cd PAT-Pipeline-for-Analysis-of-Periderm
```
```bash
conda env create -f environment_mac.yml
```
```bash
python PAT_GUI_mac_cpu.py
```
Then you could follow the steps as shown above.
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/PAT_Mac.png)

## Using the Tool in Windows 
You can use Command Prompt in Windows as shown in following, and then please use python 3.8 through installing in Microsoft Store. Before using PAT, you could install conda [Visit Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html "Go to conda Website"). After installed conda, you need to add conda to your system's PATH environment using the followign command in Command Prompt:
```bash
%UserProfile%\miniconda3\condabin\activate
```

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/win_python38.png)
```bash
git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
```
```bash
cd PAT-Pipeline-for-Analysis-of-Periderm
```
```bash
conda env create -f environment_win.yml
```
```bash
conda activate PAT_win
```

```bash
python PAT_GUI_win_cpu.py
```

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/PAT_win.png)
## Training new models
Training code may be found in the training folder. Instructions on training models are given in the training README. If you would like to collaborate on the development of new models for PAT 1.0, please contact us.

## Contact
PAT 1.0 is published in Plant Phenomics. For enquiries please contact wbusch@salk.edu, gvillarino@salk.edu, linzhang@salk.edu.

## License

[MIT](https://choosealicense.com/licenses/mit/)

