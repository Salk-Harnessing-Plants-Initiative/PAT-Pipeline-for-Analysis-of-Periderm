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


# Quick Start Guide for PAT 1.0

Welcome to the PAT (Pipeline for Analysis of Periderm) 1.0 Quick Start Guide. Follow these steps to get up and running with PAT on your system.

## Step 1: Installation

1. **Clone the repository:**

   For all systems, clone the PAT repository using Git:
   ```bash
   git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
   ```
   Navigate to the cloned directory:
   ```bash
   cd PAT-Pipeline-for-Analysis-of-Periderm
   ```

2. **Create and activate the conda environment:**

   For **Linux** and **Windows** users:
   ```bash
   conda env create -f environment.yml  # Use this for Linux
   conda env create -f environment_win.yml  # Use this for Windows
   conda activate PAT-Pipeline-for-Analysis-of-Periderm
   ```
   
   For **Mac** users:
   ```bash
   conda env create -f environment_mac.yml
   conda activate PAT-Pipeline-for-Analysis-of-Periderm
   ```

## Step 2: Download Models and Images

- Download pre-trained models and sample images from the provided link:
  [Download Models and Images](https://drive.google.com/drive/folders/13F_TSJNYKEM3DVrvaFU56FUzt8BJ9m7X?usp=sharing)
  
- Place the downloaded `models` folder in the `PAT-Pipeline-for-Analysis-of-Periderm` directory.

## Step 3: Running PAT

- **For Ubuntu (Recommended for NVIDIA GPU with at least 6GB memory):**
  ```bash
  python PAT_GUI.py
  ```

- **For Windows:**
  ```bash
  python PAT_GUI_win_cpu.py
  ```

- **For Mac (Not recommended for segmentation due to slower performance):**
  ```bash
  python PAT_GUI_mac_cpu.py
  ```

## Step 4: Using PAT

1. Load your images into PAT GUI; the tool supports `.tif`, `.png`, `.xpm`, `.jpg`, and `.bmp` formats.

2. Run the **Pre-process** to convert images to `.png` format if needed.

3. Use the **Segment** button to perform image segmentation.

4. Perform **Quality Control (QC)** by selecting high-quality segmentation results within the QC GUI.

5. Navigate the QC images using the right and left arrow keys.

6. Click **Phenotyping** to measure root lengths, with options for quick or detailed analysis.

7. After Phenotyping, choose whether to save the QC data.

8. Use the **Visualization** button for a quick view of the length measurements in a boxplot.

9. Upon completion, you will be prompted to exit PAT, delete temporary folders, or save results to a new folder.

This will download and set up all the necessary libraries, including a Python 3.8 installation.

<strong><em>If environment.yml doesn't include all libraries you need, please use "pip install XXXX" to install them or contact me.</em></strong>

&nbsp;<br>
&nbsp;<br>
<strong> The following contains details about how to run PAT 1.0. <strong>

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
&nbsp;<br>
&nbsp;<br>
<span style="color: red">
    <strong>Please put the `models` folder in the `PAT-Pipeline-for-Analysis-of-Periderm` folder.</strong>
</span>

&nbsp;<br>
&nbsp;<br>

$$\color{red}Download \space models \space and \space sample \space images \space using \space the \space above \space link.$$
&nbsp;<br>
$$\color{red}Please \space put \space the \space downloaded \space models \space folder \space in \space the \space PAT-Pipeline-for-Analysis-of-Periderm \space folder.$$ 
&nbsp;<br>
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/download_and_save.png)
&nbsp;<br>
&nbsp;<br>
PAT GUI as following: 
&nbsp;<br>
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/start.png)
&nbsp;<br>
&nbsp;<br>
You could load images in tif, png, xpm, jpg, bmp formats. After loading images, you can run Pre-process to convert images to png format. If the images you loaded are in png format, you could skip Pre-process step. 

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/loading.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/loading_folders.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/pre_processing.png)

And then, you could click Segment button to segment whole roots and periderm only. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/segmentation.png)

Then, you could click "Quality Control" button to do post-procession which can fill the gaps (the gaps were caused by dark or blur) based on the context information from both sides. If you don't want to do any QC, you could skip this step and just simply go to Phenotyping.
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/post_processing.png)

We designed a Qulity Control GUI which you could use to quickly select high quality segmentation results. In order to make easy to visualize and compare the segmentation results, especially junction parts between periderm and endodermis, original images and corresponding segmentated images of junction parts between periderm and endodermis are vertically concatenated and images for QC are saved in output/for_QC. <strong><em>Since it takes time to generate concatenated images for QC, please be patient and wait for the QC GUI to show up. :hourglass_flowing_sand:</em></strong>

<strong> Please use the right arrow key to navigate to the next image and the left arrow key to return to the previous image in the QC (Quality Control) GUI.<strong>

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/qc1.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/qc2.png)

$$\color{red} Please \space navigate \space to \space <strong>PAT-Pipeline-for-Analysis-of-periderm/output/for_QC<strong>.$$

In QC GUI, you could click left arrow to previous image or right arrow to next image; when segmentation pass the QC, please click Select buton on GUI. In case you want to un-select the image, you could go to the image which you don't want to select and click Not-Select button. Selected the images which pass QC will be saved in "selected_image_names.txt". 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/qc3.png)
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/QC_output.png)

When you click Phenotyping button, a popup window will ask if you want to measure the whole root length: if you choose Yes, the both whole root lengths and periderm lengths will be measured and save to "whole_root_length.csv" (<strong><em>It will take time to measure the whole root lengths when you don't want to calculate length quickly due to high-resolution images. :hourglass_flowing_sand:</em></strong>); I included option which you could select "calculate length quickly" through measuring in re-sized images and measurements between high-res and low-res were not significant different, so I recommend to select "calculate length quickly" except you want to get very accurate measurement. If you choose No to "Do you want to measure the whole root length?", only periderm lengths will be measured and save to "periderm_length.csv". Also, you could convert the length in the number of pixels to micrometer based on your own pxiel/micrometer ratio (e.g. our pixel/micrometer ratio is 0.5299 as default).
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/phenotyping_options.png)

After Phenotyping, a popup window will ask if you want to save phenotyping data after QC: if choose Yes, "periderm_length_after_QC.csv" will be generated. 
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/if_save_QC.png)

When Phenotyping done, you could click Visualization button to get a quick view the the periderm and/or whole root lengths in boxplot. 

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/boxplot.png)

When you done and try to close the GUI, you will be asked if you want exit PAT, delete temporary folders, save results to new folder. 

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/save_results.png)

## Using the Tool in Mac (Not Recommend since slow in Segmentation step! It is OK if you would like to try the pipeline for few images)
As of last update in 2023, using CUDA for deep learning on a Mac can be challenging due to hardware and software compatibility issues: 1) Apple has not included NVIDIA GPUs in its Mac lineup for several years; 2) Macs don't come with NVIDIA GPUs, they cannot natively support CUDA;3) The last version of macOS to support CUDA was macOS Mojave (10.14). NVIDIA has not released CUDA drivers for macOS versions beyond Mojave. So CPU need to be used if you want to run PAT in Mac. (When I used Ubuntu in my laptop (i7-11800H, NVIDIA RTX A4000 GPU), it took ~ 14 minutes to finish analysis of four images and took 5 minutes to segment; When I used CPU (9th gen 6-core Intel Core i9 2.3 GHz), it took ~ 58 minutes.)
```bash
git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
```
```bash
cd PAT-Pipeline-for-Analysis-of-Periderm
```
```bash
conda env create -f environment_mac.yml
```
Activate the environment using the following command:
```bash
conda activate PAT-Pipeline-for-Analysis-of-Periderm
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
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/win_begin.png)
```bash
git clone https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm.git
```
```bash
cd PAT-Pipeline-for-Analysis-of-Periderm
```
```bash
conda env create -f environment_win.yml
```
Activate the environment using the following command:
```bash
conda activate PAT-Pipeline-for-Analysis-of-Periderm
```

```bash
python PAT_GUI_win_cpu.py
```

![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/PAT_win.png)

<strong> We developed new QC GUI for Windows with Previous button and Next button <strong>
![PAT GUI](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm/blob/main/resources/readme/new_QC_GUI_for_win.png)
## Training new models
Training code may be found in the training folder. Instructions on training models are given in the training README. If you would like to collaborate on the development of new models for PAT 1.0, please contact us.

## Contact
PAT 1.0 is published in Plant Phenomics. For enquiries please contact wbusch@salk.edu, gvillarino@salk.edu, linzhang@salk.edu.

## License

[MIT](https://choosealicense.com/licenses/mit/)

