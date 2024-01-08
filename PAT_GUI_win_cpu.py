import sys
import os
import cv2
import shutil
import subprocess
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QProgressBar, QMessageBox,QInputDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from tkinter import Tk, messagebox
import pandas as pd
Image.MAX_IMAGE_PIXELS = None  # Removes the limit completely
#base_path = os.path.dirname(os.path.dirname(__file__))
base_path = os.path.dirname(__file__)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create output and nature_accession folders
        output_path = os.path.join(base_path, 'output')
        nature_accession_path = os.path.join(base_path, 'nature_accession')
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(nature_accession_path, exist_ok=True)

        self.loaded_images = 0
        self.setWindowTitle("PAT Pipeline for Analysis of Periderm")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=1)

        self.load_button = QPushButton("Load Image")
        left_layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_image)

        self.preprocess_button = QPushButton("Pre-process")
        left_layout.addWidget(self.preprocess_button)
        self.preprocess_button.clicked.connect(self.preprocess_image)

        self.segment_button = QPushButton("Segment")
        left_layout.addWidget(self.segment_button)
        self.segment_button.clicked.connect(self.segment_image)

        self.postprocess_button = QPushButton("Post-process")
        left_layout.addWidget(self.postprocess_button)
        self.postprocess_button.clicked.connect(self.postprocess_image)

        self.quality_control_button = QPushButton("Quality Control")
        left_layout.addWidget(self.quality_control_button)
        self.quality_control_button.clicked.connect(self.quality_control)

        self.phenotyping_button = QPushButton("Phenotyping")
        left_layout.addWidget(self.phenotyping_button)
        self.phenotyping_button.clicked.connect(self.phenotyping)

        self.Visualization_button = QPushButton("Visualization")
        left_layout.addWidget(self.Visualization_button)
        self.Visualization_button.clicked.connect(self.Visualization)
        
        
        middle_layout = QVBoxLayout()
        main_layout.addLayout(middle_layout, stretch=3)

        self.progress_bars = []
        for _ in range(7):
            progress_bar = QProgressBar()
            middle_layout.addWidget(progress_bar)
            self.progress_bars.append(progress_bar)

        self.image_label = QLabel()
        main_layout.addWidget(self.image_label, stretch=2)
        self.image_label.setScaledContents(True)

        periderm_image_path = os.path.join(base_path,
                                  'resources', 'images','DALL_Periderm.png')
        self.image = cv2.imread(periderm_image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = self.resize_image(self.image)
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        exit_reply = QMessageBox.question(self, 'Exit Confirmation',
                                          "Are you sure you want to exit PAT?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
        if exit_reply == QMessageBox.Yes:
            delete_reply = QMessageBox.question(self, 'Delete Temporary Folders',
                                                "Would you like to delete temporary folders?",
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
            if delete_reply == QMessageBox.Yes:
                folders_to_delete = [
                    "Image_Padding", "Prediction_patch", "Image_Crop",
                    "Segmentation_temp", "Segmentation_upp_v15",
                    "segmentation_upp_periderm_v04", "seg_QC", "Seg_pad", "Ori_pad",
                    "nature_accession" ,"WB_whole_root" # Add the "nature_accession" folder to the list
                ]
    
                for folder in folders_to_delete:
                    folder_path = os.path.join(base_path, 'output', folder)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
    
            save_reply = QMessageBox.question(self, 'Save Confirmation',
                                              "Would you like to save and move result folders?",
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
            if save_reply == QMessageBox.Yes:
                dest_folder_path = QFileDialog.getExistingDirectory(self, "Select Destination Folder", "")
    
                if dest_folder_path:
                    folders_to_move = [
                        "Post_processing_v09", "Final_Periderm_Segmentation_Results",
                        "measurement", "for_QC"
                    ]
    
                    files_to_move = [
                        "periderm_length.csv", "periderm_length_micrometers.csv","selected_image_names.txt",
                        "periderm_length_after_QC.csv", "periderm_length_after_QC_micrometers.csv",
                        'whole_root_length.csv', "whole_root_length_micrometers.csv",'whole_root_length_boxplot.png', 'periderm_length_boxplot.png'
                    ]
    
                    for folder in folders_to_move:
                        src_folder_path = os.path.join(base_path, 'output', folder)
                        dest_path = os.path.join(dest_folder_path, folder)
    
                        if os.path.exists(src_folder_path):
                            shutil.move(src_folder_path, dest_path)
    
                    for file in files_to_move:
                        src_file_path = os.path.join(base_path, 'output', file)
                        dest_file_path = os.path.join(dest_folder_path, file)
    
                        if os.path.exists(src_file_path):
                            shutil.move(src_file_path, dest_file_path)
    
            # Empty the "nature_accession" folder
            nature_accession_path = os.path.join(base_path, "nature_accession")
            if os.path.exists(nature_accession_path):
                for file in os.listdir(nature_accession_path):
                    file_path = os.path.join(nature_accession_path, file)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
    
            # Accept the event to close the application
            event.accept()
        else:
            # Ignore the event if the user chooses not to exit
            event.ignore()

    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            exit_reply = QMessageBox.question(self, 'Exit Confirmation',
                                             "Are you sure you want to exit PAT?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
            if exit_reply == QMessageBox.Yes:
                delete_reply = QMessageBox.question(self, 'Delete Temporary Folders',
                                                    "Would you like to delete temporary folders?",
                                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
                if delete_reply == QMessageBox.Yes:
                    folders_to_delete = [
                        "Image_Padding", "Prediction_patch", "Image_Crop",
                        "Segmentation_temp", "Segmentation_upp_v15",
                        "segmentation_upp_periderm_v04", "seg_QC", "Seg_pad", "Ori_pad","WB_whole_root"
                    ]
    
                    for folder in folders_to_delete:
                        folder_path = os.path.join(base_path,'output', folder)
                        if os.path.exists(folder_path):
                            shutil.rmtree(folder_path)
    
                save_reply = QMessageBox.question(self, 'Save Confirmation',
                                                  "Would you like to save and move result folders?",
                                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
                if save_reply == QMessageBox.Yes:
                    dest_folder_path = QFileDialog.getExistingDirectory(self, "Select Destination Folder", "")
    
                    if dest_folder_path:
                        folders_to_move = [
                            "Post_processing_v09", "Final_Periderm_Segmentation_Results",
                            "measurement", "for_QC"
                        ]
    
                        files_to_move = [
                            "periderm_length.csv", "selected_image_names.txt"
                        ]
    
                        for folder in folders_to_move:
                            src_folder_path = os.path.join(base_path,'output', folder)
                            dest_path = os.path.join(dest_folder_path, folder)
    
                            if os.path.exists(src_folder_path):
                                shutil.move(src_folder_path, dest_path)
    
                        for file in files_to_move:
                            src_file_path = os.path.join(base_path,'output', file)
                            dest_file_path = os.path.join(dest_folder_path, file)
    
                            if os.path.exists(src_file_path):
                                shutil.move(src_file_path, dest_file_path)
    
                self.close()



    def resize_image(self, image, target_height=500, target_width=500):
        height, width, _ = image.shape
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        new_height = int(target_width / aspect_ratio)
        
        if new_width > target_width:
            new_width = target_width
        else:
            new_height = target_height

        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    

    def load_image(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        dest_folder = os.path.join(base_path,'nature_accession')
        
        if folder_path:
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".xpm", ".jpg", ".bmp", ".tif"))]
            total_images = len(image_files)
            
            for image_file in image_files:
                src_image_path = os.path.join(folder_path, image_file)
                dest_image_path = os.path.join(dest_folder, image_file)
                shutil.copy2(src_image_path, dest_image_path)
                
                self.image = cv2.imread(dest_image_path)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image = self.resize_image(self.image)
                height, width, channel = self.image.shape
                bytes_per_line = 3 * width
                q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image))
                
                self.loaded_images += 1
                progress_percentage = int((self.loaded_images / total_images) * 100)
                self.progress_bars[0].setValue(progress_percentage)        # ... (rest of the method)

    def run_script(self, script_path):
        os.system(f"python {script_path}")

    def preprocess_image(self):
        # Get the path of the 'nature_accession' folder relative to the script's directory
        path = os.path.join(os.path.dirname(__file__), 'nature_accession')
    
        # Check if the directory exists
        if not os.path.exists(path):
            QMessageBox.warning(self, "Directory Not Found", f"The directory {path} was not found.")
            return
    
        # Get a list of all .tif files
        tif_files = [f for f in os.listdir(path) if f.endswith(".tif")]
        num_files = len(tif_files)
    
        # Iterate through all files in the directory
        for index, f in enumerate(tif_files):
            file_path = os.path.join(path, f)
            print(f"Converting {file_path} to PNG...")
            
            # Open the .tif file and save it as a .png file
            with Image.open(file_path) as img:
                png_path = f'{os.path.splitext(file_path)[0]}.png'
                img.save(png_path)
            print("Conversion complete!")
    
            # Update the progress bar
            progress = ((index + 1) / num_files) * 100
            self.progress_bars[1].setValue(int(progress))
    
            # Remove the original .tif file
            os.remove(file_path)
            print(f"Removed original file: {f}")


    def segment_image(self):
         # Load and display the segmentation image
        segmentation_image_path = os.path.join(base_path, 'resources', 'images', 'segmentation.png')
        self.display_image(segmentation_image_path)
        # Define the mapping between log messages and progress values
        progress_mapping = {
            "Finish padding": 20,
            "Finish segmenting": 60,
            "Finish stitching": 70,
            "Finish post-processing": 80,
            "Finish converting": 90,
            "Finish!": 100,
        }

        # Run the script as a subprocess
        process = subprocess.Popen(["python", os.path.join(os.path.dirname(__file__),'src', "segmentation_patch_cor_win_cpu.py")], stdout=subprocess.PIPE, text=True)


        # Poll process for new output and update progress bar
        while True:
            line = process.stdout.readline()
            if not line:
                break
            
            # Check if the line matches any of the progress indicators
            for key, value in progress_mapping.items():
                if key in line:
                    self.progress_bars[2].setValue(value)
                    QApplication.processEvents()  # Process any pending events
        
        # Optionally, set progress bar to 100% once script completes
        self.progress_bars[2].setValue(100)

    def postprocess_image(self):
        segmentation_image_path = os.path.join(base_path, 'resources', 'images', 'postprocessing.png')
        self.display_image(segmentation_image_path)
        
        self.run_script(os.path.join(os.path.dirname(__file__), 'src',"new_post_processing_for_measurement.py"))
        self.progress_bars[3].setValue(100)

    def quality_control(self):
        segmentation_image_path = os.path.join(base_path, 'resources', 'images', 'QC.png')
        self.display_image(segmentation_image_path)
        
        self.run_script(os.path.join(os.path.dirname(__file__), 'src',"generate_images_for_QC.py"))
        self.progress_bars[4].setValue(50)
        self.run_script(os.path.join(os.path.dirname(__file__), 'src',"PAT_QC_GUI_test.py"))
        self.progress_bars[4].setValue(100)

    def phenotyping(self):
        segmentation_image_path = os.path.join(base_path, 'resources', 'images', 'phenotyping.png')
        self.display_image(segmentation_image_path)
        root = Tk()
        root.withdraw()
    
        # Popup to ask for quick measurement
        quick_measurement_reply = QMessageBox.question(self, 'Quick Measurement', 
                                                       "Do you want to calculate length quickly?",
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
        # Select the script based on the response
        if quick_measurement_reply == QMessageBox.Yes:
            periderm_script = "measurement_resize_periderm_lr.py"
            whole_root_script = "measurement_resize_whole_root_lr.py"
        else:
            periderm_script = "measurement_resize_periderm.py"
            whole_root_script = "measurement_resize_whole_root.py"
    
        # Run the periderm measurement script
        self.run_script(os.path.join(os.path.dirname(__file__), 'src', periderm_script))
    
        # Ask the user if they want to measure the whole root length
        whole_root_length_reply = QMessageBox.question(self, 'Measure Whole Root Length', 
                                                      "Do you want to measure the whole root length?",
                                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
        if whole_root_length_reply == QMessageBox.Yes:
            self.progress_bars[5].setValue(50)
            
            # Run wb_convert.py before running whole root length script
            self.run_script(os.path.join(os.path.dirname(__file__), 'src', "wb_convert.py"))
            self.run_script(os.path.join(os.path.dirname(__file__), 'src', whole_root_script))
            
            self.progress_bars[5].setValue(100)
        else:
            self.progress_bars[5].setValue(100)
    
        # After running the measurement scripts, ask the user if they want to save the QC data
        qc_response = messagebox.askyesno("Save QC Data", "Do you want to save phenotyping data after QC?")
        
        if qc_response:
            # If the user wants to save QC data, run QC_selected.py
            self.run_script(os.path.join(os.path.dirname(__file__), 'src', "QC_selected.py"))
    
        # Close the Tkinter window
        root.destroy()
    
        # After all phenotyping steps, ask for conversion ratio and convert pixel values
        self.convert_pixels_to_micrometers()


    def convert_pixels_to_micrometers(self):
        # Prompt user for conversion ratio
        conversion_ratio, ok = QInputDialog.getDouble(self, 'Conversion Ratio',
                                                      'Enter conversion ratio (default 0.5299):',
                                                      value=0.5299, decimals=4, min=0.0)
        
        if ok:
            # Conversion process
            self.convert_measurements(conversion_ratio)
    def convert_measurements(self, conversion_ratio):
        output_folder = os.path.join(base_path, 'output')
        # Specify the CSV files to check and convert
        csv_files = ['periderm_length.csv', 'whole_root_length.csv', 'periderm_length_after_QC.csv']

        for csv_file in csv_files:
            csv_path = os.path.join(output_folder, csv_file)
            if os.path.isfile(csv_path):
                # Perform the conversion
                converted_df = self.convert_csv_pixels_to_micrometers(csv_path, conversion_ratio)
                converted_csv_path = os.path.join(output_folder, csv_file.replace('.csv', '_micrometers.csv'))
                converted_df.to_csv(converted_csv_path, index=False)
                #QMessageBox.information(self, "Conversion Complete", f"Converted {csv_file} to micrometers.")
            else:
                QMessageBox.warning(self, "File Not Found", f"The file {csv_file} does not exist in the output folder.")

    def convert_csv_pixels_to_micrometers(self, file_path, conversion_ratio):
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Apply conversion to all elements, assuming they are numeric
        converted_df = df.applymap(lambda x: self.safe_convert_pixels_to_micrometers(x, conversion_ratio))
        return converted_df

    def safe_convert_pixels_to_micrometers(self, value, conversion_ratio):
        try:
            # Ensure that the value is a number before conversion
            return float(value) / conversion_ratio
        except (ValueError, TypeError):
            # Return the value unchanged if it's not a number
            return value       


    def display_image(self, image_path):
        # Check if the image file exists
        if os.path.exists(image_path):
            # Read the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize_image(image)
    
            # Convert to QImage and then to QPixmap to display it on QLabel
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
        else:
            # If the image does not exist, display an error message
            QMessageBox.warning(self, "Image Not Found", f"The image at {image_path} was not found.")

    
    def Visualization(self):
        #app = QApplication(sys.argv)  # Initialize a QApplication

        output_folder = os.path.join(os.path.dirname(__file__), 'output')
        periderm_file = os.path.join(output_folder, "periderm_length.csv")
        whole_root_file = os.path.join(output_folder, "whole_root_length.csv")

        # Check if periderm_length.csv exists
        if os.path.exists(periderm_file):
            # Run boxplot_periderm.py
            self.run_script(os.path.join(os.path.dirname(__file__), 'src', "boxplot_periderm.py"))

            # If both periderm_length.csv and whole_root_length.csv exist
            if os.path.exists(whole_root_file):
                # Ask the user if they want to boxplot whole root length
                response = QMessageBox.question(None, "Boxplot Whole Root Length", 
                                                "Do you want to boxplot whole root length?", 
                                                QMessageBox.Yes | QMessageBox.No)

                # If user chooses yes, also run boxplot_whole_root.py
                if response == QMessageBox.Yes:
                    self.run_script(os.path.join(os.path.dirname(__file__), 'src', "boxplot_whole_root.py"))

        #sys.exit(app.exec_())  # Properly exit the application




def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
