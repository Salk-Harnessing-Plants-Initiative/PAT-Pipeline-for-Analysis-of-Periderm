import sys
import os
import cv2
import shutil
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QProgressBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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

        self.image = cv2.imread("/home/lzhang/Desktop/FY_test/DALL_Periderm.png")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = self.resize_image(self.image)
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
    

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            reply = QMessageBox.question(self, 'Exit Confirmation',
                                         "Are you sure you want to exit PAT and delete temporary folders?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    
            if reply == QMessageBox.Yes:
                folders_to_delete = [
                    "Image_Padding", "Prediction_patch", "Image_Crop",
                    "Segmentation_temp", "Segmentation_upp_v15",
                    "segmentation_upp_periderm_v04", "seg_QC", "Seg_pad", "Ori_pad"
                ]
    
                for folder in folders_to_delete:
                    folder_path = os.path.join("/home/lzhang/Desktop/FY_test", folder)
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
                            src_folder_path = os.path.join("/home/lzhang/Desktop/FY_test", folder)
                            dest_path = os.path.join(dest_folder_path, folder)
    
                            if os.path.exists(src_folder_path):
                                shutil.move(src_folder_path, dest_path)
    
                        for file in files_to_move:
                            src_file_path = os.path.join("/home/lzhang/Desktop/FY_test", file)
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
        dest_folder = "/home/lzhang/Desktop/FY_test/nature_accession"
        
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
        path = "/home/lzhang/Desktop/FY_test/nature_accession"
        os.chdir(path)
        
        # Get a list of all .tif files
        tif_files = [f for f in os.listdir(path) if f.endswith(".tif")]
        num_files = len(tif_files)
        
        # Iterate through all files in the directory
        for index, f in enumerate(tif_files):  # Only iterate through .tif files
            # Check if file is a TIF image and convert it to PNG
            print(f"Converting {f} to PNG...")
            subprocess.run(['ffmpeg', '-i', f, '-pix_fmt', 'rgba', f'{os.path.splitext(f)[0]}.png'])
            print("Conversion complete!")
            
            # Update progress bar, assuming there is a method to set its value
            progress = ((index + 1) / num_files) * 100
            self.progress_bars[1].setValue(progress)  # Update the progress bar by index
            
            # Remove the original .tif file
            os.remove(f)
            print(f"Removed original file: {f}")


    def segment_image(self):
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
        process = subprocess.Popen(["python", "/home/lzhang/Desktop/FY_test/segmentation_patch_cor.py"], stdout=subprocess.PIPE, text=True)

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
        self.run_script("/home/lzhang/Desktop/FY_test/new_post_processing_for_measurement.py")
        self.progress_bars[3].setValue(100)

    def quality_control(self):
        self.run_script("/home/lzhang/Desktop/FY_test/generate_images_for_QC.py")
        self.progress_bars[4].setValue(50)
        self.run_script("/home/lzhang/Desktop/FY_test/PAT_QC_GUI.py")
        self.progress_bars[4].setValue(100)

    def phenotyping(self):
        self.run_script("/home/lzhang/Desktop/FY_test/measurement.py")
        self.progress_bars[5].setValue(100)

    def Visualization(self):
        self.run_script("/home/lzhang/Desktop/FY_test/boxplot.py")
        self.progress_bars[5].setValue(100)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
