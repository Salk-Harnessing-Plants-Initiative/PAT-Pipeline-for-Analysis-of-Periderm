import os
import shutil
import subprocess

# Define the base path for the scripts and output
base_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(base_path, 'output')

# Define scripts and other constants
preprocess_script = os.path.join(base_path, 'src', 'preprocess.py')
segmentation_script = os.path.join(base_path, 'src', 'segmentation.py')
postprocess_script = os.path.join(base_path, 'src', 'postprocess.py')
quality_control_script = os.path.join(base_path, 'src', 'quality_control.py')
phenotyping_script = os.path.join(base_path, 'src', 'phenotyping.py')
visualization_script = os.path.join(base_path, 'src', 'visualization.py')

# Helper function to run scripts
def run_script(script_path, message):
    print(message)
    subprocess.run(['python', script_path], check=True)

# Ask user for input
def ask_user(question):
    response = input(question + " (yes/no): ").lower()
    return response == 'yes'

# Main function to handle the PAT pipeline
def main():
    try:
        # Pre-process
        if ask_user("Do you want to pre-process the images?"):
            run_script(preprocess_script, "Running pre-processing...")

        # Segment
        if ask_user("Do you want to segment the images?"):
            run_script(segmentation_script, "Running segmentation...")

        # Post-process
        if ask_user("Do you want to post-process the images?"):
            run_script(postprocess_script, "Running post-processing...")

        # Quality Control
        if ask_user("Do you want to perform quality control?"):
            run_script(quality_control_script, "Running quality control...")

        # Phenotyping
        if ask_user("Do you want to perform phenotyping?"):
            run_script(phenotyping_script, "Running phenotyping...")

        # Visualization
        if ask_user("Do you want to visualize the results?"):
            run_script(visualization_script, "Running visualization...")

        # Exit and cleanup
        if ask_user("Do you want to exit PAT and perform cleanup?"):
            if ask_user("Would you like to delete temporary folders?"):
                folders_to_delete = [
                    "Image_Padding", "Prediction_patch", "Image_Crop",
                    "Segmentation_temp", "Segmentation_upp_v15",
                    "segmentation_upp_periderm_v04", "seg_QC", "Seg_pad", "Ori_pad",
                    "nature_accession"
                ]
                for folder in folders_to_delete:
                    folder_path = os.path.join(output_path, folder)
                    if os.path.exists(folder_path):
                        shutil.rmtree(folder_path)
                        print(f"Deleted folder: {folder}")

            if ask_user("Would you like to save and move result folders?"):
                dest_folder_path = input("Enter the destination folder path: ")
                folders_to_move = [
                    "Post_processing_v09", "Final_Periderm_Segmentation_Results",
                    "measurement", "for_QC"
                ]
                files_to_move = [
                    "periderm_length.csv", "selected_image_names.txt","periderm_length_after_QC.csv",
                    'whole_root_length.csv', 'whole_root_length_boxplot.png', 'periderm_length_boxplot.png'
                ]
                # Implement the moving of folders and files here, similar to the GUI logic
                # ...

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
