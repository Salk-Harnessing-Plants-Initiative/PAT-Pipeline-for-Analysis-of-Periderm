import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageEnhance
import os

Image.MAX_IMAGE_PIXELS = None
script_dir = os.path.dirname(__file__)

# Path to the main.py directory (one level up from script_dir)
main_dir = os.path.dirname(script_dir)

class ImageSelector:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Selector")
        self.master.geometry("2000x1000")

        self.current_image = 0
        self.images = []
        self.selected_images = []
        self.zoom_scale = 1.0
        self.x = 0
        self.y = 0        

        # Create GUI elements
        self.canvas = tk.Canvas(self.master, width=1800, height=900)
        self.canvas.pack(side=tk.TOP)

        self.master.bind("<Left>", self.show_previous)
        self.master.bind("<Right>", self.show_next)
        
        self.btn_select = tk.Button(self.master, text="Select", command=self.select_image)
        self.btn_select.pack(side=tk.BOTTOM)
        
        self.lbl_selected_count = tk.Label(self.master, text="Selected: 0")
        self.lbl_selected_count.pack(side=tk.BOTTOM)

        self.master.bind("z", self.zoom_in)
        self.master.bind("x", self.zoom_out)
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.canvas.focus_set()
        
        self.folder_path = filedialog.askdirectory(initialdir= os.path.join(main_dir, 'output','periderm_pipeline_QC'))
        self.load_images()

    def load_images(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image = Image.open(os.path.join(self.folder_path, filename))
                self.images.append(image)

        if len(self.images) > 0:
            self.show_image()

    def show_image(self):
        w = 1800
        h = 900
        image = self.images[self.current_image].resize((int(w * self.zoom_scale), int(h * self.zoom_scale)),Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(self.x, self.y, image=photo, anchor=tk.NW)
        self.canvas.image = photo

    def show_previous(self, event=None):
        if self.current_image > 0:
            self.current_image -= 1
            self.show_image()

    def show_next(self, event=None):
        if self.current_image < len(self.images) - 1:
            self.current_image += 1
            self.show_image()

    def select_image(self):
        if self.current_image not in self.selected_images:
            self.selected_images.append(self.current_image)
            filename = os.path.basename(os.path.normpath(self.images[self.current_image].filename))
            self.save_image_name(filename)
            self.highlight_selected_image()
            self.update_selected_count()

    def highlight_selected_image(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_rectangle(0, 0, w, h, outline="red", width=5)

    def update_selected_count(self):
        self.lbl_selected_count.config(text="Selected: " + str(len(self.selected_images)))

    def save_image_name(self, filename):
        # Path to the output directory
        output_dir = os.path.join(main_dir, 'output')

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Path to the output file
        output_file_path = os.path.join(output_dir, "selected_image_names.txt")

        # Write the filename to the output file
        with open(output_file_path, "a") as f:
            f.write(filename + "\n")

    def on_button_press(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_move_press(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_button_release(self, event):
        pass
    
    def zoom_in(self, event=None):
        self.zoom_scale *= 1.1
        self.show_image()

    def zoom_out(self, event=None):
        self.zoom_scale /= 1.1
        self.show_image()

    def increase_brightness(self, event):
        self.update_brightness(10)

    def decrease_brightness(self, event):
        self.update_brightness(-10)

    def update_brightness(self, brightness_value):
        brightness = (brightness_value + 100) / 100
        enhancer = ImageEnhance.Brightness(self.images[self.current_image])
        enhanced_image = enhancer.enhance(brightness)
        self.images[self.current_image] = enhanced_image
        self.show_image()  

root = tk.Tk()
app = ImageSelector(root)
root.mainloop()
