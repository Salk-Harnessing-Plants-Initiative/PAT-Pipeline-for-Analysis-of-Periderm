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

        # Navigation frame for buttons
        self.nav_frame = tk.Frame(self.master)
        self.nav_frame.pack(side=tk.BOTTOM, pady=20)

        self.btn_previous = tk.Button(self.nav_frame, text="Previous", command=self.show_previous)
        self.btn_previous.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(self.nav_frame, text="Next", command=self.show_next)
        self.btn_next.pack(side=tk.RIGHT, padx=10)
        
        self.btn_select = tk.Button(self.master, text="Select", command=self.select_image)
        self.btn_select.pack(side=tk.BOTTOM)

        self.btn_not_select = tk.Button(self.master, text="Not-Select", command=self.not_select_image)
        self.btn_not_select.pack(side=tk.BOTTOM)

        self.lbl_selected_count = tk.Label(self.master, text="Selected: 0")
        self.lbl_selected_count.pack(side=tk.BOTTOM)

        self.lbl_image_name = tk.Label(self.master, text="")
        self.lbl_image_name.pack(side=tk.BOTTOM)

        self.master.bind("z", self.zoom_in)
        self.master.bind("x", self.zoom_out)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.canvas.focus_set()
        
        self.folder_path = filedialog.askdirectory(initialdir=os.path.join(main_dir, 'output', 'periderm_pipeline_QC'))
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
        #image = self.images[self.current_image].resize((int(w * self.zoom_scale), int(h * self.zoom_scale)), Image.ANTIALIAS)
        image = self.images[self.current_image].resize((int(w * self.zoom_scale), int(h * self.zoom_scale)),Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(self.x, self.y, image=photo, anchor=tk.NW)
        self.canvas.image = photo

        # Update the image name label
        filename = os.path.basename(os.path.normpath(self.images[self.current_image].filename))
        self.lbl_image_name.config(text="Image: " + filename)

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

    def not_select_image(self):
        if self.current_image in self.selected_images:
            self.selected_images.remove(self.current_image)
            self.remove_image_name()
            self.unhighlight_selected_image()
            self.update_selected_count()

    def highlight_selected_image(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_rectangle(0, 0, w, h, outline="red", width=5)

    def unhighlight_selected_image(self):
        # Redraw the image without the red rectangle
        self.show_image()

    def update_selected_count(self):
        self.lbl_selected_count.config(text="Selected: " + str(len(self.selected_images)))

    def save_image_name(self, filename):
        # Path to the output directory
        output_dir = os.path.join(main_dir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, "selected_image_names.txt")
        with open(output_file_path, "a") as f:
            f.write(filename + "\n")

    def remove_image_name(self):
        output_dir = os.path.join(main_dir, 'output')
        output_file_path = os.path.join(output_dir, "selected_image_names.txt")
        with open(output_file_path, "r") as f:
            lines = f.readlines()
        with open(output_file_path, "w") as f:
            for line in lines:
                if line.strip("\n") != os.path.basename(os.path.normpath(self.images[self.current_image].filename)):
                    f.write(line)


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
