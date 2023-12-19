import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageEnhance
import os

Image.MAX_IMAGE_PIXELS = None
script_dir = os.path.dirname(__file__)
main_dir = os.path.dirname(script_dir)

class ImageSelector:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Selector")

        # Get screen width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Set window size to a fraction of screen size
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.master.geometry(f"{window_width}x{window_height}")

        self.current_image = 0
        self.images = []
        self.selected_images = []
        self.zoom_scale = 1.0
        self.x = 0
        self.y = 0        

        # Create GUI elements with flexible sizes
        canvas_width = int(window_width * 0.85)
        canvas_height = int(window_height * 0.7)
        self.canvas = tk.Canvas(self.master, width=canvas_width, height=canvas_height)
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
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
    
        # Load the current image and get its size
        original_image = self.images[self.current_image]
        original_width, original_height = original_image.size
    
        # Calculate the resize ratio while maintaining the aspect ratio
        resize_ratio = min(canvas_width / original_width, canvas_height / original_height)
    
        # Calculate the new dimensions
        new_width = int(original_width * resize_ratio * self.zoom_scale)
        new_height = int(original_height * resize_ratio * self.zoom_scale)
    
        # Resize and display the image
        resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized_image)
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
