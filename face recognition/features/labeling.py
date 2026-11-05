import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# --- CONFIGURATION (Kept from your original) ---
WIDERFACE_DIR = Path(__file__).parent.parent / "data" / "widerface"
WIDERFACE_IMAGES_DIR = WIDERFACE_DIR / "transformed_padding"

image_folder = WIDERFACE_IMAGES_DIR
output_csv = WIDERFACE_DIR / 'labels.csv'
excluded_folder = WIDERFACE_DIR / 'excluded'
attributes = ["Bald", "Eyeglasses", "Male", "No_Beard", "Smiling", "Wearing_Hat"]

os.makedirs(excluded_folder, exist_ok=True)


class LabelerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Attribute Labeler")

        # Load Data
        if os.path.exists(output_csv):
            self.df = pd.read_csv(output_csv)
        else:
            self.df = pd.DataFrame(columns=['image_name'] + attributes)

        labeled_files = self.df['image_name'].tolist()
        all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]
        self.to_label = [f for f in all_images if f not in labeled_files]
        self.current_idx = 0

        if not self.to_label:
            messagebox.showinfo("Done", "No images left to label!")
            root.destroy()
            return

        # --- UI LAYOUT ---
        # Left side: Image display
        self.img_label = tk.Label(root)
        self.img_label.grid(row=0, column=0, padx=20, pady=20, rowspan=len(attributes) + 2)

        # Right side: Attributes/Checkboxes
        self.check_vars = {}
        for i, attr in enumerate(attributes):
            var = tk.IntVar()
            cb = tk.Checkbutton(root, text=attr, variable=var, font=("Arial", 12))
            cb.grid(row=i, column=1, sticky="w", padx=20)
            self.check_vars[attr] = var

        # Bottom UI: Buttons
        self.status_label = tk.Label(root, text="", font=("Arial", 10))
        self.status_label.grid(row=len(attributes), column=1, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=len(attributes) + 1, column=1, pady=20)

        tk.Button(btn_frame, text="Save & Next", command=self.save_next, bg="green", fg="white", width=15).pack(
            side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Exclude Image", command=self.exclude_image, bg="orange", width=15).pack(side=tk.LEFT,
                                                                                                           padx=5)
        tk.Button(btn_frame, text="Quit", command=root.quit, bg="red", fg="white", width=10).pack(side=tk.LEFT, padx=5)

        self.load_image()

    def load_image(self):
        if self.current_idx >= len(self.to_label):
            messagebox.showinfo("Finished", "All images in this session labeled!")
            self.root.destroy()
            return

        img_name = self.to_label[self.current_idx]
        img_path = os.path.join(image_folder, img_name)

        # Open and display image
        pil_img = Image.open(img_path)
        # Resize for display if it's too big, though your crops are likely fine
        pil_img.thumbnail((400, 400))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.tk_img)

        # Update Status
        self.status_label.config(text=f"Image {self.current_idx + 1} of {len(self.to_label)}\n{img_name}")

        # Reset checkboxes
        for var in self.check_vars.values():
            var.set(0)

    def save_next(self):
        img_name = self.to_label[self.current_idx]

        # Collect data
        new_data = {'image_name': img_name}
        for attr, var in self.check_vars.items():
            new_data[attr] = var.get()

        # Update DataFrame and save CSV
        self.df = pd.concat([self.df, pd.DataFrame([new_data])], ignore_index=True)
        self.df.to_csv(output_csv, index=False)

        self.current_idx += 1
        self.load_image()

    def exclude_image(self):
        img_name = self.to_label[self.current_idx]
        src_path = os.path.join(image_folder, img_name)
        dest_path = os.path.join(excluded_folder, img_name)

        if messagebox.askyesno("Exclude", f"Move {img_name} to excluded folder?"):
            shutil.move(src_path, dest_path)
            self.current_idx += 1
            self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelerGUI(root)
    root.mainloop()