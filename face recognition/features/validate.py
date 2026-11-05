import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# --- CONFIGURATION (Matches your labeling.py) ---
WIDERFACE_DIR = Path(__file__).parent.parent / "data" / "widerface"
WIDERFACE_IMAGES_DIR = WIDERFACE_DIR / "transformed_padding"

image_folder = WIDERFACE_IMAGES_DIR
labels_csv = WIDERFACE_DIR / 'labels.csv'
attributes = ["Bald", "Eyeglasses", "Male", "No_Beard", "Smiling", "Wearing_Hat"]


class ValidatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Validation & Correction Tool")

        # Load existing labels
        if not os.path.exists(labels_csv):
            messagebox.showerror("Error", "No labels.csv found! Label some images first.")
            root.destroy()
            return

        self.df = pd.read_csv(labels_csv)
        self.current_idx = 0

        if self.df.empty:
            messagebox.showinfo("Empty", "The labels.csv file is empty.")
            root.destroy()
            return

        # --- UI LAYOUT ---
        # Image Display
        self.img_label = tk.Label(root)
        self.img_label.grid(row=0, column=0, padx=20, pady=20, rowspan=len(attributes) + 2)

        # Checkboxes
        self.check_vars = {}
        for i, attr in enumerate(attributes):
            var = tk.IntVar()
            cb = tk.Checkbutton(root, text=attr, variable=var, font=("Arial", 12))
            cb.grid(row=i, column=1, sticky="w", padx=20)
            self.check_vars[attr] = var

        # Navigation & Status
        self.status_label = tk.Label(root, text="", font=("Arial", 10, "bold"))
        self.status_label.grid(row=len(attributes), column=1, pady=10)

        nav_frame = tk.Frame(root)
        nav_frame.grid(row=len(attributes) + 1, column=0, columnspan=2, pady=20)

        tk.Button(nav_frame, text="<< Previous", command=self.prev_entry, width=15).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Save Changes", command=self.save_current, bg="blue", fg="white", width=15).pack(
            side=tk.LEFT, padx=10)
        tk.Button(nav_frame, text="Next >>", command=self.next_entry, width=15).pack(side=tk.LEFT, padx=10)

        self.load_entry()

    def load_entry(self):
        """Loads the image and sets checkboxes based on CSV values."""
        row = self.df.iloc[self.current_idx]
        img_name = row['image_name']
        img_path = os.path.join(image_folder, img_name)

        if not os.path.exists(img_path):
            self.status_label.config(text=f"FILE MISSING: {img_name}", fg="red")
            return

        # Display Image
        pil_img = Image.open(img_path)
        pil_img.thumbnail((400, 400))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.img_label.config(image=self.tk_img)

        # Set Checkboxes from DataFrame
        for attr in attributes:
            val = int(row[attr])
            self.check_vars[attr].set(val)

        self.status_label.config(text=f"Entry {self.current_idx + 1} of {len(self.df)}\n{img_name}", fg="black")

    def save_current(self):
        """Updates the DataFrame in memory and saves to CSV."""
        for attr, var in self.check_vars.items():
            self.df.at[self.current_idx, attr] = var.get()

        self.df.to_csv(labels_csv, index=False)
        self.root.title(f"Saved! - {self.df.iloc[self.current_idx]['image_name']}")

    def next_entry(self):
        self.save_current()  # Auto-save on navigation
        if self.current_idx < len(self.df) - 1:
            self.current_idx += 1
            self.load_entry()
        else:
            messagebox.showinfo("End", "You have reached the last labeled image.")

    def prev_entry(self):
        self.save_current()  # Auto-save on navigation
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_entry()


if __name__ == "__main__":
    root = tk.Tk()
    app = ValidatorGUI(root)
    root.mainloop()