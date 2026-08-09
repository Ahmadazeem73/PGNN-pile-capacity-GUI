# GUI Application for Pile Bearing Capacity Prediction using Adaptive PINN Model
import os
import re
import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageGrab

 
# 1. PATHS CONFIGURATION (Matches D:\AHmad\BO PINN)
 TIMESTAMP = "20260629_213135"
BASE_DIR = r"D:\AHmad\BO PINN"

METADATA_PATH = os.path.join(BASE_DIR, f"Adaptive_PINN_Metadata_{TIMESTAMP}.joblib")
MODEL_PATH = os.path.join(BASE_DIR, f"Adaptive_PINNs_Model_{TIMESTAMP}.pth")

 
# 2. GLOBAL VARIABLES FOR MODEL & SCALERS
model = None
scaler_X = None
scaler_y = None
features = ['γt', "σv'", 'ϕ', 'Nₛₚₜ', 'L', 'D', 'As']
model_loaded = False

 
# 3. LAZY LOADING FUNCTION (Prevents import crashes on headless systems)
 def try_load_model():
    global model, scaler_X, scaler_y, features, model_loaded
    if not (os.path.exists(METADATA_PATH) and os.path.exists(MODEL_PATH)):
        print(f"[Warning] Model files not found at: {BASE_DIR}. Running in Demo/Mock Mode.")
        model_loaded = False
        return False

    try:
        import joblib
        import torch
        import torch.nn as nn

        # Load metadata
        metadata = joblib.load(METADATA_PATH)
        features = metadata.get('features', features)

        # Unpack scalers
        scalers = metadata.get('scalers', {})
        if isinstance(scalers, dict):
            scaler_X = scalers.get('scaler_X')
            scaler_y = scalers.get('scaler_y')
        elif isinstance(scalers, list) and len(scalers) >= 2:
            scaler_X = scalers[0]
            scaler_y = scalers[1]

        # Reconstruct PyTorch model dynamically
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]

        def get_numeric_sort_key(key):
            return [int(s) for s in re.findall(r'\d+', key)]

        try:
            weight_keys = sorted(weight_keys, key=get_numeric_sort_key)
        except Exception:
            weight_keys = sorted(weight_keys)

        layers = []
        num_layers = len(weight_keys)

        for i, key in enumerate(weight_keys):
            weight_tensor = state_dict[key]
            out_features, in_features = weight_tensor.shape
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                layers.append(nn.Tanh())

        model = nn.Sequential(*layers)

        sequential_state_dict = {}
        linear_indices = [idx for idx, layer in enumerate(layers) if isinstance(layer, nn.Linear)]

        for idx, key in enumerate(weight_keys):
            bias_key = key.replace('.weight', '.bias')
            seq_idx = linear_indices[idx]
            seq_weight_key = f"{seq_idx}.weight"
            seq_bias_key = f"{seq_idx}.bias"
            sequential_state_dict[seq_weight_key] = state_dict[key]
            if bias_key in state_dict:
                sequential_state_dict[seq_bias_key] = state_dict[bias_key]

        model.load_state_dict(sequential_state_dict)
        model.eval()
        model_loaded = True
        print("[Success] Successfully loaded model and metadata.")
        return True
    except Exception as e:
        print(f"Error loading model/metadata: {e}")
        model_loaded = False
        return False

# Attempt lazy load
try_load_model()

 
# 4. GUI APPLICATION CLASS
 class PilePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PGNN pile capacity predictor")
        self.root.geometry("820x780")
        self.root.configure(bg="#EAEEF2")
        self.root.resizable(False, False)

        # Style Settings (Times New Roman, publication-quality)
        self.font_title = ("Times New Roman", 20, "bold")
        self.font_section = ("Times New Roman", 13, "bold")
        self.font_label = ("Times New Roman", 15, "bold")
        self.font_entry = ("Times New Roman", 14)
        self.font_btn = ("Times New Roman", 16, "bold")
        self.font_result = ("Times New Roman", 20, "bold")
        self.font_status = ("Times New Roman", 10, "italic")

        # Color palette — matched to reference mock-up
        self.color_bg = "#EAEEF2"
        self.color_header = "#55DD6C"        # bright header green
        self.color_predict = "#55DD6C"       # predict button green
        self.color_save = "#4CAF50"          # slightly deeper save-results green
        self.color_clear = "#F3B9AE"         # soft salmon/pink clear button
        self.color_border = "#1A1A1A"        # black form border
        self.color_text_dark = "#12233F"     # dark navy text on green
        self.color_label_text = "#12233F"    # dark navy label text
        self.color_result_bg = "#D9D9D9"     # gray outer result bar
        self.color_result_text = "#12233F"
        self.color_accent = "#C0392B"

        self.create_widgets()

    def create_widgets(self):
        # 1. Header
        header_frame = tk.Frame(self.root, bg=self.color_header, height=64,
                                 highlightbackground=self.color_border, highlightthickness=1)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="PGNN ultimate pile capacity predictor",
            fg=self.color_text_dark,
            bg=self.color_header,
            font=self.font_title
        )
        title_label.place(relx=0.5, rely=0.5, anchor="center")

        # 2. Main Content Box (Form) — white card, black border
        form_frame = tk.LabelFrame(
            self.root,
            text="  Input Design Parameters  ",
            bg="white",
            fg=self.color_label_text,
            font=self.font_section,
            bd=2,
            relief="solid",
            highlightbackground=self.color_border
        )
        form_frame.configure(labelanchor="nw")
        form_frame.pack(padx=18, pady=18, fill="both", expand=True)

        form_frame.grid_columnconfigure(0, weight=3)
        form_frame.grid_columnconfigure(1, weight=2)

        # Define Input Variables and default values (representative averages)
        self.entries = {}
        inputs_config = [
            ('γt', "Unit weight of soil (γt) [kN/m³]:", "18.0", 0),
            ("σv'", "Effective vertical stress (σv') [kPa]:", "120.0", 1),
            ('ϕ', "Friction angle (ϕ) [°]:", "30.0", 2),
            ('Nₛₚₜ', "SPT blow count (Nspt):", "25", 3),
            ('L', "Pile embedment length (L) [m]:", "20.0", 4),
            ('D', "Pile diameter (D) [m]:", "0.8", 5),
            ('As', "Shaft surface area (As) [m²]:", "50.24", 6)
        ]

        for key, label_text, default_val, row in inputs_config:
            lbl = tk.Label(form_frame, text=label_text, bg="white", fg=self.color_label_text,
                            font=self.font_label, anchor="w")
            lbl.grid(row=row, column=0, padx=(20, 10), pady=12, sticky="ew")

            ent = tk.Entry(form_frame, font=self.font_entry, justify="center",
                            bd=1, relief="solid", highlightthickness=1,
                            highlightbackground="#B0B8C1", highlightcolor=self.color_header)
            ent.insert(0, default_val)
            ent.grid(row=row, column=1, padx=(10, 20), pady=12, sticky="ew")
            self.entries[key] = ent

        # 3. Action Buttons Row
        btn_frame = tk.Frame(self.root, bg=self.color_bg)
        btn_frame.pack(padx=18, pady=(0, 10), fill="x")

        btn_predict = tk.Button(
            btn_frame,
            text="Predict ultimate capacity (Qmax)",
            command=self.predict_capacity,
            bg=self.color_predict,
            fg=self.color_text_dark,
            font=self.font_btn,
            relief="raised",
            bd=2,
            height=2,
            activebackground="#3FBE57",
            activeforeground=self.color_text_dark,
            cursor="hand2"
        )
        btn_predict.pack(side="left", fill="both", expand=True, padx=(0, 4))

        btn_clear = tk.Button(
            btn_frame,
            text="Clear all",
            command=self.clear_all,
            bg=self.color_clear,
            fg=self.color_text_dark,
            font=self.font_btn,
            relief="raised",
            bd=2,
            height=2,
            activebackground="#E9A296",
            activeforeground=self.color_text_dark,
            cursor="hand2"
        )
        btn_clear.pack(side="left", fill="both", expand=True, padx=4)

        btn_screenshot = tk.Button(
            btn_frame,
            text="Save results",
            command=self.save_screenshot,
            bg=self.color_save,
            fg=self.color_text_dark,
            font=self.font_btn,
            relief="raised",
            bd=2,
            height=2,
            activebackground="#3D9140",
            activeforeground=self.color_text_dark,
            cursor="hand2"
        )
        btn_screenshot.pack(side="left", fill="both", expand=True, padx=(4, 0))

        # 4. Result Display Panel — gray outer bar, white inner box, centered text
        result_outer_frame = tk.Frame(self.root, bg=self.color_result_bg, bd=2, relief="solid",
                                       highlightbackground=self.color_border, highlightthickness=1)
        result_outer_frame.pack(padx=18, pady=(0, 12), fill="x")

        self.result_frame = tk.Frame(result_outer_frame, bg=self.color_result_bg, padx=15, pady=18)
        self.result_frame.pack(fill="both")

        self.result_label = tk.Label(
            self.result_frame,
            text="Prediction: ... kN",
            fg=self.color_result_text,
            bg=self.color_result_bg,
            font=self.font_result,
            justify="center"
        )
        self.result_label.pack(fill="x")

        # 5. Status / File Diagnostics Box
        status_frame = tk.Frame(self.root, bg="#D5DBDB", height=28)
        status_frame.pack(fill="x", side="bottom")
        status_frame.pack_propagate(False)

        status_text = (f"Active Model: Adaptive_PINN_Model_{TIMESTAMP}.pth (Loaded)"
                        if model_loaded else
                        "⚠️ Running in Demo/Mock Mode (Model files not found locally)")
        status_label = tk.Label(
            status_frame,
            text=status_text,
            fg=self.color_label_text if model_loaded else self.color_accent,
            bg="#D5DBDB",
            font=self.font_status
        )
        status_label.pack(side="left", padx=10, pady=4)

    def predict_capacity(self):
        try:
            # Retrieve and parse inputs
            input_vals = []
            for feat_name in features:
                val_str = self.entries[feat_name].get().strip()
                if not val_str:
                    raise ValueError(f"Feature '{feat_name}' cannot be empty.")
                input_vals.append(float(val_str))

            if model_loaded:
                import numpy as np
                import torch
                input_array = np.array(input_vals).reshape(1, -1)
                X_scaled = scaler_X.transform(input_array)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                with torch.no_grad():
                    pred_scaled = model(X_tensor).cpu().numpy().reshape(-1, 1)

                pred_physical = scaler_y.inverse_transform(pred_scaled).flatten()[0]
            else:
                # Mock Mode: Proportional pile logic for display/screenshot purposes
                L = float(self.entries['L'].get())
                D = float(self.entries['D'].get())
                As = float(self.entries['As'].get())
                Nspt = float(self.entries['Nₛₚₜ'].get())

                # Approximate bearing capacity: Shaft friction + End bearing
                pred_physical = (Nspt * 2.0 * (math.pi * D * L)) + (40.0 * Nspt * As)

            # Format and display result
            if pred_physical < 0:
                self.result_label.config(text=f"Prediction: {pred_physical:.2f} kN (BC Viol.)",
                                          fg=self.color_accent)
            else:
                self.result_label.config(text=f"Prediction: {pred_physical:.2f} kN",
                                          fg=self.color_result_text)

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input values. Please enter numbers only.\nDetails: {ve}")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

    def clear_all(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(text="Prediction: ... kN", fg=self.color_result_text)

    def save_screenshot(self):
        try:
            self.root.update_idletasks()
            self.root.update()

            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            w = self.root.winfo_width()
            h = self.root.winfo_height()

            # Save screenshot to current directory
            screenshot_path = f"gui_screenshot_{TIMESTAMP}.png"
            img = ImageGrab.grab(bbox=(x, y, x+w, y+h))
            img.save(screenshot_path, dpi=(600, 600))

            messagebox.showinfo("Success", f"GUI Screenshot saved successfully at:\n{screenshot_path}")
            print(f"[Success] Saved GUI Screenshot: {screenshot_path}")
        except Exception as e:
            messagebox.showerror("Screenshot Error", f"Failed to capture screenshot:\n{e}")

 
# 5. APPLICATION INITIALIZATION 
if __name__ == "__main__":
    root = tk.Tk()
    app = PilePredictionGUI(root)
    root.mainloop()
