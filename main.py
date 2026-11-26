import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd


def main():
    window = tk.Tk()
    window.title("Grundfos — Plastic ROC Stars")
    window.geometry("900x600")

    selected_file = tk.StringVar(value="")
    loaded_data = None

    def browse_file():
        nonlocal loaded_data

        filepath = filedialog.askopenfilename(
            parent=window,
            title="Select Task 0 training data",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*"),
            ],
        )

        if not filepath:
            return

        try:
            dataframe = pd.read_csv(filepath)
        except (OSError, UnicodeError, pd.errors.ParserError,
                pd.errors.EmptyDataError) as error:
            messagebox.showerror(
                "Unable to load CSV",
                str(error),
                parent=window,
            )
            return

        required_columns = [
            "speed_rpm",
            "flow_m3_h",
            "power_w",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            messagebox.showerror(
                "Missing columns",
                "The CSV is missing:\n" + "\n".join(missing_columns),
                parent=window,
            )
            return

        if dataframe.empty:
            messagebox.showerror(
                "Empty dataset",
                "The CSV has column headers but no data rows.",
                parent=window,
            )
            return

        loaded_data = dataframe
        selected_file.set(filepath)

        messagebox.showinfo(
            "Training data loaded",
            f"Loaded {len(dataframe)} rows.\n"
            "All required Task 0 columns are present.",
            parent=window,
        )

    content = ttk.Frame(window, padding=20)
    content.pack(fill="both", expand=True)

    heading = ttk.Label(
        content,
        text="Grundfos Pump Analysis",
        font=("", 20),
    )
    heading.pack(anchor="w")

    description = ttk.Label(
        content,
        text="Task 0 — Predicting power from pump speed and flow",
    )
    description.pack(anchor="w", pady=(10, 20))

    file_label = ttk.Label(content, text="Training CSV:")
    file_label.pack(anchor="w")

    file_row = ttk.Frame(content)
    file_row.pack(fill="x", pady=(5, 20))

    file_path = ttk.Entry(
        file_row,
        textvariable=selected_file,
        state="readonly",
    )
    file_path.pack(side="left", fill="x", expand=True)

    browse_button = ttk.Button(
        file_row,
        text="Browse…",
        command=browse_file,
    )
    browse_button.pack(side="left", padx=(10, 0))

    close_button = ttk.Button(
        content,
        text="Close",
        command=window.destroy,
    )
    close_button.pack(anchor="w")

    window.mainloop()


if __name__ == "__main__":
    main()