import os
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

# =========================================================
# 1. KONFIGURACJA - ZMIEŃ TE WARTOŚCI PRZED URUCHOMIENIEM
# =========================================================

# Wklej tutaj ścieżkę do wybranego pliku metrics.csv
CSV_PATH = "logs_qm9/TWOJ_KATALOG_EKSPERYMENTU/version_0/metrics.csv"

# Podaj nazwę, która pojawi się w tytule wykresu i nazwie pliku .png
EXP_NAME = "moj_wykres_qm9"

# Ustaw sztywne ramy dla osi Y. Dla QM9 błędy MSE są zazwyczaj między 0.3 a 2.0
Y_MIN = 0.3
Y_MAX = 2.0


# =========================================================

def generate_charts_qm9(file_path, exp_name, y_min, y_max):
    if not os.path.exists(file_path):
        print(f"Błąd: Nie znaleziono pliku pod ścieżką: '{file_path}'")
        return

    print(f"Odczytywanie danych z pliku: {file_path}")
    print(f"Ustawiona skala osi Y: {y_min} - {y_max}")
    metrics = pd.read_csv(file_path)

    # Sprawdzanie, których metryk treningowych użyć
    if 'train_loss_eval_mode' in metrics.columns:
        train_loss = metrics[['epoch', 'train_loss_eval_mode']].dropna().set_index('epoch')
        train_col_name = 'train_loss_eval_mode'
        train_label = 'Train Loss (Eval Mode)'
    else:
        train_loss = metrics[['epoch', 'train_loss']].dropna().set_index('epoch')
        train_col_name = 'train_loss'
        train_label = 'Train Loss'

    if 'val_loss' not in metrics.columns:
        print("Błąd: Nie znaleziono kolumny 'val_loss' w pliku CSV.")
        return

    val_loss = metrics[['epoch', 'val_loss']].dropna().set_index('epoch')

    # Tworzenie wykresu
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss.index, train_loss[train_col_name], label=train_label, marker='o', markersize=4)
    plt.plot(val_loss.index, val_loss['val_loss'], label='Validation Loss', marker='s', markersize=4)

    plt.title(f'Training vs Validation Loss per Epoch\n({exp_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')

    # Narzucenie sztywnej skali
    plt.ylim(y_min, y_max)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Zapisywanie do pliku
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(charts_dir, exist_ok=True)

    save_path = os.path.join(charts_dir, f"loss_curve_{exp_name}_{timestamp}.png")
    plt.savefig(save_path)
    print(f"Sukces! Wykres zapisano w: {save_path}")

    # Opcjonalne wyświetlanie na ekranie
    plt.show()


if __name__ == "__main__":
    generate_charts_qm9(CSV_PATH, EXP_NAME, Y_MIN, Y_MAX)