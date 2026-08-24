import os
import sys
import argparse
import numpy as np
import torch

# Navigate UP 3 levels: training -> src -> Age_Estimation
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import DataLoader
from src.transforms import transforms 
from src.utils.training_utils import load_model


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_simu_params(state_dict_path):
    """Extrait les paramètres à partir du nom du dossier."""
    dir_name = os.path.dirname(state_dict_path)
    dir_name = dir_name.split("/")[-1].split("\\")[-1]  # Compatible Linux/Windows
    params_list = dir_name.split("_")
    params = {}
    for param in params_list:
        id_char = 0
        for id_char in range(len(param)):
            if param[id_char].isdigit():
                break
        key = param[:id_char]
        value = param[id_char:]
        params[key] = value
    return params


def evaluate_vad(dataloader, model, label_mean, label_std, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    mse_loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in dataloader:
            print(
                f"Evaluation: |{'█'*int((total_batches + 1) / len(dataloader) * 20)}"
                f"{' '*int(20 - int((total_batches + 1) / len(dataloader) * 20))}| "
                f"{(total_batches + 1) / len(dataloader) * 100 :.2f}% "
                f"[{total_batches + 1}/{len(dataloader)}]",
                end="\r"
            )

            inputs, targets_norm = inputs.to(device), targets.to(device)
            outputs_norm = model(inputs)

            if outputs_norm.shape != targets_norm.shape:
                raise ValueError(
                    f"Formes incompatibles entre sorties {tuple(outputs_norm.shape)} "
                    f"et cibles {tuple(targets.shape)}"
                )

            outputs_raw = outputs_norm * label_std + label_mean
            targets_raw = targets_norm * label_std + label_mean

            all_predictions.append(outputs_raw)
            all_targets.append(targets_raw)

            total_batches += 1

    print(" " * 80, end="\r")

    predictions = torch.cat(all_predictions, dim=0).detach().cpu().numpy()
    targets = torch.cat(all_targets, dim=0).detach().cpu().numpy()

    # MSE et RMSE par dimension (0: V, 1: A, 2: D)
    mse_per_dim = np.mean((predictions - targets) ** 2, axis=0)
    rmse_per_dim = np.sqrt(mse_per_dim)
    avg_mse = np.mean(mse_per_dim)


    return avg_mse, mse_per_dim, rmse_per_dim

def main():
    parser = argparse.ArgumentParser(description="Évaluation VAD (MSE & RMSE séparés).")
    parser.add_argument("--model", type=str, default="resnet18", help="Nom du modèle.")
    parser.add_argument("--input-size", type=int, default=48, help="Taille des images.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    parser.add_argument("--dataset", type=str, default="fer", help="Dataset.")
    parser.add_argument("--split", type=str, default="Test", choices=["Test", "Val", "Train"], help="Split à évaluer.")
    parser.add_argument("--state-dict-path", type=str, required=True, help="Chemin du modèle (.pth).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-outputs", type=int, default=3, help="Nombre de sorties (V,A,D).")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Configuration du protocole et des statistiques des labels
    DataLoader.set_data_protocol("small_split")
    DataLoader._ensure_label_stats(args.dataset)

    label_mean = DataLoader.label_mean if not "pretrained" in args.state_dict_path else np.array([0.5, 0.5, 0.5])
    label_std = DataLoader.label_std if not "pretrained" in args.state_dict_path else np.array([0.5, 0.5, 0.5])

    label_mean = torch.tensor(label_mean, dtype=torch.float32, device=device)
    label_std = torch.tensor(label_std, dtype=torch.float32, device=device)

    state_dict_path = args.state_dict_path
    if not os.path.isabs(state_dict_path):
        state_dict_path = os.path.join(repo_root(), state_dict_path)

    simu_params = get_simu_params(state_dict_path)
    dropout_rate = float(simu_params.get("dropout", 0.5))

    model = load_model(
        args.model,
        num_channels=3,
        num_outputs=args.num_outputs,
        dropout_rate=dropout_rate
    )
    
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.to(device)

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=label_mean.tolist(), std=label_std.tolist())
    ])

    dataset = DataLoader(split=args.split, dataset=args.dataset, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    avg_loss, mse_per_dim, rmse_per_dim = evaluate_vad(test_loader, model, label_mean, label_std, device=device)

    print(f"\n=== Résultats d'évaluation sur '{args.split}' ===")
    print(f"MSE globale moyenne : {avg_loss:.4f}\n")
    
    dims = ["Valence (V)", "Arousal (A)", "Dominance (D)"]
    print(f"{'Dimension':<15} | {'MSE (Loss)':<10} | {'RMSE':<10}")
    print("-" * 42)
    for idx, dim_name in enumerate(dims):
        if idx < len(mse_per_dim):
            print(f"{dim_name:<15} | {mse_per_dim[idx]:<10.4f} | {rmse_per_dim[idx]:<10.4f}")

if __name__ == "__main__":
    main()