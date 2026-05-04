import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import os
from torchmetrics.functional import structural_similarity_index_measure as ssim
import warnings; warnings.filterwarnings('ignore')
import sys; sys.path.append(os.getcwd())
from fno_LDM.train.utils import set_cpu_num; set_cpu_num(1)

from fno_LDM.model import FNO


def test_reconstruct(reconstructed_model, cond, device='cuda:0'):
    reconstructed_model.to(device)
    
    Re, r = cond
    x_train = np.load(f'data/cy_/Re_{Re}_r_{r:.1f}.npy')
    uv = torch.tensor(x_train, dtype=torch.float32).unsqueeze(0).to(device)
    x_train = uv[:,:100]
    xmin = x_train.amin(dim=(0, 1, 3, 4), keepdim=True)
    xmax = x_train.amax(dim=(0, 1, 3, 4), keepdim=True)
    x_test = uv[:, 100-1:]
    x_test_normalized = (x_test - xmin) / (xmax - xmin)

    pred_step = 100

    def compute_nmse(predictions, targets):
        return torch.mean(((predictions - targets) ** 2)) / torch.mean((targets ** 2))

    def compute_rmse(predictions, targets):
        mse = torch.mean((predictions - targets) ** 2)
        return torch.sqrt(mse)

    def test_model(model, test_data, device):
        model.eval()
        num_trajectories, T, num_variables, x_dim, y_dim = test_data.shape

        nmse_list = []
        ssim_list = []
        rmse_list = []

        for trajectory_idx in range(num_trajectories):
            trajectory = test_data[trajectory_idx]
            start_time = 0
            targets = trajectory[start_time+1:start_time+1 + pred_step].to(device)
            predictions = torch.zeros((pred_step, num_variables, x_dim, y_dim)).to(device)

            for t in range(pred_step):
                if t == 0:
                    input_data = trajectory[start_time].unsqueeze(0).to(device)
                else:
                    input_data = predictions[t - 1].unsqueeze(0)
                predictions[t] = model(input_data).squeeze(0)

            ssim_values = ssim(predictions, targets, data_range=1.0).item()
            nmse_trajectory = compute_nmse(predictions, targets).item()
            rmse_trajectory = compute_rmse(predictions, targets).item()

            nmse_list.append(nmse_trajectory)
            ssim_list.append(ssim_values)
            rmse_list.append(rmse_trajectory)

        nmse_mean, nmse_std = np.mean(nmse_list), np.std(nmse_list)
        ssim_mean, ssim_std = np.mean(ssim_list), np.std(ssim_list)
        rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)

        return {
            "NMSE_mean": nmse_mean,
            "NMSE_std": nmse_std,
            "SSIM_mean": ssim_mean,
            "SSIM_std": ssim_std,
            "RMSE_mean": rmse_mean,
            "RMSE_std": rmse_std
        }, predictions, targets
    
    reconstructed_results, g_predictions, targets = test_model(reconstructed_model, x_test_normalized, device)
    
    try:
        original_model = FNO(
            in_channels=2,
            out_channels=2,
            n_modes=(12, 6),
            n_layers=4,
            hidden_channels=64,
        )
        original_model.load_state_dict(torch.load(f'zoo/cy_/fno/origin/Re{Re}_r{r}/seed0/epoch1000.pt', map_location=lambda storage, loc: storage))
        original_model.to(device)
        original_results, o_predictions, targets = test_model(original_model, x_test_normalized, device)
        
        print(f"Re={Re}, r={r:.2f} | Original: RMSE={original_results['RMSE_mean']:.4f}, SSIM={original_results['SSIM_mean']:.4f}, Generated: RMSE={reconstructed_results['RMSE_mean']:.4f}, SSIM={reconstructed_results['SSIM_mean']:.4f}")

        return o_predictions, g_predictions, targets
    except:
        print('Not find model zoo. Skip One-per-Env...')
        print(f"Re={Re}, r={r:.2f} | Generated: RMSE={reconstructed_results['RMSE_mean']:.4f}, SSIM={reconstructed_results['SSIM_mean']:.4f}")
        return None, g_predictions, targets