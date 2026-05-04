import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import sys; sys.path.append(os.getcwd())

from fno_LDM.model import FNO
from fno_LDM.model.graph_normalize import convert2dict, merge_weights_dicts, build_normlize_weights



system = 'cy_'
device = 'cuda:0'


if system == 'cy_':
    Re_values = np.linspace(200, 500, 31)
    r_values = np.linspace(10, 25, 16)
    np.random.seed(1)
    Re_grid, r_grid = np.meshgrid(Re_values, r_values)
    Re_grid_flat = Re_grid.flatten()
    r_grid_flat = r_grid.flatten()
    total_combinations = Re_grid_flat.size
    random_indices = np.random.choice(total_combinations, 96, replace=False)
    selected_Re_values = Re_grid_flat[random_indices]
    selected_r_values = r_grid_flat[random_indices]
    selected_combinations = list(zip(selected_Re_values, selected_r_values))
    
    if not os.path.exists('zoo/cy_/fno/minmax_dict.pkl'):
        weight_dicts_list =  []
        for seed in range(1):
            for i, (Re, r) in enumerate(selected_combinations):
                print(f"Re{Re}/r{r}/seed{seed}")
                file_path = f"zoo/cy_/fno/origin/Re{Re}_r{r}/seed{seed}/epoch1000.pt"
                weights_dict = convert2dict(file_path)
                weight_dicts_list.append(weights_dict)

        merged_dict = merge_weights_dicts(weight_dicts_list)
        norm_dict = build_normlize_weights(merged_dict)

        with open(f'zoo/cy_/fno/minmax_dict.pkl', 'wb') as f:
            pickle.dump(norm_dict, f)
    
    if not os.path.exists('zoo/cy_/fno/l2_distance.pt'):
        fno_Re_r = []
        for (Re, r) in selected_combinations:
            fno = FNO(
                in_channels=2,
                out_channels=2,
                n_modes=(12, 6),
                n_layers=4,
                hidden_channels=64,
            )
            fno.load_state_dict(torch.load(f'zoo/cy_/fno/origin/Re{Re}_r{r}/seed0/epoch1000.pt', map_location=lambda storage, loc: storage))
            fno.to(device)
            fno_Re_r.append(fno)
            
        snapshot_Re = torch.zeros((len(selected_combinations), 100, 2, 128, 64), device=device)
        for i, (Re, r) in enumerate(selected_combinations):
            traj = np.load(f'data/cy_/Re_{Re}_r_{r}.npy') # 200, 2, 128, 64
            snapshot = torch.from_numpy(traj).to(device)
            xmin = snapshot.amin(dim=(0, 2, 3), keepdim=True)
            xmax = snapshot.amax(dim=(0, 2, 3), keepdim=True)
            snapshot = (snapshot - xmin) / (xmax - xmin)
            snapshot_Re[i] = snapshot[:100].reshape(100, 2, 128, 64)

        fno_output_Re = torch.zeros((len(selected_combinations), len(selected_combinations), 100, 2, 128, 64))
        for i in tqdm(range(len(selected_combinations))):
            fno = fno_Re_r[i]
            fno.eval()
            for j, (Re, r) in enumerate(selected_combinations):
                snapshot = snapshot_Re[j] # 100, 2, 128, 64
                with torch.no_grad():
                    fno_output_Re[i, j] = fno(snapshot).cpu()

        l2_distances = torch.zeros((len(selected_combinations), len(selected_combinations)))
        for i in range(len(selected_combinations)):
            for j in range(len(selected_combinations)):
                output1 = fno_output_Re[i, i]  # Output of i-th FNO on snapshots at Re_i
                output2 = fno_output_Re[j, i]  # Output of j-th FNO on snapshots at Re_i
                diff = output1 - output2
                l2_distance = torch.norm(diff, p=2, dim=(1, 2, 3)).mean().item()
                l2_distances[i, j] = l2_distance
        
        l2_distances /= l2_distances.max()
        torch.save(l2_distances, 'zoo/cy_/fno/l2_distance.pt')
        
        plt.figure(figsize=(4.85, 4))
        plt.imshow(l2_distances.numpy(), aspect='auto', extent=[selected_Re_values[0], selected_Re_values[-1], selected_Re_values[0], selected_Re_values[-1]], origin='lower')
        plt.xlabel('Re (Trained FNO)')
        plt.ylabel('Re (Trained FNO)')
        plt.title('L2 Distance Between FNO Outputs')
        plt.colorbar(label='L2 Distance')
        plt.tight_layout()
        plt.savefig('zoo/cy_/fno/l2_distance.png', dpi=300)
        
        
    # surrogate label
    Re_values = np.linspace(200, 500, 31)
    r_values = np.linspace(10, 25, 16)
    np.random.seed(1)
    Re_grid, r_grid = np.meshgrid(Re_values, r_values)
    Re_grid_flat = Re_grid.flatten()
    r_grid_flat = r_grid.flatten()
    total_combinations = Re_grid_flat.size
    random_indices = np.random.choice(total_combinations, 96, replace=False)
    selected_Re_values = Re_grid_flat[random_indices]
    selected_r_values = r_grid_flat[random_indices]
    selected_combinations = list(zip(selected_Re_values, selected_r_values))
    unselected_indices = np.setdiff1d(np.arange(total_combinations), random_indices)
    unselected_Re_values = Re_grid_flat[unselected_indices]
    unselected_r_values = r_grid_flat[unselected_indices]
    unselected_combinations = list(zip(unselected_Re_values, unselected_r_values))
    
    Re_numbers, r_numbers = [], []
    for i, (Re, r) in enumerate(selected_combinations):
        Re_numbers.append(Re)
        r_numbers.append(r)
    
    l2 = torch.load('zoo/cy_/fno/l2_distance.pt')
    l2 = l2.cpu().numpy()
    pca = PCA(n_components=1).fit(l2)
    metrics = pca.transform(l2).reshape(-1)
    print('Re:', pearsonr(Re_numbers, metrics))
    print('r:', pearsonr(r_numbers, metrics))
    
    lookback, downsample = 1, 2
    X, y = np.zeros((96, 100, lookback, 128//downsample, 64//downsample)), np.zeros((96, 100))
    for i, (Re, r) in enumerate(selected_combinations):
        snapshot = np.load(f'data/cy_/Re_{Re}_r_{r:.1f}.npy') # 200, 2, 128, 64
        for j in range(lookback):
            X[i, :, j:j+1] = snapshot[j:100+j, :1, ::downsample, ::downsample].reshape(100, 1, 128//downsample, 64//downsample)
        y[i] = metrics[i] * np.ones((100,))
    X = X.reshape(96*100, lookback, 128//downsample, 64//downsample)
    X = X.reshape(96*100, lookback*128//downsample*64//downsample)
    y = y.reshape(-1, 1)

    svr = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
    svr.fit(X, y.ravel())
    y_pred = svr.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'MSE: {mse:.4f}, R^2: {r2:.4f}')
    
    plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.scatter(y, y_pred, alpha=0.6, label='Predicted')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    plt.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.set_xlim([min_val*1.1, max_val*1.1])
    ax.set_ylim([min_val*1.1, max_val*1.1])
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.savefig('zoo/cy_/fno/svr_prediction_scatter.pdf', dpi=300, bbox_inches='tight', transparent=True)
    

    Re_numbers_train, r_numbers_train = [], []
    for i, (Re, r) in enumerate(selected_combinations):
        Re_numbers_train.append(Re)
        r_numbers_train.append(r)
    Re_numbers_test, r_numbers_test = [], []
    for i, (Re, r) in enumerate(unselected_combinations):
        Re_numbers_test.append(Re)
        r_numbers_test.append(r)
    test_num = len(Re_numbers_test)
    
    X_train = np.zeros((96, lookback, 128//downsample, 64//downsample))
    X_test = np.zeros((test_num, lookback, 128//downsample, 64//downsample))
    for i, (Re, r) in enumerate(zip(Re_numbers_test, r_numbers_test)):
        snapshot = np.load(f'data/cy_/Re_{Re}_r_{r:.1f}.npy') # 200, 2, 128, 64
        for j in range(lookback):
            X_test[i, j:j+1] = snapshot[j:1+j, :1, ::downsample, ::downsample].reshape(1, 1, 128//downsample, 64//downsample)
    X_test = X_test.reshape(test_num, lookback, 128//downsample, 64//downsample)
    X_test = X_test.reshape(test_num, lookback*128//downsample*64//downsample)
    for i, (Re, r) in enumerate(zip(Re_numbers_train, r_numbers_train)):
        snapshot = np.load(f'data/cy_/Re_{Re}_r_{r:.1f}.npy') # 200, 2, 128, 64
        for j in range(lookback):
            X_train[i, j:j+1] = snapshot[j:1+j, :1, ::downsample, ::downsample].reshape(1, 1, 128//downsample, 64//downsample)
    X_train = X_train.reshape(96, lookback, 128//downsample, 64//downsample)
    X_train = X_train.reshape(96, lookback*128//downsample*64//downsample)
    y_pred_test = svr.predict(X_test)
    y_pred_train = svr.predict(X_train)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Re_numbers_train, r_numbers_train, y_pred_train, label='Seen Env', c='red', marker='x', s=40)
    ax.scatter(Re_numbers_test, r_numbers_test, y_pred_test, label='Unseen Env', marker='o', s=10)
    ax.set_xlabel('Re')
    ax.set_ylabel('r')
    ax.set_zlabel(f'First Component ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax.legend(frameon=False, loc='upper left')
    plt.savefig('zoo/cy_/fno/svr_predictions.pdf', dpi=300, pad_inches=0.5, bbox_inches='tight', transparent=True)

    with open('zoo/cy_/fno/svr_model.pkl', 'wb') as f:
        pickle.dump(svr, f)
    label_dict = {}
    for i, (Re, r) in enumerate(selected_combinations):
        label_dict[(Re, r)] = y_pred_train[i]
    with open('zoo/cy_/fno/label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    
    for i, (Re, r) in enumerate(unselected_combinations):
        label_dict[(Re, r)] = y_pred_test[i]
    with open('zoo/cy_/fno/label_dict_all.pkl', 'wb') as f:
        pickle.dump(label_dict, f)