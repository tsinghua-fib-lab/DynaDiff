import torch
from copy import deepcopy
from collections import defaultdict


def build_fno_graph_from_structure(fno):
    state_dict = fno.state_dict()
    device = fno.lifting.fcs[0].weight.device
    
    lift_node_features = []
    block_node_features = []
    proj_node_features = []
    edge_features, edge_index = [], []
    
    node_pos_ids, edge_pos_ids = [], []
    node_id_counter = 0
    layer_id = 0
        
    def connect_block(in_ids, out_ids, layer_id):
        for i, dst in enumerate(out_ids):
            for j, src in enumerate(in_ids):
                edge_features.append(torch.zeros(1))
                edge_index.append([src, dst])
                edge_pos_ids.append(layer_id)
    
    # ===== 1. lifting =====
    lifting_fcs = fno.lifting.fcs
    for i, layer in enumerate(lifting_fcs):
        weight = state_dict[f'lifting.fcs.{i}.weight'] # D_out, D_in, 1
        bias = state_dict[f'lifting.fcs.{i}.bias'] # D_out
        
        D_out = weight.shape[0]
        node_features = torch.concatenate((weight.reshape(D_out, -1), bias.reshape(D_out, -1)), dim=-1) # D_out, (D_in+1)
        
        layer_out_ids = list(range(node_id_counter, node_id_counter + D_out))
        
        lift_node_features.append(node_features)
        node_pos_ids.extend([layer_id] * D_out)
        node_id_counter += D_out

        if i > 0: # ignore input layer
            connect_block(prev_node_ids, layer_out_ids, layer_id)
        
        prev_node_ids = layer_out_ids
        layer_id += 1
    
    # ===== 2. fnoblocks =====
    n_layers = len(fno.fno_blocks.convs)
    for i in range(n_layers):
        w1 = state_dict[f'fno_blocks.convs.{i}.weights1']  # C_out, C_in, mode_h, mode_w
        w2 = state_dict[f'fno_blocks.convs.{i}.weights2']  # C_out, C_in, mode_h, mode_w
        skip = state_dict[f'fno_blocks.fno_skips.{i}.conv.weight']  # C_out, C_in, 1
        
        C_out = w1.shape[0]
        layer_out_ids = list(range(node_id_counter, node_id_counter + C_out))
        
        w1_real_imag = torch.concatenate((w1.real.flatten(1), w1.imag.flatten(1)), dim=-1) # C_out, 2*(C_in*mode_h*mode_w)
        w2_real_imag = torch.concatenate((w2.real.flatten(1), w2.imag.flatten(1)), dim=-1) # C_out, 2*(C_in*mode_h*mode_w)
        skip = skip.reshape(C_out, -1)  # C_out, C_in
        node_features = torch.concatenate((w1_real_imag, w2_real_imag, skip), dim=-1)  # C_out, 2*(2*(C_in*mode_h*mode_w))+C_in
        
        block_node_features.append(node_features)
        node_pos_ids.extend([layer_id] * C_out)
        node_id_counter += C_out
        
        connect_block(prev_node_ids, layer_out_ids, layer_id)
        
        prev_node_ids = layer_out_ids
        layer_id += 1
        
    # ===== 3. projection =====
    proj_fcs = fno.projection.fcs
    for i, layer in enumerate(proj_fcs):
        weight = state_dict[f'projection.fcs.{i}.weight'] # D_out, D_in, 1
        D_out = weight.shape[0]
        
        if fno.projection.fcs[i].bias is not None: # no bias
            bias = state_dict[f'projection.fcs.{i}.bias'] # D_out
            node_features = torch.concatenate((weight.reshape(D_out, -1), bias.reshape(D_out, -1)), dim=-1)
        else:
            node_features = weight.reshape(D_out, -1)
        
        layer_out_ids = list(range(node_id_counter, node_id_counter + D_out))
        
        proj_node_features.append(node_features)
        node_pos_ids.extend([layer_id] * D_out)
        node_id_counter += D_out

        connect_block(prev_node_ids, layer_out_ids, layer_id)
        
        prev_node_ids = layer_out_ids
        layer_id += 1
    
    return {
        'X_lift_node': lift_node_features,                             # (layer_num, N1, D1)
        'X_block_node': block_node_features,                           # (layer_num, N2, D2)
        'X_proj_node': proj_node_features,                             # (layer_num, N3, D3)
        'X_edge': torch.stack(edge_features).to(device),                          # (E, edge_dim)
        'edge_index': torch.tensor(edge_index, dtype=torch.long).to(device),      # (E, 2)
        'node_pos_ids': torch.tensor(node_pos_ids, dtype=torch.long).to(device),  # (N,)
        'edge_pos_ids': torch.tensor(edge_pos_ids, dtype=torch.long).to(device)   # (E,)
    }



def load_fno_weights_from_graph(fno, fno_graph):
    lift_node_features = fno_graph['X_lift_node']
    block_node_features = fno_graph['X_block_node']
    proj_node_features = fno_graph['X_proj_node']
    
    fno = deepcopy(fno)
    new_state = {}  # to be filled into model.load_state_dict

    # ===== 1. lifting =====
    lifting_fcs = fno.lifting.fcs
    for i, layer in enumerate(lifting_fcs):
        node_features = lift_node_features[i]  # D_out, (D_in+1)
        D_out, D_in = layer.out_channels, layer.in_channels
        
        weight = node_features[:, :D_in].reshape(D_out, D_in, 1) # D_out, D_in, 1
        bias = node_features[:, -1:].reshape(D_out) # D_out
        
        new_state[f'lifting.fcs.{i}.weight'] = weight
        new_state[f'lifting.fcs.{i}.bias'] = bias
    
    # ===== 2. fnoblocks =====
    n_layers = len(fno.fno_blocks.convs)
    for i in range(n_layers):
        node_features = block_node_features[i] # C_out, 2*(2*(C_in*mode_h*mode_w))+C_in
        C_out, C_in, mode_h, mode_w = fno.fno_blocks.convs[0].weights1.shape
        
        w1_real = node_features[:, :C_in*mode_h*mode_w].reshape( C_out, C_in, mode_h, mode_w)
        w1_imag = node_features[:, C_in*mode_h*mode_w:2*C_in*mode_h*mode_w].reshape( C_out, C_in, mode_h, mode_w)
        w2_real = node_features[:, 2*C_in*mode_h*mode_w:3*C_in*mode_h*mode_w].reshape( C_out, C_in, mode_h, mode_w)
        w2_imag = node_features[:, 3*C_in*mode_h*mode_w:4*C_in*mode_h*mode_w].reshape( C_out, C_in, mode_h, mode_w)
        skip = node_features[:, 4*C_in*mode_h*mode_w:].reshape( C_out, C_in, 1)
        
        new_state[f'fno_blocks.convs.{i}.weights1'] = torch.complex(w1_real, w1_imag)
        new_state[f'fno_blocks.convs.{i}.weights2'] = torch.complex(w2_real, w2_imag)
        new_state[f'fno_blocks.fno_skips.{i}.conv.weight'] = skip
    
    # ===== 3. projection =====
    proj_fcs = fno.projection.fcs
    for i, layer in enumerate(proj_fcs):
        node_features = proj_node_features[i]
        D_out, D_in = layer.out_channels, layer.in_channels
        
        weight = node_features[:, :D_in].reshape(D_out, D_in, 1) # D_out, D_in, 1
        bias = node_features[:, -1:].reshape(D_out) # D_out
        
        new_state[f'projection.fcs.{i}.weight'] = weight
        if fno.projection.fcs[i].bias is not None:
            new_state[f'projection.fcs.{i}.bias'] = bias
    
    # === Load parameters into fno ===
    fno.load_state_dict(new_state, strict=False)
    return fno



def batch_stack_fno_graphs(fno_list):
    graph_list = [build_fno_graph_from_structure(deepcopy(fno)) for fno in fno_list]    
    ref_keys = graph_list[0].keys()
    batched_graph = defaultdict(list)

    for key in ref_keys:
        first_item = graph_list[0][key]

        if isinstance(first_item, list):
            # Handle lists of tensors (node features)
            num_layers = len(first_item)
            for layer_idx in range(num_layers):
                try:
                    tensors_to_stack = [graph[key][layer_idx] for graph in graph_list]
                    batched_graph[key].append(torch.stack(tensors_to_stack, dim=0))
                except Exception as e:
                    print(f"Error stacking list item: key='{key}', layer_idx={layer_idx}")
                    print(f" Check shapes: {[graph[key][layer_idx].shape for graph in graph_list]}")
                    raise e
        elif isinstance(first_item, torch.Tensor):
            # Handle single tensors (X_edge, edge_index, node_pos_ids, edge_pos_ids)
            try:
                if 'X_edge' in key:
                    tensors_to_stack = [graph[key] for graph in graph_list]
                    batched_graph[key] = torch.stack(tensors_to_stack, dim=0)
                else:
                    batched_graph[key] = graph_list[0][key] # same, no batch
            except Exception as e:
                print(f"Error stacking tensor item: key='{key}'")
                print(f" Check shapes: {[graph[key].shape for graph in graph_list]}")
                raise e
        else:
            print(f"Warning: Skipping key '{key}' with unhandled type {type(first_item)}.")

    return dict(batched_graph)