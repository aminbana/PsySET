import torch
from steer_vec_utils import extraction_locations, name_to_loc_and_layer
import os

def load_steer_vectors(model_short_name, num_hidden_layers, hidden_size, concepts: list, concept_sources: list, 
                       steer_layers, steer_locs, intervention_source, normalize_steer_vector):
    
    
    assert len(concepts) == len(concept_sources), "Concepts and concept sources must have the same length"
    
    steer_layers = list(map(int, steer_layers.split(','))) if steer_layers != 'all' else list(range(num_hidden_layers))
    steer_locs = list(map(int, steer_locs.split(',')))
    
    steer_vecs = torch.zeros((len(concepts), len(steer_layers), len(steer_locs), hidden_size))
    
    for c, (concept , concept_source) in enumerate(zip(concepts, concept_sources)):
        steer_vecs_path = f'steer_vectors/{model_short_name}/{concept_source}/{concept}/{intervention_source}'
        assert os.path.exists(steer_vecs_path), f"Steer vectors path {steer_vecs_path} does not exist. Please generate the steer vectors first."
    
    
        for i, layer in enumerate(steer_layers):
            for j, loc in enumerate(steer_locs):
                steer_vecs[c, i, j] = torch.load(f'{steer_vecs_path}/layer_{layer}_loc_{loc}.pt')

    if normalize_steer_vector:
        steer_vecs = steer_vecs / torch.norm(steer_vecs, dim=-1, keepdim=True)

    return steer_vecs, steer_layers, steer_locs, torch.tensor([-1]) # for now, we only inject to the last token

def apply_wieghted_sum(steer_vector, steer_coeffs, device):
    assert len(steer_vector) == len(steer_coeffs), "Steer vector and coefficients must have the same length"
    assert len(steer_vector.shape) == 4, "Steer vector must have 4 dimensions"
    assert len(steer_coeffs.shape) == 1, "Steer coefficients must have 1 dimension"
    
    return torch.sum(steer_vector * steer_coeffs.view(-1, 1, 1, 1), dim=0).to(device)

def get_steer_fn(steer_vector, steer_layers, steer_locs, tokens_slice, hs_size, mode, renormalize):
    
    assert [steer_loc in extraction_locations.keys() for steer_loc in steer_locs], "Invalid steer locations provided"
    assert (10 not in steer_locs), "Cannot extract attention weights from this function"
        
    
    # tokens_slice = slice(None)
    if len(steer_vector.shape) == 3:
        steer_vector = steer_vector.unsqueeze(2)
    
    assert steer_vector.shape == (len(steer_layers), len(steer_locs), 1, hs_size)
    
    steer_vector = steer_vector.unsqueeze(0)  # Add batch dimension
    
    names_to_intervene = [extraction_locations[loc].replace("[LID]", str(layer)) for layer in steer_layers for loc in steer_locs]
    
    def steer_fn(input_vector, hook):
        name = hook.name
        
        seq_len = input_vector.shape[1]
        if name in names_to_intervene:
            # print(f"Steering {name} with shape {input_vector.shape} and seq_len {seq_len}")
            
            loc, layer = name_to_loc_and_layer(name)
            layer_idx = steer_layers.index(layer)
            loc_idx = steer_locs.index(loc)
            
            uA = input_vector[:, tokens_slice, :] # batch, token, hidden
            
            if renormalize:
                uA_norm = torch.norm(uA, dim=-1, keepdim=True)
            
            v = steer_vector[:, layer_idx, loc_idx, tokens_slice]#.to(input_vector.device)
            # print("v device:", v.device, "uA device:", uA.device)
            if mode == 'replace':
                uA = v
            elif mode == 'add':
                uA = uA + v
            else:
                raise ValueError("Invalid mode. Choose 'replace' or 'add'.")
            
            if renormalize:
                uA = uA / uA.norm(dim=-1, keepdim=True) * uA_norm
            
            input_vector[:, tokens_slice, :] = uA.to(input_vector.dtype)
        return input_vector
    
    return steer_fn

def get_default_steer_fn():
    return lambda input_vector, hook: None