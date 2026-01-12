import numpy as np

def get_distilled_feature_dims(keys: list[str]) -> list[int]:
    """
    For each key, determine dimensionality based on suffix (after last underscore).
    Returns a list of ints, one per key.
    """
    dims = []
    for key in keys:
        suffix = key.split("_")[-1]
        if suffix == "mask" or suffix == "ratio":
            dims.append(1)  # mask keys are not counted directly
        else:
            dims.append(3)

    assert np.sum(dims) == 66, "Total feature dims should be 66, got {}".format(np.sum(dims))
    
    return dims

def get_distilled_feature_dict(keys, distilled_features):
    """
    Given a list of keys and a distilled feature array, return a dictionary mapping keys to their corresponding feature arrays.
    """
    dims = get_distilled_feature_dims(keys)
    feature_dict = {}
    index = 0
    for key, dim in zip(keys, dims):
        feature_dict[key] = distilled_features[index:index+dim]
        index += dim
    return feature_dict

def get_state_dict(keys, state_features):
    """
    Given a list of state keys and a state feature array, return a dictionary mapping keys to their corresponding state arrays.
    """
    dims = [3 for _ in keys]  # All state keys have dimension 3
    state_dict = {}
    index = 0
    for key, dim in zip(keys, dims):
        state_dict[key] = state_features[index:index+dim]
        index += dim
    return state_dict


