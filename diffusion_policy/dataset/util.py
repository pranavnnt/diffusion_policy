

def fix_small_variance_normalizer(normalizer, key='obs', variance_threshold=1e-4, fixed_scale=1.0):
    
    """Check and fix dimensions with small variance in the normalizer. """
    
    stats = normalizer[key].get_input_stats()
    input_min = stats['min']
    input_max = stats['max']

    input_range = input_max - input_min
    small_var_mask = input_range < variance_threshold
    fixed_dims = small_var_mask.nonzero(as_tuple=True)[0].tolist()

    if len(fixed_dims) > 0:
        print(f"[fix_small_variance_normalizer] Found {len(fixed_dims)} dimensions with variance < {variance_threshold}")
        print(f"  Dimensions: {fixed_dims}")
        print(f"  Ranges: {input_range[small_var_mask].tolist()}")

        # Get current scale and offset
        scale = normalizer[key].params_dict['scale'].clone()
        offset = normalizer[key].params_dict['offset'].clone()

        # Fix small variance dimensions
        for dim in fixed_dims:
            scale[dim] = fixed_scale
            offset[dim] = 0.0

        # Update normalizer params
        normalizer[key].params_dict['scale'].data = scale
        normalizer[key].params_dict['offset'].data = offset

        print(f"  Fixed scales: {scale[small_var_mask].tolist()}")
    else:
        print(f"[fix_small_variance_normalizer] No dimensions with variance < {variance_threshold}")

    return fixed_dims