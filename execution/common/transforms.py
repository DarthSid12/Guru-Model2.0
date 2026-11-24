"""Shared crop orientation, foveation and optional log-polar transform."""
def make_transform(variant,orientation,device,crop_size=180):
    from training.salience_trans import OnTheFlyTransform
    if variant not in ['lp','cnn','plain'] or orientation not in ['U','I']:raise ValueError('Unknown transform variant/orientation')
    return OnTheFlyTransform('valid' if orientation=='U' else 'test',variant,device,crop_size=crop_size,output_shape=(crop_size,crop_size)).to(device)
