from classification.datasets import load_cxr_ehr, my_collate


def get_dataset(cfg, preprocessing_transform, train_transform, val_transform, fold=0):
    collate_fn = None
    if cfg.dataset.name == 'MIMICLab':
        train_dataset, validation_dataset, test_dataset = load_cxr_ehr(cfg=cfg)
        collate_fn = my_collate

    return train_dataset, validation_dataset, test_dataset, collate_fn
