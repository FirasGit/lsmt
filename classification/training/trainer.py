import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from classification.training.lightning import LightningClassifier
from classification.training.model import get_model
import os
import hydra
from omegaconf import DictConfig, open_dict
from classification.training.dataset import get_dataset
from classification.training.transforms import get_transform
from classification.training.best_checkpoint import get_best_checkpoint
from classification.metrics import roc_auc_score
from pytorch_lightning.loggers import WandbLogger
import wandb

#pl.seed_everything(seed=379647, workers=True)


def get_checkpoint_callback(checkpoint_dir):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                       filename='{epoch:02d}_{Val_epoch_loss:.3f}_{Val_mean_auc:.3f}',
                                                       monitor='Val_mean_auc', mode='max')
    return checkpoint_callback


def get_dataloader(cfg, train_shuffle=True, fold=0):
    # Get dataset and dataloader
    preprocessing_transform, train_transform, val_transform = get_transform(cfg=cfg)
    if cfg.meta.transforms is False:
        train_transform = None
        val_transform = None
    train_dataset, validation_dataset, test_dataset, collate_fn = get_dataset(
        cfg, preprocessing_transform=preprocessing_transform, train_transform=train_transform, val_transform=val_transform,
        fold=fold)
    sampler = WeightedRandomSampler(
        train_dataset.sample_weights, len(train_dataset.sample_weights)) if cfg.meta.imbalance_handler == 'oversampling' else None
    shuffle = train_shuffle if sampler is None else False

    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=cfg.meta.batch_size, shuffle=shuffle, num_workers=cfg.meta.num_workers, drop_last=True,
        sampler=sampler)
    val_dataloader = DataLoader(
        validation_dataset, collate_fn=collate_fn, batch_size=cfg.meta.batch_size, shuffle=False, num_workers=cfg.meta.num_workers, drop_last=False)
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=cfg.meta.batch_size, shuffle=False, num_workers=cfg.meta.num_workers, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader


def handle_checkpointing(cfg):
    checkpoint_dir = os.path.join(
        cfg.meta.output_dir, cfg.meta.prefix_name)
    checkpoint_callback = get_checkpoint_callback(checkpoint_dir)
    resume_from_checkpoint = get_best_checkpoint(
        checkpoint_dir, metric='Val_mean_auc', mode='max')
    return checkpoint_callback, resume_from_checkpoint


def get_training_settings(cfg) -> dict:
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='Val_mean_auc', patience=cfg.early_stopping.patience,
                                                         mode='max', min_delta=0.001)
    model = get_model(cfg)
    loss_fnc = torch.nn.BCEWithLogitsLoss
    return {'lr_logger': lr_logger, 'early_stopping_callback': early_stopping_callback, 'model': model, 'loss_fnc': loss_fnc}


@hydra.main(config_path="../config", config_name="base_cfg")
def run(cfg: DictConfig):
    pl.seed_everything(seed=cfg.meta.seed, workers=True)
    folds = range(5) if cfg.meta.cross_validation else cfg.meta.folds
    org_prefix_name = cfg.meta.prefix_name
    for fold in folds:
        with open_dict(cfg):
            cfg.meta.prefix_name = org_prefix_name + f'_fold_{fold}'
        print(f"Running fold {fold}")
        wandb_logger = WandbLogger(
            project=cfg.logger.wandb.project, entity=cfg.logger.wandb.entity, name=cfg.meta.prefix_name)
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(
            cfg, fold=fold)
        checkpoint_callback, resume_from_checkpoint = handle_checkpointing(cfg)
        training_settings = get_training_settings(cfg)

        if cfg.meta.checkpoint_path:
            resume_from_checkpoint = cfg.meta.checkpoint_path
            # TODO: Verify that this works as expected
            trainable_classifier = LightningClassifier.load_from_checkpoint(model=training_settings['model'], cfg=cfg,
                                                                             checkpoint_path=resume_from_checkpoint, 
                                                                             strict=cfg.meta.strict_checkpoint_reloading)
            resume_from_checkpoint = resume_from_checkpoint if cfg.meta.only_test is True else None
        else:
            trainable_classifier = LightningClassifier(
                model=training_settings['model'], cfg=cfg)

		# TODO: This is a bug with resume from checkpoint (when only test is true)
        resume_from_checkpoint = None if not cfg.meta.only_test else resume_from_checkpoint
        print("RESUMING:",  resume_from_checkpoint)
        trainable_classifier = LightningClassifier(
            model=training_settings['model'], cfg=cfg)


        # Start the PyTorch Lightning training routine
        #trainable_classifier = LightningClassifier(
            #model=training_settings['model'], cfg=cfg)
        trainable_classifier.loss_fnc = training_settings['loss_fnc']
        trainable_classifier.metrics = {'AUC': roc_auc_score}
        
        callbacks = [training_settings['lr_logger'], checkpoint_callback]
        if cfg.early_stopping.use:
            callbacks.append(training_settings['early_stopping_callback'])

            

        num_samples = None if cfg.meta.num_samples == 'None' else cfg.meta.num_samples
        trainer = pl.Trainer(gpus=cfg.meta.gpus, max_epochs=cfg.epochs, precision=cfg.meta.precision,
                             callbacks=callbacks,
                             resume_from_checkpoint=resume_from_checkpoint,
                             deterministic=cfg.meta.deterministic, logger=wandb_logger,
                             limit_train_batches=num_samples,
                             limit_val_batches=num_samples,
                             limit_test_batches=num_samples,
                             )
        if cfg.meta.only_test == False:
            trainer.fit(trainable_classifier, train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)
            trainer.test(ckpt_path='best',
                            dataloaders=test_dataloader)
        else:
            trainable_classifier = LightningClassifier.load_from_checkpoint(
                checkpoint_path=resume_from_checkpoint, model=training_settings['model'], cfg=cfg, strict=cfg.meta.strict_checkpoint_reloading)
            trainable_classifier.loss_fnc = training_settings['loss_fnc']
            trainable_classifier.metrics = {'AUC': roc_auc_score}
            trainer.test(trainable_classifier,
                         dataloaders=test_dataloader)

        # Finish the run for a new wandb process to start
        wandb.finish()


if __name__ == "__main__":
    run()
