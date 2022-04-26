import pytorch_lightning as pl

from models.superglue.attention import FavorAttention


class FavorAttentionProjectionRedrawCallback(pl.callbacks.Callback):
    def __init__(self, redraw_every_n_steps=100):
        self.redraw_every_n_steps = redraw_every_n_steps

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx, unused=0):
        if trainer.global_step % self.redraw_every_n_steps == 0:
            for module in pl_module.modules():
                if isinstance(module, FavorAttention):
                    module.resample_projection()
