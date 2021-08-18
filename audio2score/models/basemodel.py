

import torch
import pytorch_lightning as pl

from audio2score.settings import TrainingParam


class BaseModel(pl.LightningModule):
    """Base Datamodule - DO NOT CREATE DATA MODULE HERE!
    
        Shared functions.
        """

    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        # Override configure_optimizers
        optimizer = torch.optim.Adam(self.parameters(), 
                            lr=self.learning_rate, 
                            betas=(0.8, 0.8), 
                            eps=1e-4, 
                            weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                            step_size=self.schedular_step_size, 
                            gamma=self.schedular_gamma)
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        # Override configure_callbacks
        if self.moniter_metric == 'valid_loss':
            checkpoint_filelname = '{epoch}-{valid_loss:.4f}'
        elif self.moniter_metric == 'valid_wer':
            checkpoint_filelname = '{epoch}-{valid_wer:.4f}'

        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=self.moniter_metric, 
                                    filename=checkpoint_filelname, 
                                    save_top_k=3)
        earlystop_callback = pl.callbacks.EarlyStopping(monitor=self.moniter_metric, 
                                    patience=20, 
                                    mode='min')
        return [checkpoint_callback, earlystop_callback]

    def training_step(self, batch, batch_index):
        # Override training_step
        return self.process_step(batch, batch_index, step='train')

    def validation_step(self, batch, batch_index):
        # Override validation_step
        return self.process_step(batch, batch_index, step='valid')

    def test_step(self, batch, batch_index):
        # Override test_step
        return self.process_step(batch, batch_index, step='test')

    def process_step(self, batch, batch_index, step):
        """Process one step, train, validation or test.

            Args:
                batch: batch data
                batch_index: batch_index
                step: 'train', 'valid' or 'test'.
            Returns:
                dict of loss and logs.
            """

        input_data, target_data = self.prepare_batch_data(batch)
        output_data = self.predict(input_data)
        loss = self.get_loss(output_data, target_data)

        logs = self.evaluate(output_data, target_data)
        
        logs['loss'] = loss.item()
        logs = dict({(f'{step}_{key}', value) for key, value in logs.items()})
        for k, v in logs.items():
            self.log(k, v, sync_dist=True)
        return {'loss': loss, 'logs': logs}
        