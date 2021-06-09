

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
        """Override configure_optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), 
                            lr=TrainingParam.learning_rate, 
                            betas=(0.8, 0.8), 
                            eps=1e-4, 
                            weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                            step_size=2, 
                            gamma=TrainingParam.gamma)
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        """Override configure_callbacks"""
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='valid_loss', 
                                    filename='{epoch}-{valid_loss:.2f}', 
                                    save_top_k=3)
        earlystop_callback = pl.callbacks.EarlyStopping(monitor='valid_loss', 
                                    patience=5, 
                                    mode='min')
        return [checkpoint_callback, earlystop_callback]

    def training_step(self, batch, batch_index):
        """Override training_step"""
        input_data, target_data = self.prepare_batch_data(batch)
        output_data = self.predict(input_data)
        loss = self.get_loss(output_data, target_data)
        logs = self.evaluate(output_data, target_data)

        logs['loss'] = loss.item()
        logs = dict({('train_'+key, value) for key, value in logs.items()})
        self.log_dict(logs)
        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_index):
        """Override validation_step"""
        input_data, target_data = self.prepare_batch_data(batch)
        output_data = self.predict(input_data)
        loss = self.get_loss(output_data, target_data)
        logs = self.evaluate(output_data, target_data)

        logs['loss'] = loss.item()
        logs = dict({('valid_'+key, value) for key, value in logs.items()})
        self.log_dict(logs)
        return {'valid_loss': loss, 'logs': logs}

    def test_step(self, batch, batch_index):
        """Override validation_step"""
        input_data, target_data = self.prepare_batch_data(batch)
        output_data = self.predict(input_data)
        loss = self.get_loss(output_data, target_data)
        logs = self.evaluate(output_data, target_data)

        logs['loss'] = loss.item()
        logs = dict({('test_'+key, value) for key, value in logs.items()})
        self.log_dict(logs)
        return {'test_loss': loss, 'logs': logs}

    