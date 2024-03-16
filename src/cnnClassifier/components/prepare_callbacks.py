import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
from cnnClassifier import logger
from pathlib import Path

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            str(self.config.tensorboard_root_log_dir),  # Convert WindowsPath to string
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    def _create_ckpt_callbacks(self):
        checkpoint_dir_str = str(self.config.checkpoint_model_filepath)  # Convert WindowsPath to string
        if checkpoint_dir_str.endswith('.h5'):
            return tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir_str,
                save_best_only=True
            )
        else:
            raise ValueError("Invalid checkpoint model filepath")

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks(),
            self._create_ckpt_callbacks()
        ]
