from model import build_model
from generator import Parallel_array_reader_thread as Gen
import tensorflow as tf


model_path = './manual/my_checkpoint'

gen_batch_size = 100
model = build_model()

with Gen(r'dataset.hdf5', gen_batch_size) as train_gen:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='mae')
    ]
    for i in range(1, 31):
        print('Batch #', i)
        x, y = next(train_gen)

        model.fit([x[:gen_batch_size], x[gen_batch_size:]], y, epochs=3,
                  batch_size=1, callbacks=callbacks)

model.save_weights(model_path)
print('Done!')
