from model import build_model
from generator import Parallel_array_reader_thread as Gen
import matplotlib.pyplot as plt

model_path = './manual/my_checkpoint'

model = build_model()
model.load_weights(model_path)


def to_logits(x):
    return (x >= 0.5) * 1.


with Gen(r'dataset.hdf5', 1) as train_gen:
    for i in range(10):

        x, y = next(train_gen)

        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(x[0, :, :, 0].T.astype(float), cmap='binary')
        ax[0, 0].set_title('Source x')
        ax[0, 1].imshow(y[0, :, :, 0].T.astype(float), cmap='jet')
        ax[0, 1].set_title('Source y')
        ax[1, 0].imshow(x[1, :, :, 0].T.astype(float), cmap='binary')
        ax[1, 0].set_title('Target x')
        y_ = to_logits(model.predict(x[1:2]))
        ax[1, 1].imshow(y_.T.reshape(192, 192, 1), cmap='jet')
        ax[1, 1].set_title('Predict')
        plt.tight_layout()
        plt.show()
