import tensorflow as tf
import tensorflow.keras.backend as K

IMG_HEIGHT, IMG_WIDTH, IMG_COLORS = 192, 192, 1


def entropy_loss_tf(v):
    """Calculate entropy with keras backend"""
    return -1. * K.sum(v * K.log(v + 1e-30)) / (IMG_HEIGHT * IMG_WIDTH)


loss_tracker = tf.keras.metrics.BinaryCrossentropy(name="loss", from_logits=False)
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")


class UNet(tf.keras.Model):
    def train_step(self, data):
        """Redefine train step for direct entropy minimization:
        loss = Lseg(source) + Lent(target)"""
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (xs, xt), y = data

        with tf.GradientTape() as tape:
            ys_pred = self(xs, training=True)
            loss_s = K.binary_crossentropy(y, ys_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients_s = tape.gradient(loss_s, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients_s, trainable_vars))

        # Minimize Entropy
        with tf.GradientTape() as tape:
            yt_pred = self(xt, training=True)
            loss_t = 0.01 * entropy_loss_tf(yt_pred)

        trainable_vars = self.trainable_variables
        gradients_t = tape.gradient(loss_t, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients_t, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(y, ys_pred)
        mae_metric.update_state(y, ys_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

        @property
        def metrics(self):
            return [loss_tracker, mae_metric]


# UNet structure
# 1 => 16 => 32 => 64 => 128 => 256 => 128 => 64 => 32 => 16 => 1
#

inputS = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_COLORS))
inputT = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_COLORS))

# Define Layers

conv_1_1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
drop_1 = tf.keras.layers.Dropout(0.1)
conv_1_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
pool_1 = tf.keras.layers.MaxPooling2D((2, 2))

conv_2_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
drop_2 = tf.keras.layers.Dropout(0.2)
conv_2_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
pool_2 = tf.keras.layers.MaxPooling2D((2, 2))

conv_3_1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
drop_3 = tf.keras.layers.Dropout(0.2)
conv_3_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
pool_3 = tf.keras.layers.MaxPooling2D((2, 2))

conv_4_1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
drop_4 = tf.keras.layers.Dropout(0.2)
conv_4_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
pool_4 = tf.keras.layers.MaxPooling2D((2, 2))

conv_5_1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
drop_5 = tf.keras.layers.Dropout(0.3)
conv_5_2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

up_6_1 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
conv_6_2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
conv_6_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

up_7_1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
conv_7_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
conv_7_3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

up_8_1 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
conv_8_2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
conv_8_3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

up_9_1 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
conv_9_2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')
conv_9_3 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')

out = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

# Connect all

c1 = conv_1_1(inputS)
c1 = drop_1(c1)
c1 = conv_1_2(c1)
p1 = pool_1(c1)

c2 = conv_2_1(p1)
c2 = drop_2(c2)
c2 = conv_2_2(c2)
p2 = pool_2(c2)

c3 = conv_3_1(p2)
c3 = drop_3(c3)
c3 = conv_3_2(c3)
p3 = pool_3(c3)

c4 = conv_4_1(p3)
c4 = drop_4(c4)
c4 = conv_4_2(c4)
p4 = pool_4(c4)

c5 = conv_5_1(p4)
c5 = drop_5(c5)
c5 = conv_5_2(c5)

u6 = up_6_1(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = conv_6_2(u6)
c6 = conv_6_3(c6)

u7 = up_7_1(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = conv_7_2(u7)
c7 = conv_7_3(c7)

u8 = up_8_1(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = conv_8_2(u8)
c8 = conv_8_3(c8)

u9 = up_9_1(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = conv_9_2(u9)
c9 = conv_9_3(c9)

output = out(c9)


def build_model():
    model = UNet(inputs=[inputS, inputT], outputs=output)
    model.compile(optimizer='adam')
    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()
