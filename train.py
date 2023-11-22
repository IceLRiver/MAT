import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from model.mat import MAT

def train(model, x_train, y_train, x_test, y_test, save_path, 
          learning_rate=0.0001, weight_decay=0.0001, batch_size=128, num_epochs=20):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
 
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
 
    #checkpoint_filepath = r".\tmp\checkpoint"
    checkpoint_filepath =os.path.join(save_path, 'checkpoint\model_bak.hdf5')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
 
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[checkpoint_callback],
    )

    # 保存模型结构和参数
    model.save(os.path.join(save_path, 'model'))
    # # 从保存的文件中加载模型
    # loaded_model = tf.keras.models.load_model('model_path')

    # # 保存模型权重参数
    # model.save_weights('weights_path')
    # # 从保存的文件中加载模型权重参数
    # model.load_weights('weights_path')

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
 
    return history

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss =history.history['val_loss']
    
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1.1])
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([-0.1,4.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def main():
    print(tf.__version__) 
    dataset = 'CWRU' # CWRU or motor

    if dataset=='CWRU':
        work_condition = 'allHPallF'
        end = 'FE' # FE(fan end), or DE(drive end) 
        data_folder = r'e:\dataset_complet\Transformer+VMD\CWRU\CWRU_decom_com_npy' #根据实际情况
        path_x_train = os.path.join(data_folder, r'x_train '+work_condition+r' '+end+'get.npy')
        path_y_train = os.path.join(data_folder, r'y_train '+work_condition+r' '+end+'get.npy')
        path_x_train_pos = os.path.join(data_folder, r'x_train '+work_condition+' CF '+end+'get.npy')

        exper_save_folder = os.path.join('MAT\saved', dataset+'_'+work_condition+'_'+end)
        if not os.path.exists(exper_save_folder):
            # 如果不存在，则创建文件夹
            os.makedirs(exper_save_folder)
            os.makedirs(os.path.join(exper_save_folder, 'checkpoint'))
            os.makedirs(os.path.join(exper_save_folder, 'model'))
            print(f"文件夹 '{exper_save_folder}' 不存在，已创建。")

    
    input_shape = (11, 1201, 1)
    num_classes = 27
    max_fre = 12000


    # 创建模型实例
    mat = MAT(input_shape, num_classes, max_fre).call()
    # mat = mat1.call()
    mat.summary()

    # 加载数据集
    x_train = np.load(path_x_train)
    y_train = np.load(path_y_train)
    x_train_pos = np.load(path_x_train_pos)
    x_train = np.concatenate((x_train,x_train_pos), axis=1)
    print(x_train.shape, y_train.shape)
    x_train  = x_train.transpose(2,0,1)
    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
    x_train, y_train = shuffle(x_train, y_train, random_state=2) 
    print(x_train.shape,y_train.shape) 

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.3)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) # 查看数据集参数 


    # 训练
    history = train(model=mat, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, 
                    save_path=exper_save_folder)
    plot_training(history)

if __name__ == "__main__":
    main()
