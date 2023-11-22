import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

from model.mat import MAT

def main():
    print(tf.__version__) 
    dataset = 'CWRU' # CWRU or motor

    if dataset=='CWRU':
        work_condition = '0HPallF'
        end = 'FE' # FE(fan end), or DE(drive end) 
        data_folder = r'e:\dataset_complet\Transformer+VMD\CWRU\CWRU_decom_com_npy' # 根据实际情况
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


    # 加载模型
    mat = tf.keras.models.load_model(os.path.join(exper_save_folder, 'model'))
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

    # 测试
    _, accuracy, top_5_accuracy = mat.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

if __name__ == "__main__":
    main()
