
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 参照vit，将modes转换为pathces
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
 
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 1, self.patch_size, 1], #--------------------------------------------
            strides=[1, 1, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims, 1])
        return patches
    

# patches in Mode Encoder
# 对子模态分块处理
class PatchesT(layers.Layer):
    def __init__(self, num_patches, patch_sizeT=100):
        super(PatchesT, self).__init__()
        self.patch_size = patch_sizeT
        self.num_patches = num_patches
 
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=tf.transpose(images, [0,2,3,1]),
            sizes=[1, self.patch_size, 1, 1], #--------------------------------------------
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )  
        patches = tf.reshape(patches,[batch_size,  -1, self.patch_size ,self.num_patches])
        patches = tf.transpose(patches, [0,3,1,2])
        
        return patches


# Mode Encoder 模态编码器
class PatchEncoderT(layers.Layer):
    def __init__(self, num_patchesT, projection_dimT=64):
        super(PatchEncoderT, self).__init__()
        self.num_patches = num_patchesT
        self.projection_dimT = projection_dimT
        #一个全连接层，其输出维度为projection_dim
        self.projection2 = layers.Dense(units=self.projection_dimT)
        #定义一个嵌入层，这是一个可学习的层
        #输入维度为num_patches，输出维度为projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dimT
        )
 
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        #batch_size = tf.shape(patch)[0]
        #patch_shape=patch.shape[2]
        #patch=tf.reshape(patch, [batch_size, -1, patch_shape])
        encoded = self.projection2(patch) + self.position_embedding(positions)#-------batch*6(f)*xxx(6p)*64
        return encoded

# Transformer Encoder  
# max_fre: 最高模态中心频率
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, max_fre, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.type_patches = max_fre
        #一个全连接层，其输出维度为projection_dim
        #self.projection = layers.Conv2D(1,[1,100],strides=1)
        self.projection = layers.Dense(units=projection_dim*4, activation=tf.nn.gelu)
        #self.projection = layers.MaxPooling2D(pool_size=(1, 2),strides=(1, 2), padding='valid')
        self.projection2 = layers.Dense(units=projection_dim)
        #定义一个嵌入层，这是一个可学习的层
        #输入维度为num_patches，输出维度为projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=self.type_patches, output_dim=projection_dim
        )
 
    def call(self, patch):
        batch_size = tf.shape(patch)[0]
        positions = patch[:,:,-1,:]
        positions=tf.reshape(positions,[batch_size, self.num_patches])
        patch=patch[:,:,:-1,:]
        patch_shape=patch.shape[2]#------------------
        patch=tf.reshape(patch, [batch_size, -1, patch_shape])#------------------
        #patch=self.projection(patch)
        #patch = layers.Dropout(0.4)(patch)#------------------------
        
        #patch_shape=patch.shape[2]
        #patch=tf.reshape(patch, [batch_size, -1, patch_shape])
        encoded = self.projection2(patch) + self.position_embedding(positions)
        return encoded
    
class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.dense1 = layers.Dense(hidden_units[0], activation=tf.nn.gelu)
        self.dense2 = layers.Dense(hidden_units[1], activation=tf.nn.gelu)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x) 
        return x
   
class TransformerEncoder(layers.Layer):
    def __init__(self, projection_dimT, num_heads, transformer_layers=4):
        super(TransformerEncoder, self).__init__()
        self.transformer_layers = transformer_layers
        self.projection_dimT = projection_dimT
        self.num_heads = num_heads

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.MLA = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.projection_dimT, dropout=0.2
        )
        self.mlp = MLP(hidden_units=[self.projection_dimT*2 ,self.projection_dimT], dropout_rate=0.2)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.layer_norm(x)
        # Create a multi-head attention layer.
        attention_output = self.MLA(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = self.layer_norm(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        x = layers.Add()([x3, x2])
        return x

# mat模型
class MAT(layers.Layer):
    def __init__(self, input_shape, num_classes, max_fre, patch_size=1201, patch_sizeT=100, projection_dim=64, 
                 num_heads=4, transformer_layers=4):
        super(MAT, self).__init__()
        self.input_shape_all = input_shape
        self.num_classes = num_classes
        self.max_fre = max_fre
        self.patch_size = patch_size
        self.patch_sizeT = patch_sizeT
        self.num_patches = input_shape[0]*input_shape[1]//patch_size
        self.num_patchesT = ((patch_size-1)//patch_sizeT)
        self.projection_dimT = projection_dim
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_head_units = [1024, 512]
        self.transformer_units = [
            projection_dim * 2,
            projection_dim,
        ]

        self.Patches = Patches(self.patch_size)
        self.PatchesT = PatchesT(num_patches=input_shape[0]*input_shape[1]//patch_size, patch_sizeT=patch_sizeT)
        self.PatchEncoderT = PatchEncoderT(self.num_patchesT, self.projection_dimT)
        self.mlp1 = MLP(hidden_units=[unit // 2 for unit in self.mlp_head_units], dropout_rate=0.4)
        self.mlp2 = MLP(hidden_units=self.mlp_head_units, dropout_rate=0.4)

    def call(self):
        inputs = layers.Input(shape=self.input_shape_all)
        # Augment data.
        augmented = inputs
        #augmented = augmented_train_batches(inputs)    
        # Create patches.
        patches = self.Patches(augmented)
        patchesT= patches[:,:,:-1,:]
        patchesT = self.PatchesT(patchesT)
        encoded_patchesT = self.PatchEncoderT(patchesT)

        for _ in range(self.transformer_layers):
            encoded_patchesT = TransformerEncoder(self.projection_dimT, self.num_heads)(encoded_patchesT)


        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patchesT)
        #representation = layers.Flatten()(representation)
        num_dense = representation.shape[2] * representation.shape[3]
        representation = tf.reshape(representation, [-1, num_dense])
        representation = layers.Flatten()(representation)
        representation = tf.reshape(representation, [-1, self.num_patches, num_dense])
        
        representation = layers.Dropout(0.4)(representation)
        # Add MLP.
        featuresT = self.mlp1(representation)
        #-------------------------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------------------------
        positionf=patches[:,:,-1,:]
        patches=tf.concat([featuresT, positionf], axis=2)
        
        
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.max_fre, self.projection_dim)(tf.expand_dims(patches, 3))
    
        for _ in range(self.transformer_layers):
            encoded_patches = TransformerEncoder(self.projection_dim, self.num_heads)(encoded_patches)
    
        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.4)(representation)
        # Add MLP.
        features = self.mlp2(representation)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model
    

if __name__ == "__main__":
    print(tf.__version__) 
    # 创建模型实例
    input_shape = (11, 1201, 1)
    num_classes = 26
    max_fre = 12000

    mat1 = MAT(input_shape, num_classes, max_fre)
    mat = mat1.call()
    mat.summary()