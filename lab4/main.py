from prep_data import prep_data_mlp, prep_data_cnn
from run import run, run_pipeline
from models.mlp import MLP_model
from models.cnn import CNN_model_simple, CNN_model_pooling
from models.full_cnn import CNN_full_model

import tensorflow as tf
print(tf.version.VERSION)

if __name__ == "__main__":
    mlp_data = prep_data_mlp()
    cnn_data = prep_data_cnn()

    print("\n\n\t MLP TEST \n")
    run(mlp_data, MLP_model)

    print("\n\n\t CNN SIMPLE TEST \n")
    run(cnn_data, CNN_model_simple)

    print("\n\n\t CNN WITH POOLING TEST \n")
    run(cnn_data, CNN_model_pooling)

    print("\n\n\t FULL CNN TEST \n")
    run(cnn_data, CNN_full_model)

    print("\n\n\t AUGMENTATION CNN TEST \n")
    run_pipeline(cnn_data, CNN_full_model)
