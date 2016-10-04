export DATA_DIR=/shoprunner/tensorflow/data
export MODEL_PATH=/shoprunner/tensorflow/models/inception/inception-v3/model.ckpt-157585
export TRAIN_DIR=$DATA_DIR/train
export VALIDATION_DIR=$DATA_DIR/validate
export OUTPUT_DIRECTORY=$DATA_DIR/output
export LABELS_FILE=$DATA_DIR/labels_file.txt
export TRAIN_EVAL_DIR=$DATA_DIR/train_eval
export VALIDATION_EVAL_DIR=$DATA_DIR/validate_eval

bazel build inception/build_image_data
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8

bazel build inception/flowers_train
bazel-bin/inception/flowers_train --train_dir="${TRAIN_EVAL_DIR}" --data_dir="${OUTPUT_DIRECTORY}" --fine_tune=True --pretrained_model_checkpoint_path="${MODEL_PATH}"  --initial_learning_rate=0.001 --input_queue_memory_factor=2 --num_gpus=4

bazel build inception/flowers_eval
bazel-bin/inception/flowers_eval \
  --eval_dir="${VALIDATION_EVAL_DIR}" \
  --data_dir="${OUTPUT_DIRECTORY}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_EVAL_DIR}" \
  --input_queue_memory_factor=1 \
  --run_once

  -- Using the model
bazel build tensorflow/examples/label_image:label_image && \
bazel-bin/tensorflow/examples/label_image/label_image \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--output_layer=final_result \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg


# tensorflow inception serving
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md

# building output
bazel build -c opt tensorflow_serving/...

#exporting the model (this ShopRunner version has the custom number of classes)
sudo bazel-bin/tensorflow_serving/example/sr_export --checkpoint_dir=/shoprunner/tensorflow/data/train_eval/ --export_dir=inception-export

# pip install grpcio
# running the server
sudo bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=model.ckpt-145000 --model_base_path=inception-export

# running the client
sudo bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image=/shoprunner/ithinkshoe.jpg
# ShopRunner service (hacked the bazel BUILD file to have the sr_client recognized by bazel-bin)
sudo bazel-bin/tensorflow_serving/example/sr_client
that returns the classification result such as:
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 2
      }
    }
    string_val: "goldfish, Carassius auratus"
    string_val: "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 2
      }
    }
    float_val: 3.83907604218
    float_val: -0.294668972492
  }
}

