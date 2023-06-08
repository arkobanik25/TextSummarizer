
# TextSummarizer


## Generating proto source for python 
```
cd py_src
python3 -m grpc_tools.protoc -I../protos --python_out=. --pyi_out=. --grpc_python_out=. ../protos/tokenize.proto
```

## Third Party Libraries
The includes for the onnx run time library have been included however, to build you will need the .so files. To get them, build the onnxruntime repo. Located [here] (https://github.com/Microsoft/onnxruntime). Once you build transfer over the libonnxruntime.so files

Files to move to third_party/
```
libonnxruntime.so
libonnxruntime.so.1.16.0
```

## ONNX Files
Due to the storage constraints of github, the .onnx files for the encoder and decoder cannot be stored. One way to obtain these files is to use Optimum export cli. Information can be found [here.] (https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)

To get the .onnx files from the standard bart-large-cnn model use the following command:

```
optimum-cli export onnx --model facebook/bart-large-cnn --task seq2seq-lm  --optimize O2 bart-cnn-onnx
```

Make sure the following files are in the same directory that ./TextSummarizer is being run from
```
encoder_model.onnx
decoder_model.onnx
```