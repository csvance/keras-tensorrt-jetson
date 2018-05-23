nVidia's Jetson platform is arguably the most powerful family of devices for deep learning at the edge. In order to achieve the full benefits of the platform, a framework called TensorRT drastically reduces inference time for supported network architectures and layers. However, nVidia does not currently make it easy to take your existing models from Keras/Tensorflow and deploy them on the Jetson with TensorRT. One reason for this is the python API for TensorRT only supports x86 based architectures. This leaves us with no real easy way of taking advantage of the benefits of TensorRT. However, there is a harder way that does work: To achieve maximum inference performance we can export and convert our model to .uff format, and then load it in TensorRT's C++ API.

###Caveats
- Keep in mind that many layers are not supported by TensorRT 3.0. The most obvious omission is BatchNorm, which is used in many different types of deep neural nets.
- Concatenate only works on the channel axis and if and only if the other dimensions are the same

## Training and exporting to .pb
- Train your model
- If using Jupyter, restart the kernel you trained your model in to remove training layers from the graph
- Reload the models weights
- Use an export function like the one in [this notebook][notebook] to export the graph to a .pb file

## Converting .pb to .uff
I suggest using the [chybhao666/cuda9_cudnn7_tensorrt3.0:latest Docker container][https://github.com/chybhao666/TensorRT] to access the script needed for converting a .pb export from Keras/Tensorflow to .uff format for TensorRT import. Once you download it, you can execute the following command to convert your network, replacing 'dense_1/BiasAdd' with the name of your output layer.

```
cd /usr/lib/python2.7/dist-packages/uff/bin
# List Layers and manually pick out the last one
python convert_to_uff.py tensorflow --input-file /

# Convert to .uff
python convert_to_uff.py tensorflow -o /path/to/graph.uff --input-file /path/to/graph.pb -O dense_1/BiasAdd
```

More information on the .pb export and .uff conversion is available from [nVidia][uff]

## Loading the .uff into TensorRT C++ Inference API
I have create a generic class which can load the graph from a .uff file and setup TensorRT for inference. It supports any number of inputs and outputs and is available on my [Github][cpp]

[notebook]: https://github.com/csvance/keras-tensorrt/training/training.ipynb
[cpp]: https://github.com/csvance/keras-tensorrt/inference
[uff]: https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#exporttftouff
[docker]: https://github.com/chybhao666/TensorRT
