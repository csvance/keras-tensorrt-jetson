# keras-tensorrt-jetson
nVidia's Jetson platform is arguably the most powerful family of devices for deep learning at the edge. In order to achieve the full benefits of the platform, a framework called TensorRT drastically reduces inference time for supported network architectures and layers. However, nVidia does not currently make it easy to take your existing models from Keras/Tensorflow and deploy them on the Jetson with TensorRT. One reason for this is the python API for TensorRT only supports x86 based architectures. This leaves us with no real easy way of taking advantage of the benefits of TensorRT. However, there is a harder way that does work: To achieve maximum inference performance we can export and convert our model to .uff format, and then load it in TensorRT's C++ API.

## 1. Training and exporting to .pb
- Train your model
- If using Jupyter, restart the kernel you trained your model in to remove training layers from the graph
- Reload the models weights
- Use an export function like the one in [this notebook][notebook] to export the graph to a .pb file

## 2. Converting .pb to .uff
I suggest using the [chybhao666/cuda9_cudnn7_tensorrt3.0:latest Docker container][docker] to access the script needed for converting a .pb export from Keras/Tensorflow to .uff format for TensorRT import.

```
cd /usr/lib/python2.7/dist-packages/uff/bin
# List Layers and manually pick out the output layer
# For most networks it will be dense_x/BiasAdd, the last one that isn't a placeholder or activation layer
python convert_to_uff.py tensorflow --input-file /path/to/graph.pb -l

# Convert to .uff, replace dense_1/BiasAdd with the name of your output layer
python convert_to_uff.py tensorflow -o /path/to/graph.uff --input-file /path/to/graph.pb -O dense_1/BiasAdd
```

More information on the .pb export and .uff conversion is available from [nVidia][uff]

## 3. Loading the .uff into TensorRT C++ Inference API
I have create a generic class which can load the graph from a .uff file and setup TensorRT for inference. It supports any number of inputs and outputs and is available on my [Github][cpp]. It can be built with [nVidia nSight Eclipse Edition][eclipse] using a remote toolchain [(instructions here)][nsight]

### Caveats
- Keep in mind that many layers are not supported by TensorRT 3.0. The most obvious omission is BatchNorm, which is used in many different types of deep neural nets.
- Concatenate only works on the channel axis and if and only if the other dimensions are the same. If you have multiple paths for convolution, you are limited to concatenating them only when they have the same dimensions.


[notebook]: https://github.com/csvance/keras-tensorrt-jetson/blob/master/training/training.ipynb
[cpp]: https://github.com/csvance/keras-tensorrt-jetson/blob/master/inference/
[uff]: https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#exporttftouff
[docker]: https://github.com/chybhao666/TensorRT
[nsight]: https://devblogs.nvidia.com/remote-application-development-nvidia-nsight-eclipse-edition/
[eclipse]: https://developer.nvidia.com/nsight-eclipse-edition
