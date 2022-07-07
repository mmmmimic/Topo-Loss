# A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology
This is an **unofficial** implementation of the TPAMI'20 paper [A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology](https://arxiv.org/pdf/1910.01877.pdf).

In our implementation, we extend the topological loss to multiclass segmentation. The overall loss is the average topological loss over the categories of interest.    

## Dependencies
* Pytorch 1.0+
* NumPy
* SciPy
* [topologylayer](https://github.com/bruel-gabrielsson/TopologyLayer)

## Acknowledgements
This implementation is based on the [official implementation of this paper](https://github.com/JamesClough/topograd).
