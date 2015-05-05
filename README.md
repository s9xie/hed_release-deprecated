#Holistically-Nested Edge Detection


We provide the pretrained model and testing code for the edge detection framework Holistically-Nested Edge Detection (HED). Please see the Arxiv paper for technical details. The pretrained model achieved an ODS=.782 and OIS=.804 on BSDS benchmark dataset. 

We provide a ipython notebook at **examples/hed_release/HED-tutorial.ipynb** to do the testing and visualization, batch testing is available. NVIDIA K40 or equivlent GPU is recommended. Alternatively, you can use CPU to do the testing by change **net_full_conv.set_mode_gpu()** to **net_full_conv.set_mode_cpu()**.

The fusion-output, averaged output and individual outputs from 5 scales will be produced. Note that if you want to evaluate the results on BSDS benchmarking dataset, you should do the standard non-maximum suppression, which is not included in this repo. 
We used Piotr's Structured Forest matlab toolbox available here **https://github.com/pdollar/edges**.

If you encounter any issue when using our code or model, please let me know.

This code is based on Caffe. Thanks to the contributors of Caffe. Thanks @shelhamer and @longjon for providing fundamental implementations that enable fully convolutional training/testing in Caffe.

    @misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}}
    }
