In our experiment, the evaluation pipeline is 
1. Store the edge prediction results in IPython Notebook to individual .mat files using scipy.io.savmat() function.
2. using nms_process.m to get NMS processed png files.
3. using EvalEdge.m to get the final evaluation results. 
4. To get the "late merging" results reported in the paper, run merge_res.m (simply add up nms processed files).

You still need to download Piotr's edge toolbox to make this work.
This is highly redundant, and for now we release these scripts so that the reported results can be exactly reproduced. (Numerical precision of edge map saved can affect the performance a little bit, e.g. directly save the png files before NMS)
We plan to port the NMS code and evaluation code to python very soon.
Contact s9xie(AT)eng.ucsd.edu for questions.
