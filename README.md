# PyTorch GOTURN tracker

## Demo

Three demo codes.

demo.py: uses only one instance of PygoTurn

demoBis.py: uses two instances, that move in forward and backward directions, respectively.

demo_delta.py: uses two instances both moving in the forward direction, separated delta frames.

### [Download pretrained model](https://drive.google.com/file/d/1szpx3J-hfSrBEi_bze3d0PjSfQwNij7X/view?usp=sharing)

Navigate to `pygoturn/src` and do:

```
python3 demo.py -w /path/to/pretrained/model
```
Arguments:

`-w / --model-weights`: Path to a PyTorch pretrained model checkpoint.   

`-d / --data-directory`: Path to a tracking sequence which follows [OTB format](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html).   

`-s / --save-directory`: Directory to save sequence images with predicted bounding boxes.

`-method `: method used for combining the bounding boxes of the two instances. iou: chose the BB with max IoU (uses the ground truth). mean: computes the mean between the two BBs (no grund truth). By default we just use one of the instances.

`-plots `: directory where to save the numpy arrays of IoU for plotting.

Example:

`python3 PygoTurn/PygoTurn\ Implementation/src/demo_delta.py -w pytorch_goturn.pth.tar -d PygoTurn/PygoTurn\ Implementation/sequences-train/book/ -s PygoTurn/PygoTurn\ Implementation/our_results/book/ -plots PygoTurn/PygoTurn\ Implementation/our_results/plots/book -method mean`
