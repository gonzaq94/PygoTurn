# PygoTurn

To execute the code, you just need to download the pretrained model as indicated in the file README.txt in the "PygoTurn Implementation" folder and copy it outside the repository. You then launch the program with the following command line:

python3 PygoTurn/PygoTurn\ Implementation/src/demo.py -w pytorch_goturn.pth.tar -d PygoTurn/PygoTurn\ Implementation/sequences-train/book/ -s PygoTurn/PygoTurn\ Implementation/our_results/book/ -plots PygoTurn/PygoTurn\ Implementation/our_results/plots/book -method mean

demo.py: uses only one instance of PygoTurn

demoBis.py: uses two instances, that move in forward and backward directions, respectively.

demo_delta.py: uses two instances both moving in the forward direction, separated delta frames.

Two commands added:

-method: method used for combining the bounding boxes of the two instances. iou: chose the BB with max IoU (uses the ground truth). mean: computes the mean between the two BBs (no grund truth). By default we just use one of the instances.

-plots: directory where to save the numpy arrays of IoU for plotting.

Please, if you are working with Pycharm or another IDE, do not add to the repository the folders "venv" and "
.idea" inside the PygoTurn Implementation folder. You can avoid this by adding each modified file manually (git add file.py) or by adding all the files and then removing some of them (git add . ; git rm --cached PygoTurn\ Implementation/venv/ -r ; git rm --cached PygoTurn\ Implementation/.idea/ -r)
