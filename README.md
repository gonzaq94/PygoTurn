# PygoTurn

To execute the code, you just need to download the pretrained model as indicated in the file README.txt in the "PygoTurn Implementation" folder and copy it outside the repository. You then launch the program with the following command line:

python3 PygoTurn/PygoTurn\ Implementation/src/demo.py -w pytorch_goturn.pth.tar -d PygoTurn/PygoTurn\ Implementation/sequences-train/book/ -s PygoTurn/PygoTurn\ Implementation/our_results/book/

Please, if you are working with Pycharm or another IDE, do not add to the repository the folders "venv" and "
.idea" inside the PygoTurn Implementation folder. You can avoid this by adding each modified file manually (git add file.py) or by adding all the files and then removing some of them (git add . ; git rm --cached PygoTurn\ Implementation/venv/ -r ; git rm --cached PygoTurn\ Implementation/.idea/ -r)
