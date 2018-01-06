My take on [a good tutorial] (https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-09-06-another-keras-tutorial-for-neural-network-beginners.ipynb) explaining the basics of neural networks and keras in python.
I added a surrounding docker container.

Commands to build and run everything:
```
docker build -t mltest .
docker run mltest
```
For development just map the current folder to the app-folder:
```
docker run -v "/folder/with/code":/app mltest
```