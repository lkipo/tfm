# Goals 
For the presentation of this master's thesis the goals are as it follows:

## Main goal
The main goal is to achieve a graph that represents the vascular structure.
In order to do that a pipeline will be created.

The process will work something like this:
CT image -> Vessel segmentation -> Skeletonization -> skeleton reconnection -> Graph.
i also would like to test some other variants. Mainly those that imply skipping skeletonization and using some process to obtain the graph directly after the vessel segmentation.

Lets now face the diffrent sub-tasks.
### Vessel segmentation
This will be made trough a neural network. I'd like to explore current SOTA networks for this task, but it is not the main goal of this project. Besides from that, we'll also use the weights that are being used on production on our Parent project (removirt).

### Skeletization
i've done an extensive work on skeletonization before. Some methods like kimimaro provide a graph, others dont, like scipy's skeletonize or various other methods. I've also developed myself a metric to test the overall coverage of a skeleton. i dont want this to be the main focus of this project since i'm hoping to get it published on a scientific journal and it has already been my degree's thesis. 

For this purpose we should test kimimaro, basic skeletonize functions, and the centerline prediction of DeepVesselNet. you should also check for other networks online that are able to predict centerlines.

### Vascular reconnection
This is the main goal of this project.
My main focus for this has been adapting CCO's reconnection to perform a reconnection over the points that an skeleton has. I have got that to work somewhat decently for a single tree. For three trees i am still on it. we need to figure it out. the  method that i've been exploring is the usage of heuristic rules combined with an optimization process. The optimization process is not working right and it would be nice to at least get it working. I dont mind computational cost, we could test a bunch of combinations if necesary to check who the best one is acordign to murray's law and other diffrent rules that are described on whitehead's paper.

Another method worth testing would be to have a neural network do that and minimize these cost functions. This is where i'd like to have graph neural networks and semi-supervised learning.

## Data

We will be working mainly with three datasets. 
The first one is the medical segmentation decathlon's task 08, which has annotations of the liver's vessel over CT scans.

The second one is a sinthetic dataset developed under whitehead's methods. This generation method has a bunch of hyperparameters, which are worth testing and on which WE NEED to tweak to find the optimal behavour. We have developed an extension of this method to be able to generate 3 trees, however, the trees cross themselves and they do not resemble the liver's structure, where the 8 Couinaud segments are clearly diffrenciated and separated, not interlaced. CCO originally checks for this type of crossings. We should do that.

The third dataset is a set of revised skeletons that i have published on zenodo. This are made over the medical decathlon dataset. The annotation process was as it follows. CT Scan -> skeletonize -> supervission -> supervised skeleton.

## Writing and structure
The general structure (chapters) is good as it is. however, if you feel the need to change something on the second order, i'll find it OK, but lets try to avoid it when possible.

## Training and step skipping
In order to test the diffrent parts or skips, dont be afraid to use known data and not the one that the pipeline got untill that point. For the reconnection part, you can use the zenodo skeletons, for the skeletonization, you can use the data from the training labels of the medical segmentation decathlon dataset.

## bibliography
you have the consulted sources on reconnection-sources/