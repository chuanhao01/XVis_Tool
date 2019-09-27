# Guide 

This section will mainly focus on giving a more high level and general explaination of the code. This section will also try and give a more general explaination of the concepts behind the code(although I make no promises I can explain it well).  

Here is a more detailed general layout of the code:  

![Detailed layout of the code](content/wiki/guide/detailed_layout.png)  

In other words, the code will initialise all the variables you need and start with some default values, such as which layer is selected. As you move on, and change these variables, such as the selected layer, the activation and heatmap are computed again and shown in the drawer class.

### Other classes

As for using the other utility classes, you can refer to their specific sections below. I will however give a breif guide for each of them.

`drawer.py`:  
The drawer class. Its main use is to take numpy arrays and draw them out in the cv2 terminal. It accomadates for inputs with different sizes and resizes them dynamically to fit in their respective locations.  

`xai_heatmap.py`:  
The class that helps to generate heatmaps based on a given model, input image and layer name.

`xai_activations.py`:  
The class that helps to generate the activations of all filters in a given layer of a model when the input image is passed through.

`xvis_tool`:  
The high level class that wraps around the above three classes to create the tool. 