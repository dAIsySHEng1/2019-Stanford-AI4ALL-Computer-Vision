# 2019 Stanford AI4ALL Computer Vision

This repository contains the Jupyter notebook files that the 2019 Stanford AI4ALL computer vision group worked on at camp. The topic of our group reseach project was to map global poverty through analyzing patterns in Google Maps satellite images of areas that were impoverished and areas that weren't in poverty. 

Through the guidance of our two amazing project mentors, Andrew Kondrich (andrewk1) and Boxiao Pan (CaesarPan), we explored different machine learning models, such as a fully connected neural network and a convolutional neural network (CNN), to find the optimal solution. 

A large obstacle that our group encountered was an imbalanced dataset (80% 0s and 20% 1s), which affected the accuracy of our original convolutional neural network as our model was taking raw image inputs and then running them through filters. As a result, our model was not learning but minimized the cost function by guessing which label was more abundant. This kept the accuracy at either 20%, 50%, or 80%. 

We worked with our dedicated mentors to find a less biased model and ended up deciding to pass features extracted from another CNN (that of Mr. Stefano Ermon) into a fully connected neural network. 

The first hand experience of the importance of balanced data really enlightened our group and though our final model accuracy was not as high as the biased CNN model, it learned much better.


Andrew and Boxiao also posted the AI4ALL computer vision material on their github pages, which can be visited at:
