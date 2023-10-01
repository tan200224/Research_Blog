# Full Connected Layer
What does Fullay Connected Mean?
* It means all nodes in one layer are connected to the outputs of the next layer
* It takes the 3D Volumne output of the pervious layer and flattens it into a single vector that is used for input in the next layer.
* It's someitmes called the Dense Layer.

<img width="554" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/ce98781d-9c38-4804-9b8a-d34e40e45419">


# The Purpose of the Fully Connected Layer
* It compiles the data/output extracted from previous layer to form the final output
* It's an easy way of learning non-linear combination of these features


# Softmax Layer
### Why do we need softmax layer
* We ned to produce probablity outocmes for each class in Network.
* Softmasx converts the logs (The final output of the CNN) into probabilties
* It takes the exponents of every output and the normalized each output by the sum of the exponents.
* It guarantees a well behaved distribution (Ex: All sum up to 1 and no values are zero).

<img width="599" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/3b55b391-2e89-487d-850e-8bba3410764f">
