## Variational AutoEncoder (VAE)

Variational AutoEncoder is an autoencoder and a generative model with other architecture

The encoder takes input and turns it into some vector, and the decoder takes a vector and expands it to be the input sample. In autoencoder, we dont really care about what the output is. But the latent vector

Sampling 
<img width="688" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/fd6dd3da-7cd5-457a-9429-9aefadffa474">

with autoencoder, we are not able to generate an image because we are just randomly picking a vector from the distribution pool of the input

However, the Variational encoder helps us determine where to pick the useful vectors from the input distribution. 

<img width="410" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/09c9ce16-3344-47ca-8767-4a0cf25717da">

<img width="784" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/854505c9-d6df-43d1-88de-045bc8d0d3b1">

<img width="868" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/15a051f7-28e8-4605-bf37-5796b7613974">

<img width="844" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/1a7fd5bf-0d7c-47cf-a299-80e066a7a56d">

<img width="859" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/ea5b0e9b-68e8-4c98-a31d-61a019e2b081">
