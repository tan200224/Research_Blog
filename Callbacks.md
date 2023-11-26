# Callbacks
* Callbacks are used to perform different actions at different stages during training.
* Why is this useful?
  * What if we wanted to save each model after each epoch?
  * What if we set out model to train for 100 epochs but after 30, it started to overfit? Wouldn't we want some way to stop training?
  * Why if we want to log our information somewhere some that is can analyze later on?

### Callbacks are the solution
Callbacks can be used to perform:
  * Early stopping
    * During training process our validation loss may stagnate or stop decreasing and sometimes actually start to increase (overfitting)
    * We can use Callbacks to implement Early stopping, so we can stop the training when the model is overfitted and not waste time
  * Model checkpointing
    * During training we can periodically save the weights after each epoch.
    * This allows us to resume training in the event of crash
    * The checkpoint file contains the model's weights or the model itself
  * Learning Rate Scheduler
    * we can avoid having our loss oscillate around the global minimum by attempting to reduce the learn rate by a specific amount
    * If no improvement is seen in our monitored metric, we wait a certain number of epochs then this callback reduces the learning rate by a factor 
  * Logging
    * We can automatically log our model's training stats and view them later using TensorBoard or others
  * Remote Monitoring
  * Custom Functions
  



