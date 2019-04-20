This is a port to Tensorflow 2.0 of this [excellent implementation](https://github.com/MarkPKCollier/NeuralTuringMachine) of an improved version of Neural Turing Machine by Mark Collier. For more information about the original implementation and their paper, please visit the link.


## Neural Turing Machine - TF 2.0

This repository does several modifications on the original codes:
 - Rewritten the code to be able to run with Tensorflow 2.0 and Eager Execution.
 - AutoGraph is used to make execution as fast as a static graph.
 - Rewritten layers, initializers, optimizers and losses with the new Keras API.
 - Moved layer and variable initializations to the class constructor.
 - Tested with tensorflow==2.0.0-alpha0 on GPU.
 - Comparison to Tensorflow's implementation of NTM is removed, since `tensorflow.contrib` module no longer exists in Tensorflow 2.0. 

## Usage

```python
from ntm import NTMCell

cell = NTMCell(num_controller_layers, num_controller_units, num_memory_locations, memory_size,
               num_read_heads, num_write_heads, shift_range=3, output_dim=num_bits_per_output_vector,
               clip_value=clip_controller_output_to_value)

# Initialization
ntm = tf.keras.layers.RNN(
    cell=cell, return_sequences=True, return_state=False,
    stateful=False, unroll=True)

# Run
outputs = ntm(inputs)
```

## Run the sample tasks

To run the sample tasks provided by the original authors, run
```
python run_tasks.py --mann ntm --init_mode constant --use_local_impl true --experiment_name experiment_name
```

Note: at the beginning of the run, Tensorflow will try to compile python codes into a static graph, which can take up to 10 minutes.

To generate graphs from a past run, change `EXPERIMENT_NAME` in `produce_heat_maps.py` and run
```
python produce_heat_maps.py
```

The outputs will be in `head_logs/img`.