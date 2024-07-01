# WARNING
Respect the hierarchy! Only pure kernels are allowed to be here. This is to make sure that when a kernel returns more than one output tensors, only a single `sycl::event` would suffice to make sure that the output tensors are ready.

# Layers vs Kernels
- A layer can use multiple kernels to do something. It returns a dictionary of `sycl::event`s.
- A kernel is a single operation that can operate on multiple inputs and multiple outputs. It just returns a single `sycl::event`.
