# Feature Learning on Whitened Image Patches

The provided test image for this example is taken from the van Hateren's Natural Image Dataset [1].


## Requirements
To run this example, make sure to have completed the installation instructions [described here](../../README.md) and to have the `tvo` environment activated.

```bash
conda activate tvo
```

The example additionally requires `tifffile`, `imageio`, and `tvutil` to be installed (for details on tvutil see [here](https://github.com/tvlearn/tvutil)). 


## Get started
To start the experiment, run `python main.py`. To see possible options, run, e.g.:

```bash
usage: main.py [-h] [--image_file IMAGE_FILE] [--patch_size PATCH_SIZE PATCH_SIZE] [--no_patches NO_PATCHES] [--output_directory OUTPUT_DIRECTORY] [-H H] [--Ksize KSIZE] [--selection {fitness,uniform}] [--crossover] [--no_parents NO_PARENTS]
               [--no_children NO_CHILDREN] [--no_generations NO_GENERATIONS] [--no_epochs NO_EPOCHS] [--viz_every VIZ_EVERY]

Train EBSC on whitened image patches

optional arguments:
  -h, --help            show this help message and exit
  --image_file IMAGE_FILE
                        Full path to image file (.png, .jpg, .tiff, ...) used to extract training patches (default: ./data/image.tiff)
  --patch_size PATCH_SIZE PATCH_SIZE
                        Patch size, (height, width) tuple (default: (10, 10))
  --no_patches NO_PATCHES
                        Number of image patches to extract for training (default: 5000)
  --output_directory OUTPUT_DIRECTORY
                        Directory to write training output and visualizations to (will be output/<TIMESTAMP> if not specified) (default: None)
  -H H                  Number of generative fields to learn (default: 128)
  --Ksize KSIZE         Size of the K sets (i.e., S=|K|) (default: 10)
  --selection {fitness,uniform}
                        Selection operator (default: fitness)
  --crossover           Whether to apply crossover. Must be False if no_children is specified (default: False)
  --no_parents NO_PARENTS
                        Number of parental states to select per generation (default: 5)
  --no_children NO_CHILDREN
                        Number of children to evolve per generation (default: 3)
  --no_generations NO_GENERATIONS
                        Number of generations to evolve (default: 2)
  --no_epochs NO_EPOCHS
                        Number of epochs to train (default: 100)
  --viz_every VIZ_EVERY
                        Create visualizations every Xth epoch. Set to no_epochs if not specified. (default: 1)
```


## Distributed execution

For distributed execution on multiple CPU cores (requires MPI to be installed), run with `mpirun -n <n_proc> python main.py ...`, e.g.:

```bash
env TVO_MPI=1 mpirun -n 4 python main.py
```

To run on GPU (requires cudatoolkit to be installed), run, e.g.:

```bash
env TVO_GPU=0 python main.py
```


# Reference
[1] Independent Component Filters of Natural Images Compared with Simple Cells in Primary Visual Cortex}. J. H. van Hateren and A. van der Schaaf. _Proceedings: Biological Sciences_ 265(1394):359-366, 1998.
