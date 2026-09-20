Run on a remote cluster:
$ ./run-remote.sh ./make-dir.sh ./images info
$ ./run-remote.sh ./make-dir.sh ./images clean
$ ./run-remote.sh ./make-dir.sh ./images all
$ ./run-remote.sh env TVO_GPU=0 PARALLEL_RUNS=4 ./make-dir.sh ./images/ all

Run local:
$ ./make-dir.sh ./images info
$ ./make-dir.sh ./images clean
$ ./make-dir.sh ./images all
$ env PARALLEL_RUNS=4 ./make-dir.sh ./images/ all
$ env TVO_GPU=2 ./make-single.sh ./images/denoise.sh all
