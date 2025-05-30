# Data Preparation

To keep data preparation as fast as possible, this crate moves much
of what is found in ``../model_development`` to rust, so we can
take advantage of parallelism.

On my 16 core i7 with ~100 gigs of ram, the initial processing was
going to take 24+ hours.

This reduces that to minutes on the entire e-gmd dataset.

```
time ./target/release/data_preparation --input /mnt/scratch/e-gmd-v1.0.0/ --out ../output --threads=18
Records: 45537

real    4m57.934s
user    23m20.301s
sys     2m40.202s
```

