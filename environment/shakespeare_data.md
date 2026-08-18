# Shakespeare play data

`/app/data/shakespeare.npz` contains the fixed runner's processed training split.

Each sample contains one dialogue line. Act and scene headers are absent. Speaker labels and stage directions are also absent. Lines are clipped to 72 ASCII bytes and shorter lines are zero-padded.

The fixed runner feeds shuffled raw batches. Submitted code receives the samples and label indexes. It also receives `num_labels`.

The source release was dated August 24, 2025. Its SHA-256 was `3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37`. Project Gutenberg marks the source public domain in the United States.
