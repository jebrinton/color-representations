# color-representations


This is the base stucture of the steering vectors:

```
steering_vectors/
├── synthetic/
│   ├── gemma/
│   │   ├── red/
│   │   │   ├── 16.pt
│   │   │   └── 16.json
│   │   └── blue/
│   │       ├── 16.pt
│   │       └── 16.json
│   └── llama/
│       └── red/
│           ├── 16.pt
│           └── 16.json
└── common_crawl/
    └── gemma/
        └── red/
            ├── 16.pt
            └── 16.json
```

TODO:
- create a new dataset
- run the jobs on the last token
- copy over the tensors from those jobs
- analysis
    - cosine similarity matrices for each submodule color x color heatmap
    - hierarchical clustering (t-SNE?)
    - average cosine similarity by module (avg. magnitude of cossim)
    - PCA of layer with highest average cosine similarity
- steering
    - steer on the last token position
