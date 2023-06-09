# Towards Understanding and Improving GFlowNets, ICML 2023
Code for Towards Understanding and Improving GFlowNet Training, ICML 2023.
This code is provided as-is and intended as a reference for how our GFlowNet improvement proposals were implemented, and how experiments were performed. While it can serve as a basis for a package, it is not intended for this purpose, as some of our coding choices traded off increased flexibility in GFlowNet design for experimentation, at the cost of runtime speed.

Cite as (bibtex)

```
@InProceedings{towardsunderstandinggflownets,
  title = 	 {Towards Understanding and Improving GFlowNet Training},
  author =       {Shen, Max Walt and Bengio, Emmanuel and Hajiramezanali, Ehsan and Loukas, Andreas and Cho, Kyunghyun and Biancalani, Tommaso},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  year = 	 {2023},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
}
```

### Code references
Our implementation of substructure-guided GFlowNets is in `gflownet/GFNs/model.py`.
The substructure guide, and scoring function, is implemented in `gflownet/guide.py`. 

### Large files
Large files `sehstr_gbtr_allpreds.pkl.gz` and `block_18_stop6.pkl.gz` are available for download at https://figshare.com/articles/dataset/sEH_dataset_for_GFlowNet_/22806671
DOI: 10.6084/m9.figshare.22806671
These files should be placed in `datasets/sehstr/`.