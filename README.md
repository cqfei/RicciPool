RicciPool 
=============
Our code is built based on StructPool(https://github.com/Nate1874/StructPool), DGCNN(https://github.com/muhanzhang/pytorch_DGCNN) and Graph UNet(https://github.com/HongyangGao/Graph-U-Nets). Thanks a lot for their code sharing! 

Usage
------------
First, use `cal_curvature.py` to compute curvature flow and save curvature values for each graph at different iterations. Then, in the graph pooling code file `pool.py`, call the precomputed curvature values offline.  All computed curvature results are saved in the `curvatures` folder, while geometry-flow-updated weights are stored in the `weights` folder.  

In pool.py, the following code aims to load offline weight adjusted by ricci flow: 
> ```python
> # Load adjusted weight via Ricci flow
> A = torch.load(f'./weights/{cmd_args.data}/graph{graph.order}_lr{self.lr}_alpha{self.alpha}_iter{self.number_iterations}.pt')
> ```
The parameters in upper code:
- `cmd_args.data`: Dataset name  
- `graph.order`: Graph ID  
- `self.lr`/`self.alpha`: Parameters for Ollivier Ricci curvature computation (default: 1.0 and 0.5)  
- `self.number_iterations`: Iteration count for geometric flow  

 After configuring relevant parameters, users can directly run `main.py`, e.g.:  
> ```python
> cmd_args.batch_size = 50
> cmd_args.data = 'PROTEINS'
> cmd_args.gm = 'DGCNN'
> cmd_args.latent_dim = [64]
> cmd_args.mode = 'gpu'
> cmd_args.gpu = 'cuda:1'
> cmd_args.number_iterations = 5
> cmd_args.sortpooling_k = 32
> device = cmd_args.gpu
> cross_fold = 10
> ```

Key Implementation Notes:
1. Curvature Calculation: Precompute curvatures offline via `cal_curvature.py` → saves to `curvatures/`
2. Weights Storage: Ricci-flow-tuned weights persist in `weights/` using parameterized filenames  
3. Graph Pooling: `pool.py` loads precomputed curvatures from `curvatures/`  
4. Runtime: Execute `main.py` with parameter configurations (dataset, GNN model, GPU settings, etc.)  
5. Hyperparameters:  
   - Curvature computation (`lr`, `alpha`)  
   - Training configs (`batch_size`, `latent_dim`, `sortpooling_k`)  
   - Resources (`gpu`, `mode`)  
   - Flow iterations (`number_iterations`)  

This workflow decouples curvature computation from graph pooling, optimizing efficiency through offline preprocessing and structured parameterization.
