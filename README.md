# BotCGP

## Quick Start
0. Prepare the environment

    ```
    conda create -n botcp python=3.7
    conda activate botcp
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
    pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
    pip install torch-sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
    pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html
    pip install torch-geometric==2.1.0
    <!-- pip install -r requirements.txt -->
    <!-- conda install --yes --file requirements.txt -->
    ```
1. Download Datasets
    MGTAB:[link](https://drive.usercontent.google.com/download?id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe&export=download&authuser=0);  Cresci-15:[link](https://drive.usercontent.google.com/download?id=1AzMUNt70we5G2DShS8hk5qH95VR9HfD3&export=download&authuser=0); Twibot-20:[link](https://github.com/BunsenFeng/TwiBot-20)

2. Process Dataset
    ```
    <!-- Process MGTAB -->
    python edge_process.py
    ```

3. Run

    ```
    <!-- MGTAB -->
    nohup python -u con_stru_model.py --con_K 64 --mo_K 32 --community_num 32  1>Result/community_num64_model.txt 2>Error/community_num64_model.txt &
    <!-- Cresci-15 -->
    nohup python -u con_stru_model.py --con_K 32 --mo_K 32 --community_num 32  1>Result/community_num64_model.txt 2>Error/community_num64_model.txt &
    <!-- Twibot-20 -->
    nohup python -u con_stru_model.py --con_K 64 --mo_K 64 --community_num 32  1>Result/community_num64_model.txt 2>Error/community_num64_model.txt &
    ```

4. Note

    Experimental results may vary slightly due to different hardware configurations.