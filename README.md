## Set up environment
```
pip install -r requirements.txt
```

## Run model 
Run this command to get different student models for PPI dataset & GAT architecture

```
python main.py --mode={"full", "teacher", "mi", "att", "fit"}
```
mode:
- teacher: KD
- full:   training student use full supervision
- mi:     LSP
- fit: FitNet
- att: Attention Transfer