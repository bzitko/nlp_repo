import torch
import numpy as np
from IPython.display import display, Markdown

def tensor2md(tensor, round=3, latex=False):
    s = len(tensor.shape)
    
    if s == 0:
        num = tensor.item()
        if isinstance(num, int):
            return str(num)
        if isinstance(num, str):
            return num
        return f"{float(num):.{round}}"

    txt = r"\begin{bmatrix} "
    if s % 2 == 0:
        m, n = tensor.shape[:2]
        rows = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(tensor2md(tensor[i, j], round=round))
            rows.append(" & ".join(row))
        txt += r" \\ ".join(rows)
    elif s % 2 == 1:
        n = tensor.shape[0]
        row = []
        for i in range(n):
            row.append(tensor2md(tensor[i], round=round))
        txt += r" & ".join(row) 
    txt += r"\end{bmatrix}"
    if latex:
        txt = "$" + txt + "$"
    return txt

def look(*args, round=3, end=" "):
    txt = []
    for arg in args:
        if isinstance(arg, (torch.Tensor, np.ndarray)):
            txt.append(tensor2md(arg, round=round, latex=True))
        else:
            txt.append(arg)
    display(Markdown(end.join(txt)))  
  