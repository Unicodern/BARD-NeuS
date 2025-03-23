# BARD-NeuS
Neural implicit surface reconstruction method aimed at reconstructing precise surface details <br><br>
<img src="https://github.com/Unicodern/BARD-NeuS/blob/main/example/Multi-view%20image.png" width="277" height="220" />
<img src="https://github.com/Unicodern/BARD-NeuS/blob/main/example/mesh.png" width="237" height="220" />
<img src="https://github.com/Unicodern/BARD-NeuS/blob/main/example/color.png" width="237" height="220" />
---
## Training :blush:
```python
python exp_runner.py --mode train --conf <conf_file> --case <case_name>
```
According to the different contents of the dataset files, replace '<conf_file>' in the following four situations:<br>
>There are depth maps and masks in the dataset: ```wmask_wdepth.conf```<br>
>There is a depth map in the dataset, but no mask: ```womask_wdepth.conf```<br>
>There is a mask in the dataset but no depth map: ```wmask_wodepth.conf```<br>
>There are no depth maps or masks in the dataset: ```womask_wodepth.conf```<br>

'<case_name>' also needs to be replaced with the name of the scene to be reconstructed.
## Extract mesh :smirk:
```python
python exp_runner.py --mode validate_mesh --conf <conf_file> --case <case_name> --is_continue
```
