# Synthesizability for Virtual Screening
Scoring synthesizability of candidates for VS

### Download Sample File
Sample data and model can be found in the [link to sample](https://drive.google.com/file/d/1i0rhaFsuK7Mx3dG5cweeMLOw6nYHbedF/view?usp=sharing).

### Writing Config File
First, open the .yaml file in config folder and write appropriate arguments.

### Training Model
To train the model, use the following command:

```
  python train.py --config={directory_to_train.yaml} --save_dir={directory_to_save_train_log_and_model} --ngpu={ngpu} --batch_size={batch_size}
```

### Testing Model
To test the model, use the following command:

```
  python test.py --config={directory_to_test.yaml} --save_dir={directory_to_save_test_log}
```

### Plot the Test Result
To plot the density curve, use the following command:

```
  python plot.py --data_dir={directory_to_result_pkl} --save_dir={directory_to_save_png_file}
```
