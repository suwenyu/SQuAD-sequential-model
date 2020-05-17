# SQuAD_Sequential_Model

### Preprocessing
Create a new folder called data
```bash
$ mkdir data
```

Download the qa data into data
file structure
```
data 
  |-- train-v1.1.json
  |-- dev-v1.1.json
```

```bash
$ python3 code/preprocess.py
```

Generate Preprocess files
```
data 
  |-- train.context
  |-- train.question
  |-- train.answer
  |
  |-- dev.context
  |-- dev.question
  |--dev.answer
```

### Download Pre-trained model under data
```
data 
  |-- glove.6B
        |-- glove.6B.100d.txt
        |-- glove.6B.200d.txt
  		...
```

### Run the RNN Model
Must use python2 and tensorflow 1.x version
Create virtual environment and install needed packages
```bash
$ python2 -m virtualenv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```

Run the program
```bash
$ python code/main.py --experiment_name=bidaf_best --dropout=0.15 --batch_size=60 --hidden_size_encoder=150 --embedding_size=100 --do_char_embed=False --add_highway_layer=True --rnet_attention=False --bidaf_attention=True --answer_pointer_RNET=False --smart_span=True --hidden_size_modeling=150 --mode=train

```

### Experiment Results
| Model         | F1         |  EM         |
| ------------- |:----------:| :---------: |
| RNN + Bidaf   | **68.42%** | **53.69%**  |
| RNN + Attn    | 47.57%     | 34.66%      |





