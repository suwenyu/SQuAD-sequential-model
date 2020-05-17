# SQuAD_Sequential_Model

This project attempts to conduct a word-level BiLSTM to work in tandem with the bi-direction attention flow mechanism for machine comprehension problem. Experiments conducted on the Stanford Question Answering Dataset(SQuAD) attest the effectiveness of our models and others.


### Dataset Description
Standford Question Answering Dataset(SQuAD) is the main dataset that we are going to use for this project.It is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
![img](./image/squad_ans.png "img")



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
Install required packages
```bash
pip install tensorflow
pip install numpy
```

Run the program
```bash
$ python code/main.py --experiment_name=bidaf_best --dropout=0.15 --batch_size=60 --hidden_size_encoder=150 --embedding_size=100 --do_char_embed=False --add_highway_layer=True --rnet_attention=False --bidaf_attention=True --answer_pointer_RNET=False --smart_span=True --hidden_size_modeling=150 --mode=train

```

### Experiment Results
| Model         | F1         |  EM         |
| ------------- |:----------:| :---------: |
| LSTM + Bidaf  | **68.42%** | **53.69%**  |
| LSTM + Attn   | 47.57%     | 34.66%      |
| DRQA          | 78.9%      | 69.4%       |
| BERT          | 91.8%      | 85.1%       |





