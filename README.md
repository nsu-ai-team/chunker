# Chunker
Chunker for russian and english (and other languages, if you have a dataset)

## Getting Started

### Prerequisites
You should have python3 installed on your machine (we recommend Anaconda3 package) and modules listed in requirements.txt. If you do not have them, run in Terminal
```
pip3 install -r requirements.txt
```
Installing and Usage
Linux / MacOS
To install this project on your local machine, you should run the following commands in Terminal:
```
cd YOUR_FOLDER
git init
git clone  https://github.com/nsu-ai/chunker
```
The project is now in YOUR_FOLDER.

To use this project, run
```
cd chunker
from chunker import Chunker
```
Datasets for russian and english are available here: https://drive.google.com/open?id=1BpWtEu1voKR314OkGIbY4MUsxi4DIk1A

Also there is the pretrained model for russian.

### Commands
*Dataset should be the file with three (or two) columns, wherу the first one is a word, the second one is its morphotags and the third one (optional) is its golden label. Columns should divided by ' ' and sentences should be divided by '\n'.*

+ **Chunker.load(path_to_crf)** - load a model from pickle
+ **Chunker.fit(path_to_train, path_to_save_model)** - train a model using a dataset from path_to_train, then save it to path_to_save_model
+ **Chunker.predict(X, path_to_data, with_y)** - predict labels for X (already prepared dataset for crf) or dataset from path_to_data. with_y can be True if there are golden labels and False if there is not.
+ **Chunker.fit_transform(path_to_data)** - train a model on a dataset from path_to_data and then predict labels for it.
### Examples of using Chunker:


## Contributing
...

## Authors

Anna Mosolova

Ivan Bondarenko

Vadim Fomin


See also the list of [contributors](https://github.com/nsu-ai/text_augmentation/contributors) who participated in this project.

## License
## Acknowledgments
