
# Model Card

Inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf), we’re providing some accompanying information about the multimodal model.   

## Model Details

Text Classification using LSTM

### Model Date

May 2020

### Model Type

A sequence model is used to avoid problems with words predominantly featured in both Positive and Negative tweets. The model architecture is as follors. First a tokenisation is done to all the dataset. The first layer is an Embedding Layer that generates an embedding vector for each input sequence. Then, a Conv1D Layer used to convolve data into smaller feature vectors. Next, a Long Short Term Memory and a Fully Connected Layer for classification. Finally there is a sigmoid to get a probability score as an output.

### Model Versions

[More Information Needed]

### Documents

- [Kaggle Notebook](https://www.kaggle.com/code/arunrk7/nlp-beginner-text-classification-using-lstm/notebook?scriptVersionId=33960457)


## Model Use

### Intended Use

This model is intended for NLP beginners to learn text classification using LSTM. Its original code uses the sentiment140 dataset.

## Data

The model was trained on the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api.

### Data Mission Statement

According to the creators of the dataset:

"Our approach was unique because our training data was automatically created, as opposed to having humans manual annotate tweets. In our approach, we assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. We used the Twitter Search API to collect these tweets by using keyword search"

## Performance and Limitations

### Performance

According to the creaters of the model:

"It's a pretty good model we trained here in terms of NLP. Around 80% accuracy is good enough considering the baseline human accuracy also pretty low in these tasks. Also, you may go on and explore the dataset, some tweets might have other languages than English. So our Embedding and Tokenizing wont have effect on them. But on practical scenario, this model is good for handling most tasks for Sentiment Analysis.""

## Limitations

[More Information Needed]

### Bias and Fairness

[More Information Needed]


## Feedback

### Where to send questions or comments about the model

Please use [this Google Form](https://forms.gle/Uv7afRH5dvY34ZEs9)

