
# Dataset Card for Sentiment 140 Dataset

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:**  http://help.sentiment140.com/home 
- **Repository:** https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/sentiment140/sentiment140.py
- **Paper:** Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. CS224N project report, Stanford, 1(12), 2009.
- **Leaderboard:** https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477&sortBy=voteCount
- **Point of Contact:** http://help.sentiment140.com/contact

### Dataset Summary

This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter API. The tweets have been labeled according to the sentiment they create (0 = negative and 4 = positive).

### Supported Tasks and Leaderboards

Brand management (e.g. windows 10)

Polling (e.g. obama)

Planning a purchase (e.g. kindle)

### Languages

Sentiment140 API supports English and Spanish. English is the default language.

## Dataset Structure

### Data Instances

An example of 'train' looks as follows.

`{
    "date": "23-04-2010",
    "query": "NO_QUERY",
    "sentiment": 3,
    "text": "train message",
    "user": "train user"
}`

### Data Fields

It contains the following 6 fields:

- target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

- ids: The id of the tweet ( 2087)

- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

- flag: The query (lyx). If there is no query, then this value is NO_QUERY.

- user: the user that tweeted (robotickilldozr)

- text: the text of the tweet (Lyx is cool)

### Data Splits

The data set has only one split: train set, with 1600000 instances.

## Dataset Creation

### Curation Rationale

Sentiment140 was created by Alec Go, Richa Bhayani, and Lei Huang, who were Computer Science graduate students at Stanford University.

### Source Data

Tweets with positive and negative emoticons.

#### Initial Data Collection and Normalization

The training data was post-processed. Emoticons are removed for training purposes. All tweets containing both positive and negative emoticons are filtered out and removed. Retweets or tweets copied from another user have been removed. Tweets containing ":P" are removed. The retweet or repeated tweets are removed from the dataset. The test data was manually collected using web applications.

#### Who are the source language producers?

Twitter users. 

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

Alec Go, Richa Bhayani and Lei Huang

### Licensing Information


### Citation Information

@article{go2009twitter,
  title={Twitter sentiment classification using distant supervision},
  author={Go, Alec and Bhayani, Richa and Huang, Lei},
  journal={CS224N project report, Stanford},
  volume={1},
  number={12},
  pages={2009},
  year={2009}
}

### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.

