# Book-Sentiment-Analysis-Trends 📖💭
Discover the emotional journey of a book with sentiment analysis. This project harnesses the power of RoBERTa to map out sentiment trends throughout a literary work.


## Overview
With RoBERTa, this tool processes sentences in a book, identifying the emotions conveyed. By preserving the original order of sentences and collating the results, it crafts a visual representation of the sentiment flow as the narrative progresses.


## Usage
1. Prepare your Book: Obtain a book in .txt format. A good source for classic literature is Project Gutenberg (https://www.gutenberg.org/).
2. Run the Analysis:
  - Execute main.py and replace the placeholder .txt with your book. This step will produce a .pkl file.
  - Execute process_results.py to save the sentiment data into a .csv file.
  - Finally, run show.py to visualize the sentiment trends with plots.


## Example Results
Analysis of "Romeo and Juliet":

![](./romeo_and_juliet_join_plot.png)

![](./romeo_and_juliet_tight_plot.png)
