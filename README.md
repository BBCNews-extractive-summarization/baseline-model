# baseline_model
## Detailed tasks for the coding part (subject to changes)

### split Dataset into training, development, and test corpus (0)

use test_split to split the test set data into 80%, 10%, and 10% (0.1)

### get sentence embeddings (1)

#### title extraction (1.1)
splitting the text into sentences (1.2)
removing the stop words (1.3)
sentence embedding using Universal Sentence Encoder(1.4)
(https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)

### analyze cos similarities (2)

#### Location method: <strong>score(Si) = 1/i * 0.8+ 1/j * 0.2 </strong>  (i = the position the sentence is in the article, j = the position the sentence is in the paragraph) compute the score for each sentence except for the title  Close to beginning of doc; Beginning of paragraph (2.1) 
(https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.1413&rep=rep1&type=pdf)
reference:

   
#### Title method (2.1): 
Get cos similarity scores between sentence embedding of all the sentences （including title) using <strong>NumPy.inner(feature, feature)</strong>
Normalize the score using the cos similarity score of the title itself to make the final score in the range of 0-1 (?) and add the score to the previous score of the sentence
(https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.1413&rep=rep1&type=pdf)

#### Reduce frequency (2.2)
Add together the sentence score got from the location and title method
Semantic clustering
<strong>(#sentences in a cluster/ #sentences in the article)</strong>
rank the sentences based on their sentence scores and pick the sentence with the highest score
https://aclanthology.org/P06-1106.pdf

*Other possibility: Decide the threshold of the cos similarity score (determine what score counts as similar semantic meaning, eg: cos similarity score < 0.1 for two sentences)

Rank all sentences and take the top K sentences (3)

Add all the categories of scores up and rank all the sentences
Pick the top K sentences. K = #sentences in sample summary
Recover the original order as every sentence was in the original article

### Evaluation (4)


### reference paper link
https://link.springer.com/chapter/10.1007/978-981-33-4893-6_20
