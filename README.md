# News-Engagement Project Overview
News organizations often turn to social media to share their stories, in order to reach a wide swath of readers quickly. While useful, social media platforms such as Twitter and Reddit also attract unproductive discussions that may detract from the news story, such as jokes about the news story's headline that don't engage with the article content. News organizations therefore often need a way to find engaging comments automatically, to better understand how their readership is reacting to their stories.

This project provided an in-depth linguistic analysis of questions and comments posed to news stories on social media, with the goal of helping news organizations anticipate the information needs of their audience. This consisted of collecting social media data related to several major news organizations, labeling the data according to level of engagement, and developing machine learning methods to predict comment engagement based on the text of the news article and the headline.

# Project Methods
For the scope of this project, I used articles from The Guardian, a British daily newspaper, and comments on Reddit posts, particularly the news, worldnews, qualitynews, inthenews, and anythinggoesnews subreddits. In the future, the scope of this project can be expanded to include other news organizations like New York Times, Reuters, Washington Post, and AP News, and other social media platforms like Twitter. 

# Determining Engagement Metrics
Jaccard Similarity: the intersection between a comment and the respective article text - how many words overlap between both - and divides this by the total number of words in both bodies of text. 

Semantic Similarity: using an NLP technique called word2vec, converts each word to a numeric vector and computing cosine similarities between these vectors

Comment: “I pray the Cardinal recovers with a new outlook on life, the virus, vaccines, and the individuals /organizations he has previously maligned; and that he will evangelize for the use of the vaccine, masks, social distancing, and any other methods we can in order to control the spread of this virus and unnecessary loss of life.”

Article Headline: “Vaccine skeptic US cardinal on ventilator after Covid diagnosis”

Jaccard Similarity: 4/(56 + 6) = 0.071428571 
Semantic Similarity: 0.591390132

# Registering for The Guardian Key
Go to https://open-platform.theguardian.com/, click "register for a developer key", and follow the instructions. 
