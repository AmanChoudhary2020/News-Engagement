def generate_document_topic_counts(model,corpus,file,topic_counts):
    topics_arr = []
    for doc in corpus:
        topics = model[doc]
        max = 0
        max_doc = []
        for el in topics:
            if (el[1] > max):
                max = el[1]
                max_doc = el
           
        topic = max_doc[0]

        if (topic == 0):
            topic = "Middle East"
        elif (topic == 1):
            topic = "Data Security"
        elif (topic == 2):
            topic = "Politics"
        elif (topic == 3):
            topic = "COVID"
        elif (topic == 4): 
            topic = "China"

        topics_arr.append(topic)

        if (topic not in topic_counts.keys()):
            topic_counts[topic] = 1
        else:
            topic_counts[topic] += 1

    file['topic'] = topics_arr
