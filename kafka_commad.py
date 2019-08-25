import os
from kafka import KafkaConsumer

class Commad :
    def __init__(self, path = "/usr/local/kafka/bin/" ) :
        self.consumer = KafkaConsumer(
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='latest' , # 'earliest',
            enable_auto_commit= True ,)
        self.path = path
    def show_topic(self,) :
        """
        topic 보여주기
        """
        print("Topic : " , list(self.consumer.topics()))
    def create_topic(self, partition , replication , topic) :
        """
        partition : 파티션 (int)
        replication : 죽었을 때 방지? (int)
        """
        command = "{}kafka-topics.sh --create --bootstrap-server localhost:9092".format(self.path)
        command2 = "{} --replication-factor {} --partitions {} --topic {}".\
        format(command , replication , partition , topic)
        if os.system(command2) == 0 :
            return "topic  `{}` 생성 partition : {} replication : {} ".\
        format(topic , partition , replication)
        else :
            return "topic `{}` 생성 실패".format(topic)
    def delete_topic(self , topics) :
        """
        topics : topic list (list)
        """
        if type(topics) == list :
            pass
        else :
            topics = [topics]
        for topic in topics :
            c = "{}kafka-topics.sh --zookeeper localhost:2181 --delete --topic {}".\
            format(self.path , topic)
            if os.system(c) == 0 :
                print("Topic `{}` 제거 완료".format(topic))
            else :
                print("Topic `{}` 제거 실패 or 이미 제거".format(topic))
    def delete_Consumer_group(self, group) :
        c = "{}kafka-consumer-groups.sh --zookeeper localhost:2181 --delete --group {}".\
        format(self.path , group)
        if os.system(c) == 0 :
            print("Consumer Group `{}` 제거 완료".format(group))
        else :
            print("Consumer Group `{}` 제거 실패 or 이미 제거".format(group))

        