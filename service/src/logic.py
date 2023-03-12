import os


class NeuralNetwork:
    def __init__(self, group_id=0):
        self.group_id = group_id

    def tune(self, texts):
        pass

    def load_weights(self, group_id):
        pass

    def generate(self, hint):
        # print(os.listdir(".")) #результат ['src', 'requirements.txt', 'content', 'train_test_datasets', 'Makefile', 'Dockerfile', 'weights']
        # print(os.listdir("content")) #результат ['.gitempty'] короче он видит эту папку и если что можно возвращать ссылку по типу http://localhost:20000/content/123.png
        return hint
