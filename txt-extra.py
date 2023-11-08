


if __name__ == "__main__":
    with open('train.txt', 'r', encoding='UTF-8') as file:
        newcontent = file.read().replace('.jpg','_1.jpg_pro.jpg')
    with open('train.txt', 'a', encoding='UTF-8') as file:
        file.write(newcontent)
