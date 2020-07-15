from flask import Flask, render_template,request
import pickle
import os, string, re
import glob
from pyvi import ViTokenizer, ViPosTagger

app = Flask(__name__)

list_label = ["Chính Trị Xã Hội","Đời Sống","Khoa Học","Kinh Doanh","Pháp Luật","Sức Khỏe","Thế Giới","Thể Thao","Văn Hóa","Vi Tính"]

svm_model = "SVM_model.pkl"  
with open(svm_model, 'rb') as f:
    classifier = pickle.load(f)
    
vectorizer_save = "tfidf_vectorizer.pkl"  
with open(vectorizer_save, 'rb') as f2:
    vectorizer = pickle.load(f2)


# Các bước tiền xử lý văn bản

def normalText(sent):
    sent = str(sent).replace('_',' ').replace('/',' trên ')
    sent = re.sub('-{2,}','',sent)
    sent = re.sub('\\s+',' ', sent)
    patHagTag = r'#\s?[aăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ]+'
    patURL = r"(?:http://|www.)[^\"]+"
    sent = re.sub(patURL,'website',sent)
    sent = re.sub(patHagTag,' hagtag ',sent)
    sent = re.sub('\.+','.',sent)
    sent = re.sub('(hagtag\\s+)+',' hagtag ',sent)
    sent = re.sub('\\s+',' ',sent)
    return sent

# Tách từ
def tokenizer(text):
    token = ViTokenizer.tokenize(text)
    return token

# Loại bỏ stopword trong tiếng Việt
with open("vi_stopwords.txt","r",encoding='utf8') as file:
    stopwords = file.read().split('\n')
def remove_stopword(text):
    text_new = " "
    for word in text.split(' '):
        if word.replace("_", " ").strip() not in stopwords:
            text_new +=  word +" "
    return text_new

def clean_doc(doc):
    # Tách dấu câu ra khỏi chữ trước khi tách từ
    for punc in string.punctuation:
        doc = doc.replace(punc,' '+ punc + ' ')
    # Thay thế các link web = website, hagtag ~ hagtag
    doc = normalText(doc)
    # Tách từ đối với dữ liệu
    doc = tokenizer(doc)
    # Đưa tất cả về chữ thường
    doc = doc.lower()
    # Xóa nhiều khoảng trắng thành 1 khoảng trắng
    doc = re.sub(r"\?", " \? ", doc)
    # Thay thế các giá trị số thành ký tự num
    doc = re.sub(r"[0-9]+", " num ", doc)
    # Xóa bỏ các dấu câu không cần thiết
    for punc in string.punctuation:
        if punc !="_":
            doc = doc.replace(punc,' ')
    doc = re.sub('\\s+',' ',doc)
    return doc


@app.route("/")
def home():
    return render_template("Demo.html")

@app.route("/Demo")
def demo():
    return render_template("Demo.html")


@app.route("/dulieu")
def dulieu():
    print("TRUE")
    return render_template("dulieu.html")


@app.route("/phantichketqua")
def phantichketqua():
    return render_template("phantichketqua.html")


@app.route("/dudoanfile")
def dudoanfile():
    return render_template("InputFile.html")


@app.route("/video")
def video():
    return render_template("videodemo.html")


@app.route("/ketquadudoanfile", methods=['POST','GET'])
def ketquadudoanfile():
    fileupload = request.files['file']
    try:
        text = fileupload.read().decode('utf-16')
    except:
        text = fileupload.read().decode('utf-8')
    text_cleaned = clean_doc(text)
    input_vector = vectorizer.transform([text_cleaned])
    index = classifier.predict(input_vector)[0]
    label = list_label[index]
    print(list_label[index])
    return render_template("InputFile_Result.html", data = [{"query":text, "label" : label}])


@app.route("/analysis/", methods=['POST','GET'])
def classify_text():
    text = request.form['query']
    text_cleaned = clean_doc(text)
    input_vector = vectorizer.transform([text_cleaned])
    index = classifier.predict(input_vector)[0]
    label = list_label[index]
    return render_template("results.html", data = [{"query":text, "label" : label}])

if __name__ == "__main__":
    app.run(debug=True)
