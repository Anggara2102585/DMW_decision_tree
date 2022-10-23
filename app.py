import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Init Flask
app = Flask(__name__)

df = pd.read_csv('buys_computer.csv')

# pisahkan kolom feature(ke x) dengan target(ke y)
x = df.iloc[:,:-1].values   # slicing berdasarkan indeks(iloc), [:,:-1] semua row dan column awal sampai kedua dari akhir
y = df.iloc[:,-1].values

enc = LabelEncoder()    # untuk mengubah string menjadi angka

# mengubah data string menjadi numerik
x[:,0] = enc.fit_transform(x[:,0])  # age
x[:,1] = enc.fit_transform(x[:,1])  # income
x[:,2] = enc.fit_transform(x[:,2])  # student
x[:,3] = enc.fit_transform(x[:,3])  # credit_rating
y      = enc.fit_transform(y)       # buys_computer
""" 
hasil:
age: middle_aged(0), senior(1), youth(2)
income: high(0), low(1), medium(2)
student: no(0), yes(1)
credit_rating: excellent(0), fair(1)
buys_computer: no(0), yes(1)
"""

model = DecisionTreeClassifier()

# melakukan pelatihan model terhadap data
model.fit(x,y)

# Route beranda atau halaman awal
@app.route("/")
# function index
def index():
    # render view
    return render_template('index.html')

# route prediction untuk pasing data dari view menggunakan method POST
@app.route('/prediction', methods=['POST'])
# function prediction
def prediction():
    # simpan data dari masukan user
    age             = int(request.form['age'])
    income          = int(request.form['income'])
    student         = int(request.form['student'])
    credit_rating   = int(request.form['credit_rating'])

    # melakukan prediksi dari masukan user menggunakan model yg sudah dilatih
    predicted = model.predict([[age, income, student, credit_rating]])

    # ubah value age dari number menjadi string
    if age == 0:
        age = "Middle Aged"
    elif age == 1:
        age = "Senior"
    elif age == 2:
        age = "Youth"

    # ubah value income dari number menjadi string
    if income == 0:
        income = "High"
    elif income == 1:
        income = "Low"
    elif income == 2:
        income = "Medium"

    # ubah value student dari number menjadi string
    student = "Yes" if student else "No"

    # ubah value credit_rating dari number menjadi string
    credit_rating = "Fair" if credit_rating else "Excellent"

    #ubah value hasil prediksi dari number menjadi string
    predicted = "Yes" if predicted else "No"

    # render template dan passing data hasil prediksi ke dalam view
    return render_template('index.html', age = age, income = income, student = student, credit_rating = credit_rating, predicted = predicted)

# driver
if __name__ == "__main__":
    app.run(debug=True) # change to true while debugging