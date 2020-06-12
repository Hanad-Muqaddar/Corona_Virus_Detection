from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


file = open('model.pkl', 'rb')
lg = pickle.load(file)
file.close()

@app.route('/', methods=["GET", 'POST'] )
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        fever = float(mydict['fever'])
        age = int(mydict['age'])
        pain = int(mydict['pain'])
        runnynose = int(mydict['runnynose'])
        breath = int(mydict['breath'])
        input_features = [fever,pain,age, runnynose, breath]
        result = lg.predict([input_features])
        # return 'Hello, World!' + str(result)
        if result == [0]:
            result = "Positive"
        else:
            result = "Negative"
        return render_template('show.html', inf=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
