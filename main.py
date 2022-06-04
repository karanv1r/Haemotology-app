from flask import Flask,render_template,request,make_response
from flask_pymongo import MongoClient
from werkzeug.utils import secure_filename
import pdfkit


from cancer import testcancer
from malaria import testmalaria
from blood_classify import classify

import os


client = MongoClient('mongodb+srv://karan:karan@cluster0.raf06.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db = client.TaskManager

if db.settings.find({'name': 'task_id'}).count() <= 0:
    print("task_id Not found, creating....")
    db.settings.insert_one({'name':'task_id', 'value':0})

app=Flask(__name__)
app.config['SECRET_KEY']='thisisasecret'
app.config['UPLOADED_IMAGES_DEST']='uploads/images'

app_root = os.path.dirname(os.path.abspath(__file__))
app_root = app_root.replace("\\", "/")
print("APPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPROUTEEEEEEEEEEEEEEEEEEEEEE",app_root)
app.config['UPLOAD_FOLDER'] = app_root
@app.route("/")
def home():
    return render_template('patho-test-site.html'
                           )
@app.route('/',methods = ['POST'])
def home_post():
   print("hhhhhhoooooooooo")
   if request.method == 'POST':
      print("yooooooo")
      ans1="na"
      ans3="na"
      ans2="na"
      ans="na"
      fname = request.form['FirstNameTestSite']
      lname=request.form['LastNameTestSite']
      dref=request.form['DoctorTestSite']
      pno=request.form['PhoneTestSite']
      city=request.form['CityTestSite']
      ml=request.form['mlmodel']

      file = request.files['ImageTestSite']
      # print("yuhuuuuuuuuuu")
      filename = secure_filename(file.filename)
      kick=os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(kick)
      # print(type(filename))

      # print("this is "+filename)
      kick = kick.replace("\\", "/")
      print(kick)


      if(ml=="malaria"):
          try:
              ans=testmalaria(kick)
              print(ans)
              if(ans[0]=='p'):
                ans1=1
              else:
                ans1=0
          except Exception:
              ans="Invalid Image Input"
              return ans

      elif(ml=="cancer"):
          try:
              ans=testcancer(kick)
              print(ans)
              if (ans[0] == 'p'):
                ans2 = 1
              else:
                ans2 = 0
          except Exception:
              ans="Invalid Image Input"
              return ans
      elif(ml=="blood_cell_type"):
          try:
              ans=classify(kick)
              print(ans)
              ans3=ans
          except Exception:
              ans="Invalid Image Input"
              return ans

      test_result=ml


      db.settings.insert_one({'first-name':fname,'last-name':lname,'doctor-reference':dref,'phone-no.':pno,'city':city,
                              'test-result':test_result,'malaria':ans1,'leukemia':ans2,'blood-cell':ans3})

      global rendered
      rendered=render_template('final_ans.html',fname=fname,lname=lname,dref=dref,city=city,
                           ans=ans)
      return render_template('final_ans.html',fname=fname,lname=lname,dref=dref,city=city,
                           ans=ans)
@app.route("/history")
def history():

    res_to_history=db.settings.find ()


    # print("count is ",res_to_history)
    i=0
    arr=[]
    for doc in res_to_history:
        arr.append(doc)
        # print(doc)
        i=i+1
    print(arr)
    arr.reverse()
    return render_template('history.html'
                           ,res_to_history=arr,num=i)

@app.route("/pdf")
def pdf():
    path_wkhtmltopdf = r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    pdf=pdfkit.from_string(rendered,False,configuration=config)
    response=make_response(pdf)
    response.headers['Content-Type']='application/pdf'
    response.headers['Content-Disposition']='inline;filename=output.pdf'
    return response


app.run(debug=True)