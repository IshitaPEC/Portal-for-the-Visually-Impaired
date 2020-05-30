from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login as auth_login
from login.models import Records

from django.urls import reverse
from django.template import RequestContext


import numpy as np
import cv2
import pickle

from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc
from PIL import Image

from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from mychat.settings import BASE_DIR
# Create your views here.

def webcam(request):
    return render(request,'webcam.html')


def create_dataset(request):
    #print request.POST

    #Corresponding to this user id, we will have saved the login details of the user
    #userId = request.POST.get('userId')

    #print (cv2.__version__)
    # Detect face
    #Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier(BASE_DIR+'/ml/haarcascade_frontalface_default.xml')
    #capture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = 1
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        #cam.read will return the status variable and the captured colored image
        #returns bool variable whether we have capture the image properly
        ret, img = cam.read()
        #the returned img is a colored image but for the classifier to work we need a greyscale image
        #to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        #1.3 -> scale factor- >How much image size is reduced at each scale
        #5->min_neighbors-> minimum neighbours of a rectangle that must exist to consider that rectangle
        #min size, max size -> other arguments of detect multi scale
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for(x,y,w,h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite(BASE_DIR+'/ml/dataset/user.'+str(id)+'.'+str(sampleNum)+'.jpg', gray[y:y+h,x:x+w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            #This is the rectangle which is around our face when we run the command-> green rectangle around our face
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        #This is just to shhow the user how s/he is looking or will look
        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Face",img)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        #once we have clicked 35 images of the user, break, no more images needed
        if(sampleNum>35):
            break
    #releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

    return render(request, 'new.html')



def eigenTrain(request):
    #CREATING OUR MODEL
    path = BASE_DIR+'/ml/dataset'
    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= df.getImagesWithID(path)
    # Spliting training and testing dataset
    #Here faces deals with our faces-> it is the feature array and ids deals with the labels (what we have to finally predict ggiven corresponding our faces)
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    n_classes = y_test.size
    target_names = ['Ishita Arora', 'Udish Arora']
    n_components = 15
    #Here we are extracting the top n eigenfaces from the total faces
    #n -> we have defined as n_components and total faces are X_train.shape[0]
    #Shape gives us the dimensions of our numpy array
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))

    #Create object of PCA class-> unsupervised machine learning model
    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    #Purpose of reshape in numpy is to change the shape of my array (2D->1D), (3D->2D) etc
    #Here we are changing the dimensions to be able to plot it properly
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    #GridSearchCV -> hyperparameter tuning
    #by crossvalidation, does the best parameter selection of C and gamma
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    print(X_train_pca)
    print(y_train)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    #this clf.best_estimator gives us the best match of parametes-> C,epsilon, gamma etc

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    #Note that for the classifier-> our features were the numpy arrays and the labels were corresponding ids
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    # ##############################

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the gallery of the most significative eigenfaces
    #Purpose of the statement below is to simply number the eigen faces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    #FIRST WE SAVE THE STATE OF THE CLASSIFIER-> SVC
    svm_pkl_filename = BASE_DIR+'/ml/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    #Purpose is to save the state of my classifier i.e the values of 'C','gamma' among others -. so that we donot have to retune our hyperparameter
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()


    #NOW WE SAVE THE STATE OF OUR PCA MODEL
    pca_pkl_filename = BASE_DIR+'/ml/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

    return render(request,'new.html')



def detectImage(request):

    #We reload the svm classifier as we had stored in our pickle file
    svm_pkl_filename =  BASE_DIR+'/ml/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    #print "Loaded SVM model :: ", svm_model

    #We rolad our PCA as we had stored it in our pickle file
    pca_pkl_filename =  BASE_DIR+'/ml/serializer/pca_state.pkl'
    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    #print pca

    '''
    First Save image as cv2.imread only accepts path
    '''
    #im = Image.open(userImage)
    #im.show()
    #We get the path of the image
    imgPath = BASE_DIR+'/ml/uploadedImages/mypic.jpg'


    '''
    Input Image
    '''
    try:
        inputImg = casc.facecrop(imgPath)
        inputImg.show()
    except :
        print("No face detected, or image not recognized")
        return redirect('/error_image')

    #Convert our imge into a numpy array of pixels
    imgNp = np.array(inputImg, 'uint8')
    #Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    #print imgFlatten
    #print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    #print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    print (pred[0])
    return redirect('details/'+str(object=pred[0]))

    #return HttpResponseRedirect("http://127.0.0.1:8000/intermediate")



def details(request, id):
    record = Records.objects.get(id=id)
    print(id)
    uname= record.name
    passw= record.password
    user = authenticate(username=uname, password=passw)
    context = {
        'record' : record
    }
    if user is not None:
        auth.login(request,user)
        users = User.objects.all()
        #print(username)
        return render(request,'intermediate.html',{'current_user':uname,'users':users})
    return render(request, 'details.html',context)


def onetime(request):
    user=User.objects.create_user(username='tul',password='a3122',email='a@gmail.com')
    user.save()
    user1=User.objects.create_user(username='ari',password='p3097',email='p@gmail.com')
    user1.save()
    print('created')
    return redirect('login')


def check(request):
    if request.method == 'POST':
        uname = request.POST.get('n1')
        #emailid=request.POST.get('emailid')
        passw = request.POST.get('n2')
        user = authenticate(username=uname, password=passw)
        if user:
            if user.is_active:
                auth_login(request, user)
                users=User.objects.all()
                return render(request,'result.html',{'current_user':username,'users':users})
            else:
                return HttpResponse("<h1>User Inactive!</h1>")
        else:
            return HttpResponse("<h1>User Unauthenticated!</h1>")
    else:
        print("incorrect")
        return redirect('/')
