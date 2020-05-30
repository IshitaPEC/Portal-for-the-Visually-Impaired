from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from chatapp.models import Msgs,Job_Details
from django.template import RequestContext
from django.core import serializers
from time import sleep
import os
from selenium import webdriver
from django.views.generic import ListView,DetailView
from django.utils.decorators import method_decorator
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementClickInterceptedException
import time
from bs4 import BeautifulSoup
import pandas as pd
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
import json
from django.core import serializers
# Create your views here.
def home(request):
    #First home page
    return render(request,'index.html')

def new(request):
    users = User.objects.all()
    return render(request,'new.html',{'users':users})

def result(request):
    print("donee")
    return render(request,'result.html')

def inter(request):
    print("inter ")
    return render(request,'intermediate.html')

def check(request):
    if request.method=='POST':
        username=request.POST['n1']
        password=request.POST['n2']
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            users = User.objects.all()
            print(username)
            return render(request,'result.html',{'current_user':username,'users':users})
        else:
            print("incorrect")
            return redirect('/login')
    users = User.objects.all()
    return render(request,'result.html',{'current_user':request.user.username,'users':users})


def chatbase(request,receiver):
    #POINT IS WHERE DO WE GET THE RECIEVER FROM?
    #The reciever has been got from the dynamic url
    #So whenever url corresponding to this function is calles, it will have a variable name here 'reciever' and it will automatically send it as input
    #But how do we access that url?
    #That can be seen in the templates
    #In the results.html page we can see that we have all the users and its usenames do we have formulated a url using that username
    #If url is of the form /msg/{{user.username}}
    #This is how we can create a dynamic url
    #All the users are present Here
    users = User.objects.all()
    #Chats are ordered by the time that they were written at
    chats = Msgs.objects.order_by('created_at')
    #We filter messages based on the reciever and sender
    temp=Msgs.objects.filter(receiver=request.user.username, username=receiver,read=False)
    read_msg = serializers.serialize("json", temp)
    #We send reciever, all users, current users, chats
    return render(request,'chatbase.html',{'receiver':receiver,'users':users, 'current_user':request.user.username,'chats':chats,'read_msg':read_msg})



def send_msg(request,receiver):
    if request.method=='POST':
        msg=request.POST.get('theInput')
        mymsg = Msgs.objects.create(username=request.user.username, receiver=receiver, msg=msg, del_sen=False, del_rec=False,read=False)
        mymsg.save()
        return HttpResponseRedirect("http://127.0.0.1:8000/intermediate")
    temp=Msgs.objects.filter(receiver=request.user.username, username=receiver,read=False)
    read_msg = serializers.serialize("json", temp)
    return render(request,'writemessage.html',{'receiver':receiver,'read_msg':read_msg})
    # return chatbase(request,receiver)

def read_msgs(request,receiver):
    temp=Msgs.objects.filter(receiver=request.user.username, username=receiver)
    print(temp)
    print(temp.count())
    if(temp.count()!=0):
        temp_obj=temp[temp.count()-1]
        temp2=[]
        temp2.append(temp_obj)
        read_msg = serializers.serialize("json", temp2)
        return render(request,'listenmessage.html',{'receiver':receiver,'read_msg':read_msg})
    return render(request,'listenmessage.html',{'receiver':receiver})





#FOR JOB SCRAPING
#SAMPLE ASSISTANCE REQUIREMENT -> assistance for the Visually Impaired
#SAMPLE TEST CASES: BLIND, ANY STATE AS PER NEED->Hyderabad, Maharashtra, Kerela, Rajasthan
def lookforassistance(request):
    #delete results for any previous search results
    Job_Details.objects.all().delete()
    #form - request is POST
    #Both GET and POST method is used to transfer data from client to server in HTTP protocol but
    #Main difference between POST and GET method is that GET carries request parameter appended in URL string
    #while POST carries request parameter in message body which makes it more secure way of transferring data
    if request.method == "POST":
        # requirement for selenium- install web driver-> set path
        driver = webdriver.Chrome(executable_path=r"C:\Users\Dell\Desktop\chromedriver_win32\chromedriver.exe")
        # get function-> fetches the URL we want to scrape
        driver.get('https://www.indeed.com')
        # to be done with any pop-ups

        driver.refresh()
        #get data from the form
        jobs_list = request.POST.get('job_name')
        #loc = request.POST.get('city')
        jobloc=jobs_list.split()
        # find element by id
        jobs=jobloc[0]
        loc=jobloc[1]
        what = driver.find_element_by_id('text-input-what')
        where = driver.find_element_by_id('text-input-where')
        # Whatever we type on our keyboard is passed to that particular instance via send_keys
        #try:
        what.send_keys(Keys.CONTROL + "a")
        #First Delete if we have any data already on the search bar
        what.send_keys(Keys.DELETE)
        what.send_keys(jobs)
        where.send_keys(Keys.CONTROL + "a")
        where.send_keys(Keys.DELETE)
        where.send_keys(loc)
        where.send_keys(Keys.RETURN)
        #except ElementNotInteractableException:
            #create a dataframe-> append to csv
        df = pd.DataFrame(columns=('Job_Title', 'Company_Name', 'Location', 'Salary'))
        count = 0

        #Run via pages
        while count != 2:
            count = count + 1
            #get the source-code of the page using the selenium driver
            source = driver.page_source
            #We use beautiful Soup to parse over the
            soup = BeautifulSoup(source, 'html.parser')
            SOUP_JOBS = soup.find_all(class_= 'jobsearch-SerpJobCard unifiedRow row result clickcard')
            SELENIUM_JOBS = driver.find_elements_by_xpath("//div[@class = 'location accessible-contrast-color-location']")
            iterator = 0
            for i in range(3) :
                #We use find_all function of the soup object
                # Over here class_= 'jobsearch-SerpJobCard unifiedRow row result clickcard' specifies a unique job
                SOUP_JOBS = soup.find_all(class_= 'jobsearch-SerpJobCard unifiedRow row result clickcard')
                SELENIUM_JOBS = driver.find_elements_by_xpath("//div[@class = 'jobsearch-SerpJobCard unifiedRow row result clickcard']")
                #In that job, we access unique attributes as folllows
                Job_Title = SOUP_JOBS[i].find(attrs={'data-tn-element' : 'jobTitle'})['title']
                Company_Name = SOUP_JOBS[i].find(class_ = 'company').text
                Location = SOUP_JOBS[i].find(class_ = 'location accessible-contrast-color-location').text
                Salary = 'NEGOTIABLE'
                if SOUP_JOBS[i].find(class_ = 'salaryText') != None:
                    Salary = SOUP_JOBS[i].find(class_ = 'salaryText').text

                print(Company_Name)
                url= 'https://www.indeed.co.in'+SOUP_JOBS[i].a['href']

                #Here we store all this information as a csv
                csv_dict = [{'Job_Title':Job_Title, 'Company_Name':Company_Name, 'Location':Location, 'Salary':Salary,'url':url}]
                temp_df_entry = pd.DataFrame(csv_dict)
                #Here we store it in a model
                b=Job_Details.objects.create(job_name=Job_Title, company_name=Company_Name, location=Location, salary=Salary,url=url)
                b.save()
                df = df.append(temp_df_entry, ignore_index = True)
                df.to_csv('God_Given_Gift.csv')
            #Finally we move to the next page
            try:
                next_page = driver.find_element_by_partial_link_text('Next ')
                next_page.click()
            #while dealing  with the pop-up problem
            except ElementClickInterceptedException:
                driver.find_element_by_partial_link_text('No, thanks').click()
                print('Pop Up Closed!!')
                next_page = driver.find_element_by_partial_link_text('Next ')
                next_page.click()
            except:
                pass
        #Takes Us to a new page
        return HttpResponseRedirect(reverse('listofjobs'))
    return render(request,'assistance.html')


#inbuilt feature of django-> allows us to list down the elements of our model
#get queryset enables you to fetch the instances from the model


def listofjobs(request):
    jobs=Job_Details.objects.all()
    job_name_list=[]
    for job in jobs:
        job_name_list.append(job.job_name)
    print(job_name_list)
    job_name_list = json.dumps({'job_name_list':job_name_list})
    context={
        'jobs':jobs,
        'job_name_list':job_name_list
        }


    return render(request,'listofjobs.html',context)
