from django.shortcuts import render,redirect
from django.template import RequestContext
from django.core import serializers
import requests
import json
# Create your views here.
API_KEY='172e38ca9a464cc282a1400d18caf814'
def get_news(request):
	if request.method=='POST':
		num=request.POST.get('theInput')
		if(num.lower()=='a'):
			URL="http://newsapi.org/v2/top-headlines?country=in&category=science&apiKey="+API_KEY
		elif(num.lower()=='b'):
			URL="http://newsapi.org/v2/top-headlines?country=in&category=entertainment&apiKey="+API_KEY
		elif(num.lower()=='c'):
			URL="http://newsapi.org/v2/top-headlines?country=in&category=sports&apiKey="+API_KEY
		elif(num.lower()=='d'):
			URL="http://newsapi.org/v2/top-headlines?country=in&category=health&apiKey="+API_KEY
		else:
			URL="http://newsapi.org/v2/top-headlines?country=in&category=technology&apiKey="+API_KEY

		web_page=requests.get(URL).json()

		article=web_page["articles"]
		results=[]
		#Get Title
		for i in article:
			results.append(i["title"])
		#results=results[0:3]
		read_news = json.dumps({'results':results})
		#results contain all title of news articles
		#Print(Speech) the results
		print("Top headlines are")
		print(num)
		for i in range(len(results)):
			print(i+1,results[i])
		return render(request,'news_updates.html',{'results':results,'read_news':read_news})
	return render(request,'news.html')
	#return render(request,'news.html')

def weather_info(request):
	#API_KEY=apikey
	api_key='bf3b8f84fa9149f6c1b8b1c0858a44a9'
	#URL for the main website openweathermap.org
	URL="http://api.openweathermap.org/data/2.5/weather?"
	#take input city
	#convert this to speech
	if request.method=='POST':
		city_name=request.POST.get('theInput')

	#getting complete URL
		comp_url=URL+"q="+city_name+"&appid="+api_key

		response=requests.get(comp_url)
		#create json object so as to get information using "main" keyword.
		x=response.json()

		y=x["main"]

		#get temperature and humidity
		temperature=y["temp"]

		humidity=y["humidity"]
		#to speech

		print("Temperature:",temperature)
		print("Humidity:",humidity)
		result=[]
		result.append('temperature in Kelvin is ')
		result.append(temperature)
		result.append('humidity is ')
		result.append(humidity)
		read_weather = json.dumps({'result':result})
		return render(request,'weather_updates.html',{'temperature':temperature,'humidity':humidity,'read_weather':read_weather})
	return render(request,'weather.html')
