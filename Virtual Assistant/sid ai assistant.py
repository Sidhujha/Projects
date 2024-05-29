import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import os
import webbrowser

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

engine.say("hello sir i am a personal assistant sid what can i do for you")
engine.runAndWait()

def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'sid' in command:
                command = command.replace('sid', '')
                print(command)
    except:
        pass
    return command


def run_sid():
    command = take_command()
    print(command)
    if 'play' in command or 'gana' in command or 'gane' in command or 'video' in command:
        song = command.replace('play', '')
        talk('playing ')
        pywhatkit.playonyt(song)
    elif 'time' in command:    
        time=datetime.datetime.now().strftime('%I:%M %p')
        print(time)
        talk('current time is'+time)   
    elif 'who is' in command:
        person = command.replace('who is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info) 
    elif 'kaun hai' in command:
        person = command.replace('kaun hai', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info)     
    elif "who are you" in command or "tell me about yourself" in command:
        about="hello,my name is sid,i am an assistant,you can command me to perform various tasks."
        print(about)
        talk(about)
    elif "your name" in command or "naam" in command:
        name="my name is sid"
        print(name)
        talk(name)
    elif "who am i" in command or "main kaun" in command:
        me="you are my boss"
        print(me)
        talk(me)
    elif "namaste" in command:
        a="namastey ji"
        print(a)
        talk(a)
    elif "ram ram ji" in command:
        a="aapko bee raam raam ji"
        print(a)
        talk(a)
    elif "how are you" in command or "tum kaisi ho" in command or "kya hal" in command:
        b="i am fine, thank you\nhow are you?"
        print(b)
        talk(b)
    elif "open" in command.lower():
        if "chrome" in command.lower():
            talk("opening google chrome")
            os.startfile(
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"
            )
        elif "youtube" in command.lower():
            talk("opening youtube")
            webbrowser.open("https://www.youtube.com/")
        elif "google" in command.lower():
            talk("opening google")
            webbrowser.open("https://www.google.com/")
    elif "search" in command.lower():
        ind=command.lower().split().index("search")
        search=command.split()[ind+1:]
        webbrowser.open(
            "https://www.google.com/search?q=" + "+".join(search)
        )
        talk("searching" + str(search) + "on google")
    elif "google" in command.lower():
        ind=command.lower().split().index("google")
        search=command.split()[ind+1:]
        webbrowser.open(
            "https://www.google.com/search?q=" + "+".join(search)
        )
        talk("searching" + str(search) + "on google")
    elif "where is" in command:
        ind=command.lower().split().index("is")
        loc=command.split()[ind+1:]
        url="https://www.google.com/maps/place/" + "".join(loc)
        talk("this is where" + str(loc) + "is")
        webbrowser.open(url)
    else:
        talk('please say the command again.') 

run_sid()
