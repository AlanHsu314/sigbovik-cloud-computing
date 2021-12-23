from cmu_112_graphics import *
import random
import time
import requests
import webbrowser

base_url = 'http://ec2-52-15-193-200.us-east-2.compute.amazonaws.com:5000'

NUM_IMAGES = 1013

def appStarted(app):
    if not hasattr(app, 'splashScreen'):
        app.splashScreen = True
        webbrowser.open_new("https://www.youtube.com/watch?v=8YGlzSl6cxU&ab_channel=%D7%92%D7%99%D7%90%D7%90%D7%92%D7%9D%D7%A4%D7%95%D7%A1%D7%A4%D7%95%D7%A1")
        return

    idx = random.randint(1, 1013)
    app.idx = idx
    print(idx)
    app.gt_map = app.loadImage(base_url + f'/gt_map/{idx}')
    app.gt_map = app.scaleImage(app.gt_map, 1/2)
    app.image = app.loadImage(base_url + f'/img/{idx}')
    app.image = app.scaleImage(app.image, 1/2)

    app.image_display = ImageTk.PhotoImage(app.image)
    app.gt_map_display = ImageTk.PhotoImage(app.gt_map)

    app.startTime = time.time()

    app.boxes = [(0, 300, app.width//2, 400), (app.width//2, 300, app.width, 400),
                 (0, 400, app.width//2, 500), (app.width//2, 400, app.width, 500)]

    app.colors = ['cyan', 'red', 'green', 'yellow']

    app.choices = ['dog', 'cat', 'cow', 'koz']

def timerFired(app):
    if app.splashScreen:
        return
    if time.time() - app.startTime > 10:
        label = random.choice(app.choices)
        sendMessage(app, label)
        app.showMessage(f"Too slow.\nWe have chosen {label.upper()} for you.")
        appStarted(app)

def sendMessage(app, label):
    requests.post(base_url + f'/annotate/{app.idx}/{label}')

def keyPressed(app, event):
    if app.splashScreen:
        app.splashScreen = False
        appStarted(app)

def mousePressed(app, event):
    if app.splashScreen:
        app.splashScreen = False
        appStarted(app)

    for label, (x1, y1, x2, y2) in zip(app.choices, app.boxes):
        if x1 < event.x < x2 and y1 < event.y < y2:
            sendMessage(app, label)
            appStarted(app)
            return
    
def redrawAll(app, canvas):
    if app.splashScreen:
        canvas.create_text(app.width//2, app.height//2,
            text='Welcome to my 112 tp.\nA youtube video should open automatically.\nFor maximum immersion, I recommend playing\nit on loop while you play my game.',
            font='ComicSans 20 bold', fill='black')
        return
    canvas.create_image(0, 0, image=app.image_display, anchor='nw')
    canvas.create_image(app.width//2, 0, image=app.gt_map_display, anchor='nw')

    for box, color, label in zip(app.boxes, app.colors, app.choices):
        canvas.create_rectangle(*box, fill=color, outline='black')
        canvas.create_text((box[0] + box[2]) // 2, (box[1] + box[3]) // 2, 
            text=label.upper(), fill='black')

runApp(width=600, height=500)