from flask import Flask
from app import views
from camera import VideoCamera
from flask import Flask, render_template, Response
import time

app = Flask(__name__)

# url
app.add_url_rule('/base','base',view_func=views.base)
app.add_url_rule('/','index',view_func=views.index)
app.add_url_rule('/faceapp', 'faceapp', view_func=views.faceapp)
app.add_url_rule('/expression', 'expression', view_func=views.expression,methods=['GET','POST'])
app.add_url_rule('/realTime', 'realTime', view_func=views.realTime)

def gen(camera):
    start_time = time.time()

    while True:
        if ( int(time.time() - start_time) >=1 ):
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            start_time = time.time()            

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# run

if __name__ == "__main__":
    app.run(debug=True)