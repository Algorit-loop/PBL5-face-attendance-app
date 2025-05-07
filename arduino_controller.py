import serial
import time

arduino = serial.Serial(port='COM6', baudrate=9600, timeout=1)
time.sleep(2)  # Wait for the Arduino to reset

def shine_led():
    try:
        arduino.write(b'1') 
        return {"message": "LED ON signal sent to Arduino!"}
    except Exception as e:
        return {"error": str(e)}
    
def off_led():
    try:
        arduino.write(b'0') 
        return {"message": "LED ON signal sent to Arduino!"}
    except Exception as e:
        return {"error": str(e)}
    
def open_door():
    try:
        arduino.write(b'o')  
        return {"message": "Door open!"}
    except Exception as e:
        return {"error": str(e)}
    
def close_door():
    try:
        arduino.write(b'c')  
        return {"message": "Door open!"}
    except Exception as e:
        return {"error": str(e)}