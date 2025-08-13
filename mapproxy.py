#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple MapProxy - Shows rover path on web map
"""
import json
import time
import threading
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import paho.mqtt.client as paho
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

# Global variables
rover_path_data = []
last_update_time = 0

class SimpleMapHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass
    
    def do_GET(self):
        if self.path == '/':
            self.serve_map_page()
        elif self.path == '/path_data':
            self.serve_path_data()
        else:
            self.send_error(404)
    
    def serve_path_data(self):
        """Serve current rover path data"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        
        web_data = []
        for i, point in enumerate(rover_path_data):
            web_data.append({
                'id': i,
                'lat': point.get('lat', 0),
                'lon': point.get('lon', 0),
                'command': point.get('command', 'Unknown'),
                'time': point.get('time', ''),
                'type': point.get('type', 'PATH')
            })
        
        self.wfile.write(json.dumps(web_data).encode('utf-8'))
    
    def serve_map_page(self):
        """Serve the map webpage"""
        html = """[... your HTML stays exactly the same ...]"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

def mqtt_listener():
    """Listen for path data from subscriber"""
    global rover_path_data, last_update_time
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("✅ MapProxy connected to MQTT")
            client.subscribe("medibot/rover/path_data", qos=0)
        else:
            print(f"❌ MQTT connection failed: {rc}")
    
    def on_message(client, userdata, msg):
        global rover_path_data, last_update_time
        try:
            if msg.topic == "medibot/rover/path_data":
                rover_path_data = json.loads(msg.payload.decode())
                last_update_time = time.time()
                print(f"📍 Path updated: {len(rover_path_data)} points")
        except Exception as e:
            print(f"⚠️ MQTT message error: {e}")
    
    client = paho.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    while True:
        try:
            print(f"📡 Connecting to MQTT broker {MQTT_HOST}:{MQTT_PORT}...")
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.loop_forever()
        except Exception as e:
            print(f"❌ MQTT error: {e}")
            print("🔄 Retrying in 5 seconds...")
            time.sleep(5)

def main():
    """Main function"""
    print("🗺️ Simple MapProxy Starting...")
    print("📍 Shows rover path on web map")
    
    mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
    mqtt_thread.start()
    time.sleep(2)
    
    try:
        print("🌐 Starting web server on http://localhost:8001")
        server = HTTPServer(('localhost', 8001), SimpleMapHandler)
        
        print("\n✅ MapProxy Ready!")
        print("🌐 Open: http://localhost:8001")
        print("📡 Listening: medibot/rover/path_data")
        print("\nPress Ctrl+C to stop...")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 MapProxy stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
