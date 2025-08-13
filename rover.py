#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Rover Simulator - No SITL required
Simulates rover movement and publishes path data to MapProxy
"""
import paho.mqtt.client as paho
import json
import time
import math
from datetime import datetime
import threading
from dotenv import load_dotenv
import os

load_dotenv()



# Rover Configuration
class SimpleRover:
    def __init__(self, start_lat=-35.363261, start_lon=149.165230):
        self.lat = start_lat  # SITL home position
        self.lon = start_lon
        self.heading = 0  # North = 0, East = 90, South = 180, West = 270
        self.speed = 2.0  # meters per second
        self.is_moving = False
        self.path_data = []
        
    def calculate_new_position(self, direction, distance):
        """Calculate new position based on direction and distance"""
        # Convert distance to lat/lon offset
        # Rough conversion: 1 degree lat ≈ 111,000 meters
        # 1 degree lon ≈ 111,000 * cos(lat) meters
        
        lat_per_meter = 1.0 / 111000.0
        lon_per_meter = 1.0 / (111000.0 * math.cos(math.radians(self.lat)))
        
        # Direction mappings
        direction_map = {
            'forward': 0, 'north': 0,
            'backward': 180, 'south': 180, 
            'left': 270, 'west': 270,
            'right': 90, 'east': 90
        }
        
        bearing = direction_map.get(direction.lower(), 0)
        bearing_rad = math.radians(bearing)
        
        # Calculate offset
        lat_offset = distance * lat_per_meter * math.cos(bearing_rad)
        lon_offset = distance * lon_per_meter * math.sin(bearing_rad)
        
        new_lat = self.lat + lat_offset
        new_lon = self.lon + lon_offset
        
        return new_lat, new_lon
    
    def add_path_point(self, lat, lon, command, point_type="PATH"):
        """Add point to path data"""
        point = {
            'lat': float(lat),
            'lon': float(lon), 
            'command': command,
            'time': datetime.now().strftime('%H:%M:%S'),
            'timestamp': time.time(),
            'type': point_type
        }
        self.path_data.append(point)
        return point
    
    def move(self, direction, distance, mqtt_client=None):
        """Simulate rover movement"""
        if self.is_moving:
            print("🚫 Rover is already moving!")
            return False
            
        try:
            distance = float(distance)
            print(f"\n🎯 Moving rover: {direction} {distance}m")
            
            self.is_moving = True
            
            # Add start point
            start_point = self.add_path_point(
                self.lat, self.lon, 
                f"{direction.upper()} {distance}m - START", 
                "START"
            )
            print(f"📍 Start: {self.lat:.8f}, {self.lon:.8f}")
            
            # Publish start point
            if mqtt_client and mqtt_client.is_connected():
                mqtt_client.publish("medibot/rover/path_data", json.dumps(self.path_data))
            
            # Calculate destination
            dest_lat, dest_lon = self.calculate_new_position(direction, distance)
            print(f"🎯 Target: {dest_lat:.8f}, {dest_lon:.8f}")
            
            # Simulate movement by moving in small steps
            steps = max(10, int(distance))  # At least 10 steps
            step_distance = distance / steps
            movement_time = distance / self.speed  # Total time for movement
            step_time = movement_time / steps
            
            for step in range(steps):
                # Calculate intermediate position
                progress = (step + 1) / steps
                intermediate_lat = self.lat + (dest_lat - self.lat) * progress
                intermediate_lon = self.lon + (dest_lon - self.lon) * progress
                
                # Update rover position
                self.lat = intermediate_lat
                self.lon = intermediate_lon
                
                # Add moving point
                move_point = self.add_path_point(
                    self.lat, self.lon,
                    f"{direction.upper()} - MOVING ({step+1}/{steps})",
                    "MOVING"
                )
                
                print(f"   📍 Step {step+1}/{steps}: {self.lat:.8f}, {self.lon:.8f}")
                
                # Publish update
                if mqtt_client and mqtt_client.is_connected():
                    mqtt_client.publish("medibot/rover/path_data", json.dumps(self.path_data))
                
                # Wait for next step
                time.sleep(step_time)
            
            # Add end point
            end_point = self.add_path_point(
                self.lat, self.lon,
                f"{direction.upper()} {distance}m - END",
                "END"
            )
            
            # Final publish
            if mqtt_client and mqtt_client.is_connected():
                mqtt_client.publish("medibot/rover/path_data", json.dumps(self.path_data))
            
            print("✅ Movement completed!")
            self.is_moving = False
            return True
            
        except Exception as e:
            print(f"❌ Movement error: {e}")
            self.is_moving = False
            return False

# Global rover instance
rover = SimpleRover()
mqtt_client = None

# MQTT Configuration
MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT"))
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        client.subscribe("medibot/rover/move", qos=0)
        print("📡 Subscribed to medibot/rover/move")
    else:
        print(f"❌ MQTT connection failed: {rc}")

def on_message(client, userdata, msg):
    """Handle MQTT movement commands"""
    try:
        if msg.topic == "medibot/rover/move":
            payload = msg.payload.decode()
            print(f"\n📨 Received command: {payload}")
            
            parts = payload.split()
            if len(parts) >= 2:
                direction, distance = parts[0], parts[1]
                
                # Validate command
                valid_directions = ['forward', 'backward', 'left', 'right', 'north', 'south', 'east', 'west']
                if direction.lower() not in valid_directions:
                    print(f"❌ Invalid direction: {direction}")
                    return
                
                try:
                    dist_val = float(distance)
                    if dist_val <= 0 or dist_val > 100:
                        print("❌ Distance must be between 0 and 100 meters")
                        return
                except:
                    print("❌ Invalid distance value")
                    return
                
                # Execute movement in separate thread
                move_thread = threading.Thread(
                    target=rover.move,
                    args=(direction, distance, client),
                    daemon=True
                )
                move_thread.start()
                
            else:
                print("❌ Invalid command format. Use: [direction] [distance]")
                
    except Exception as e:
        print(f"⚠️ Message handling error: {e}")

def status_reporter():
    """Periodically report rover status"""
    global rover, mqtt_client
    
    while True:
        try:
            if mqtt_client and mqtt_client.is_connected():
                status = {
                    'lat': rover.lat,
                    'lon': rover.lon,
                    'heading': rover.heading,
                    'is_moving': rover.is_moving,
                    'path_points': len(rover.path_data),
                    'timestamp': time.time()
                }
                
                # Publish status (optional - for debugging)
                mqtt_client.publish("medibot/rover/status", json.dumps(status))
                
                if not rover.is_moving:
                    print(f"🤖 Rover Status: Lat={rover.lat:.8f}, Lon={rover.lon:.8f}, Points={len(rover.path_data)}")
            
            time.sleep(30)  # Report every 30 seconds
            
        except Exception as e:
            print(f"⚠️ Status report error: {e}")
            time.sleep(30)

def main():
    """Main function"""
    global mqtt_client, rover
    
    print("🤖 Simple Rover Simulator Starting...")
    print("📍 No SITL required - Pure simulation!")
    print(f"🏠 Starting position: {rover.lat:.8f}, {rover.lon:.8f}")
    
    # Setup MQTT client
    try:
        mqtt_client = paho.Client()
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        
        print(f"\n📡 Connecting to MQTT broker: {MQTT_HOST}:{MQTT_PORT}")
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
        
        # Start status reporter thread
        status_thread = threading.Thread(target=status_reporter, daemon=True)
        status_thread.start()
        
        print(f"\n🎉 Rover Simulator Ready!")
        print(f"📡 Listening for movement commands on: medibot/rover/move")
        print(f"🗺️ Publishing path data to: medibot/rover/path_data")
        print(f"🎮 Send commands like: 'forward 5', 'left 3', etc.")
        print(f"\nPress Ctrl+C to exit...")
        
        # Start MQTT loop
        mqtt_client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down rover simulator...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if mqtt_client:
            mqtt_client.disconnect()
        print("👋 Rover simulator stopped!")

if __name__ == "__main__":
    main()