#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Publisher - Sends movement commands to rover
"""
import paho.mqtt.publish as publish
import time
from dotenv import load_dotenv
import os

# MQTT Configuration
load_dotenv()

MQTT_HOST = os.getenv("MQTT_HOST")
MQTT_PORT = int(os.getenv("MQTT_PORT"))

def send_movement_sequence():
    """Send a sequence of movement commands"""
    print("🚀 Starting movement sequence...")
    
    commands = [
        "forward 5",
        "right 3", 
        "backward 4",
        "left 2",
        "forward 3"
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"📡 Sending command {i}/{len(commands)}: {cmd}")
        
        try:
            publish.single(
                topic="medibot/rover/move",
                payload=cmd,
                hostname=MQTT_HOST,
                port=MQTT_PORT
            )
            print(f"✅ Command sent successfully")
            
            # Wait between commands to allow rover to complete movement
            if i < len(commands):
                print("⏳ Waiting 10 seconds for movement to complete...")
                time.sleep(10)
                
        except Exception as e:
            print(f"❌ Failed to send command: {e}")
    
    print("🏁 Movement sequence completed!")

def interactive_mode():
    """Interactive command sender"""
    print("\n🎮 Interactive Command Mode")
    print("Available commands:")
    print("  forward [distance]   - Move forward X meters")
    print("  backward [distance]  - Move backward X meters") 
    print("  left [distance]      - Move left X meters")
    print("  right [distance]     - Move right X meters")
    print("  quit                 - Exit")
    print("\nExample: forward 5")
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not cmd:
                continue
                
            # Validate command format
            parts = cmd.split()
            if len(parts) != 2:
                print("❌ Invalid format. Use: [direction] [distance]")
                continue
                
            direction, distance = parts
            if direction not in ['forward', 'backward', 'left', 'right']:
                print("❌ Invalid direction. Use: forward, backward, left, right")
                continue
                
            try:
                float(distance)
            except:
                print("❌ Invalid distance. Must be a number")
                continue
            
            # Send command
            print(f"📡 Sending: {cmd}")
            publish.single(
                topic="medibot/rover/move",
                payload=cmd,
                hostname=MQTT_HOST,
                port=MQTT_PORT
            )
            print("✅ Command sent! Check subscriber for execution...")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🚀 Simple MQTT Publisher")
    print(f"📡 Broker: {MQTT_HOST}:{MQTT_PORT}")
    print(f"📢 Topic: medibot/rover/move")
    
    print("\nChoose mode:")
    print("1. Send predefined sequence")
    print("2. Interactive mode")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == '1':
            send_movement_sequence()
        elif choice == '2':
            interactive_mode()
        else:
            print("Invalid choice. Running predefined sequence...")
            send_movement_sequence()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()