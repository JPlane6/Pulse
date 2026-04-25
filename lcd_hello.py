from RPLCD.i2c import CharLCD
import time

lcd = CharLCD('PCF8574', 0x27, cols=20, rows=4)

def hello():
    try:
        lcd.clear()
        lcd.cursor_pos = (1, 0)
        lcd.write_string("HELLO AYUSH SHARMA!")
        lcd.cursor_pos = (2, 0)
        lcd.write_string("This is PULSE!")
        # Move to the third line for a status update
        lcd.cursor_pos = (3, 0)
        lcd.write_string("oh & daivik i guess")
        

    except Exception as e:
        print(f"Could not connect to LCD: {e}")

if __name__ == "__main__":
    hello() 