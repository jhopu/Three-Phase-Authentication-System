# Obstacle Detection and Avoidance Vehicle

## Project Overview

This project focuses on developing an autonomous vehicle equipped with an obstacle detection and avoidance system. The vehicle uses an ultrasonic sensor mounted on a servo motor to scan its environment, combined with a custom PID (Proportional-Integral-Derivative) controller to navigate around obstacles. The PID controller helps in making smooth and adaptive adjustments to the vehicle's direction to ensure it avoids collisions and navigates effectively.

![image](https://github.com/user-attachments/assets/5a3d9188-c33a-479c-84d4-e0b529459209)

## Components Used

### Hardware

- **Arduino Board:** The central controller for reading sensor data, performing PID computations, and controlling the motors.
- **Ultrasonic Sensor (HC-SR04):** Measures the distance between the sensor and obstacles. This sensor is mounted on a servo motor to provide a 360-degree scanning capability.
- **Servo Motor (SG90 or similar):** Rotates the ultrasonic sensor to scan the environment in a 180-degree range.
- **Motor Driver (L298N or similar):** Controls the direction and speed of the DC motors based on the signals from the Arduino.
- **DC Motors:** Provide propulsion for the vehicle, allowing it to move forward, backward, and turn.

### Software

- **Arduino IDE:** The integrated development environment used for writing, compiling, and uploading the code to the Arduino board.

## Features

- **Obstacle Detection:** The ultrasonic sensor scans the environment by rotating and measuring distances to the nearest obstacles.
- **PID Control:** A custom PID controller calculates the necessary adjustments to the vehicle's direction based on obstacle proximity.
- **Adaptive Movement:** The vehicle adjusts its movement in real-time, deciding whether to move forward, turn left, or turn right based on the PID output.

## Project Functionality

### Overview

1. **Scanning:** The servo motor rotates the ultrasonic sensor to scan for obstacles at different angles.
2. **Distance Measurement:** The ultrasonic sensor measures the distance to the closest obstacle in each scanned direction.
3. **PID Computation:** The PID controller processes the distance measurements to determine the necessary steering adjustments.
4. **Movement Control:** The vehicle’s motors are controlled based on the PID output to navigate around obstacles.

### Code Explanation

#### Key Functions

- **`setup()`**
  - Initializes the pins for the ultrasonic sensor, servo motor, and motor driver.
  - Attaches the servo motor and sets its initial position.
  - Starts serial communication for debugging purposes.

- **`loop()`**
  - Scans the environment in a 180-degree range to detect obstacles.
  - Computes the distance to the nearest obstacle and the angle at which it is detected.
  - Uses the PID controller to calculate the appropriate output for steering adjustments.
  - Controls the vehicle's movement based on the PID output to avoid obstacles.

- **`readDistance()`**
  - Sends a pulse from the ultrasonic sensor to measure the distance to an obstacle.
  - Returns the distance in centimeters.

- **Movement Functions**
  - **`moveForward()`**: Activates the motors to move the vehicle forward.
  - **`turnLeft()`**: Activates the motors to turn the vehicle left.
  - **`turnRight()`**: Activates the motors to turn the vehicle right.
  - **`stopMovement()`**: Stops all motor activity.

#### PID Control

The PID control algorithm is used to adjust the vehicle's direction based on the distance measurements from the ultrasonic sensor:

- **Proportional (Kp)**: Reacts to the current distance error. The higher the error, the more corrective action is applied.
- **Integral (Ki)**: Accumulates past errors to correct systematic biases. It helps in addressing persistent deviations.
- **Derivative (Kd)**: Considers the rate of change of the error to smooth out the response and prevent oscillations.

The PID controller adjusts the vehicle’s steering to maintain a desired distance from obstacles by continuously calculating and applying corrections.

## Wiring Instructions

1. **Ultrasonic Sensor:**
   - Connect the **Trig** pin to digital pin **7** on the Arduino.
   - Connect the **Echo** pin to digital pin **6** on the Arduino.

2. **Servo Motor:**
   - Connect the servo’s signal pin to digital pin **9** on the Arduino.

3. **Motor Driver:**
   - Connect the **IN1** pin to digital pin **2** on the Arduino (Left Motor Forward).
   - Connect the **IN2** pin to digital pin **3** on the Arduino (Left Motor Backward).
   - Connect the **IN3** pin to digital pin **4** on the Arduino (Right Motor Forward).
   - Connect the **IN4** pin to digital pin **5** on the Arduino (Right Motor Backward).

## Usage

1. **Wiring:**
   - Ensure all components are connected as described above.

2. **Upload Code:**
   - Open the Arduino IDE and upload the provided code to your Arduino board.

3. **Testing:**
   - Power up the Arduino and observe the vehicle’s behavior. The vehicle should autonomously navigate around obstacles by adjusting its direction based on the distance measurements.

## Customization

- **PID Tuning:** Adjust the `Kp`, `Ki`, and `Kd` constants to fine-tune the PID controller for your specific setup.
- **Scan Settings:** Modify `scanInterval` and `scanAngleStep` for different scanning frequencies and resolutions.
- **Movement Functions:** Customize the motor control functions to match your vehicle’s specific movement characteristics.

## Dependencies

- **Arduino IDE:** Required for programming the Arduino board.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.


