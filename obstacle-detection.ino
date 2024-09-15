#include <Servo.h>

// Pins
const int trigPin = 7;
const int echoPin = 6;
const int servoPin = 9;
const int motorPin1 = 2;
const int motorPin2 = 3;
const int motorPin3 = 4;
const int motorPin4 = 5;

// Servo and PID variables
Servo myServo;
double setpoint = 20; // Desired distance from the obstacle
double input, output;
double previousError = 0;
double integral = 0;
double Kp = 1.5; // Proportional gain
double Ki = 0.1; // Integral gain
double Kd = 0.01; // Derivative gain

// Obstacle detection
int distance;
const int scanInterval = 500; // Time between scans
const int scanAngleStep = 30; // Angle step for scanning

void setup() {
  // Initialize pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(motorPin3, OUTPUT);
  pinMode(motorPin4, OUTPUT);

  // Initialize servo
  myServo.attach(servoPin);
  myServo.write(90); // Center position

  Serial.begin(9600);
}

void loop() {
  // Scan surroundings and determine obstacle distance
  int minDistance = 100; // Initialize with a high value
  int minAngle = 90; // Default to center position

  for (int angle = 0; angle <= 180; angle += scanAngleStep) {
    myServo.write(angle);
    delay(scanInterval); // Allow time for sensor reading

    distance = readDistance();
    if (distance < minDistance) {
      minDistance = distance;
      minAngle = angle;
    }
  }

  // Compute PID control based on minimum distance
  input = minDistance;
  double error = setpoint - input;
  integral += error * (scanInterval / 1000.0); // Convert milliseconds to seconds
  double derivative = (error - previousError) / (scanInterval / 1000.0);

  output = Kp * error + Ki * integral + Kd * derivative;
  previousError = error;

  // Adjust vehicle movement based on PID output
  if (output > 0) {
    turnLeft();
  } else if (output < 0) {
    turnRight();
  } else {
    moveForward();
  }

  // Debugging information
  Serial.print("Min Distance: ");
  Serial.print(minDistance);
  Serial.print(" cm, Angle: ");
  Serial.print(minAngle);
  Serial.print(" degrees, PID Output: ");
  Serial.println(output);

  delay(100);
}

int readDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH);
  int distance = duration / 29 / 2;

  return distance;
}

void moveForward() {
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorPin3, HIGH);
  digitalWrite(motorPin4, LOW);
}

void turnLeft() {
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
  digitalWrite(motorPin3, HIGH);
  digitalWrite(motorPin4, LOW);
}

void turnRight() {
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorPin3, LOW);
  digitalWrite(motorPin4, HIGH);
}

void stopMovement() {
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorPin3, LOW);
  digitalWrite(motorPin4, LOW);
}
