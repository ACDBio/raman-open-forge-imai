#include <Stepper.h>

#define STEPS 200
#define BUF_SIZE 64

Stepper stepperx(STEPS, 23, 19, 18, 26);
Stepper steppery(STEPS, 16, 17, 22, 21);

int speedX = 60;
int speedY = 100;
int stepSize = 150;

char buf[BUF_SIZE];
int bufPos = 0;

void setup() {
  Serial.begin(9600);

  stepperx.setSpeed(speedX);
  steppery.setSpeed(speedY);

  pinMode(23, OUTPUT);
  pinMode(18, OUTPUT);
  pinMode(19, OUTPUT);
  pinMode(26, OUTPUT);
  pinMode(16, OUTPUT);
  pinMode(17, OUTPUT);
  pinMode(22, OUTPUT);
  pinMode(21, OUTPUT);

  motorxStop();
  motoryStop();

  Serial.println("STAGE_READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (bufPos > 0) {
        buf[bufPos] = '\0';
        processCommand(buf);
        bufPos = 0;
      }
    } else {
      if (bufPos < BUF_SIZE - 1) {
        buf[bufPos++] = c;
      }
    }
  }
}

// Simple prefix check
bool startsWith(const char* str, const char* prefix) {
  while (*prefix) {
    if (*str != *prefix) return false;
    str++;
    prefix++;
  }
  return true;
}

// Get integer after prefix (with or without space)
// "MOVE_X_POS" -> defaultVal
// "MOVE_X_POS 500" -> 500
// "MOVE_X_POS500" -> 500
int getArg(const char* cmd, int prefixLen, int defaultVal) {
  const char* p = cmd + prefixLen;
  while (*p == ' ') p++;  // skip spaces
  if (*p == '\0') return defaultVal;
  return atoi(p);
}

void processCommand(const char* cmd) {

  if (startsWith(cmd, "MOVE_X_POS")) {
    int s = getArg(cmd, 10, stepSize);
    stepperx.step(s);
    motorxStop();
    Serial.print("OK MXP ");
    Serial.println(s);
  }
  else if (startsWith(cmd, "MOVE_X_NEG")) {
    int s = getArg(cmd, 10, stepSize);
    stepperx.step(-s);
    motorxStop();
    Serial.print("OK MXN ");
    Serial.println(s);
  }
  else if (startsWith(cmd, "MOVE_Y_POS")) {
    int s = getArg(cmd, 10, stepSize);
    steppery.step(s);
    motoryStop();
    Serial.print("OK MYP ");
    Serial.println(s);
  }
  else if (startsWith(cmd, "MOVE_Y_NEG")) {
    int s = getArg(cmd, 10, stepSize);
    steppery.step(-s);
    motoryStop();
    Serial.print("OK MYN ");
    Serial.println(s);
  }
  else if (startsWith(cmd, "SET_SPEED_X")) {
    int v = getArg(cmd, 11, -1);
    if (v > 0 && v <= 300) {
      speedX = v;
      stepperx.setSpeed(speedX);
      Serial.print("OK SSX ");
      Serial.println(speedX);
    } else {
      Serial.println("ERR SSX");
    }
  }
  else if (startsWith(cmd, "SET_SPEED_Y")) {
    int v = getArg(cmd, 11, -1);
    if (v > 0 && v <= 300) {
      speedY = v;
      steppery.setSpeed(speedY);
      Serial.print("OK SSY ");
      Serial.println(speedY);
    } else {
      Serial.println("ERR SSY");
    }
  }
  else if (startsWith(cmd, "SET_STEP_SIZE")) {
    int v = getArg(cmd, 13, -1);
    if (v > 0 && v <= 5000) {
      stepSize = v;
      Serial.print("OK SS ");
      Serial.println(stepSize);
    } else {
      Serial.println("ERR SS");
    }
  }
  else if (startsWith(cmd, "PING")) {
    Serial.println("PONG");
  }
  else if (startsWith(cmd, "STOP")) {
    motorxStop();
    motoryStop();
    Serial.println("OK STOP");
  }
  else if (startsWith(cmd, "GET_STATUS")) {
    Serial.print("ST ");
    Serial.print(speedX);
    Serial.print(" ");
    Serial.print(speedY);
    Serial.print(" ");
    Serial.println(stepSize);
  }
  else {
    Serial.print("ERR ");
    Serial.println(cmd);
  }
}

void motorxStop() {
  digitalWrite(23, LOW);
  digitalWrite(18, LOW);
  digitalWrite(19, LOW);
  digitalWrite(26, LOW);
}

void motoryStop() {
  digitalWrite(16, LOW);
  digitalWrite(17, LOW);
  digitalWrite(22, LOW);
  digitalWrite(21, LOW);
}