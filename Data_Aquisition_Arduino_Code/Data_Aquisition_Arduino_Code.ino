#include "ICM_20948.h" // Click here to get the library: http://librarymanager/All#SparkFun_ICM_20948_IMU

#define USE_SPI       // Uncomment this to use SPI

#define Serial Serial

#define WIRE_PORT Wire // Your desired Wire port.      Used when "USE_SPI" is not defined
// The value of the last bit of the I2C address.
// On the SparkFun 9DoF IMU breakout the default is 1, and when the ADR jumper is closed the value becomes 0
#define AD0_VAL 1//1

ICM_20948_I2C myICM; // Otherwise create an ICM_20948_I2C object

// sEMG
#if defined(ARDUINO) && ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif

#include "EMGFilters.h"

#define TIMING_DEBUG 1

#define SensorInputPin A0 // input pin number

int accXmss = 0;
int accYmss = 0;
int accZmss = 0;

EMGFilters myFilter;

int sampleRate = SAMPLE_FREQ_1000HZ;

int humFreq = NOTCH_FREQ_50HZ;


unsigned long timeStamp;
unsigned long timeBudget;
unsigned long timeinit;
unsigned long inittime;


// ====================
void setup() {
myFilter.init(sampleRate, humFreq, true, true, true);
  Serial.begin(115200);
  //Serial.println("one");
  timeBudget = 1e6 / sampleRate;
  WIRE_PORT.begin();
  //WIRE_PORT.begin(0x69);
  WIRE_PORT.setClock(400000);
  delay(500);
  //Serial.println("two");

    bool initialized = false;
      
      //Serial.println("three");
      myICM.begin(WIRE_PORT, AD0_VAL);
      //Serial.print(F("Initialization of the sensor returned: "));
      //Serial.println(myICM.statusString());
      if (myICM.status != ICM_20948_Stat_Ok)
      {
        //Serial.println("Trying again...");
        delay(500);
      }
      else
      {
      initialized = true;
      delay(500);
      //Serial.println("IMU Connected");
      }
      
    
  bool success = true; // Use success to show if the DMP configuration was successful

  // Initialize the DMP. initializeDMP is a weak function. You can overwrite it if you want to e.g. to change the sample rate
  success &= (myICM.initializeDMP() == ICM_20948_Stat_Ok);

  success &= (myICM.enableDMPSensor(INV_ICM20948_SENSOR_GAME_ROTATION_VECTOR) == ICM_20948_Stat_Ok); //Uncomment

  success &= (myICM.setDMPODRrate(DMP_ODR_Reg_Quat6, 0) == ICM_20948_Stat_Ok); // Set to the maximum ///uncomment

  success &= (myICM.enableFIFO() == ICM_20948_Stat_Ok);

  // Enable the DMP
  success &= (myICM.enableDMP() == ICM_20948_Stat_Ok);

  // Reset DMP
  success &= (myICM.resetDMP() == ICM_20948_Stat_Ok);

  // Reset FIFO
  success &= (myICM.resetFIFO() == ICM_20948_Stat_Ok);

  //Serial.println("end void setup");

  //Serial.println(F("Press any key to continue"));

  //while (!Serial.available()) // Wait for the user to press a key (send any serial character)
    //;
}

// =====================
void loop() {
 // Serial.print("Time Stamp, RAW ACCEL X, RAW ACCEL Y, RAW ACCEL X, RAW GYRO X, RAW GYRO Y, RAW GYRO Z, EMG4ARM, EMGBICP, EMG SHLD ");
 // Serial.println();

  //timeinit = 0;
  inittime = 0;
  timeinit = micros();
  Serial.print(inittime);

  EMG_ONE();
  delayMicroseconds(500);

  Serial.flush();
  Serial.end();
  Serial.begin(115200);
  
  //Serial.println(F("Time Stamp, RAW ACCEL X, RAW ACCEL Y, RAW ACCEL X, RAW GYRO X, RAW GYRO Y, , RAW GYRO Z, ROT POS X, , ROT POS Y, ROT POS Z, EMG4ARM, EMGBICP, EMG SHLD "));

  //while (!Serial.available()) // Wait for the user to press a key (send any serial character)
    //;
}


// =====================

void EMG_ONE() {
for (int j = 0; j <749; j++)  // about 230 samples per second 150 per second  1050 7 seconds
{ 
  timeStamp = micros() - timeinit;
  Serial.print(timeStamp);
  Serial.print(",");

  IMU_ONE();
  Serial.flush();
  //IMU_TWO();
  Serial.flush();
  //Serial.print("EMG:Forearm Bicep Shoulder");
  //Serial.print(",");
  
  for (int analogPin = 0; analogPin < 3; analogPin++)
  {
    
  
    // timeStamp = micros() - timeStamp
    int Value = analogRead(analogPin);

    // filter processing
    int DataAfterFilter = myFilter.update(Value);
    //int envlope = sq(DataAfterFilter);

    int envlope = Value;
    int envlope_one;
    int envlope_two;
    int envlope_three;

      if (analogPin == 0) {
      envlope_one = Value;
    } else if (analogPin == 1) {
      envlope_two = Value;
    } else if (analogPin == 2) {
      envlope_three = Value;
    }
  
    //timeStamp = micros() - timeStamp;
    //if (TIMING_DEBUG) {
      if (analogPin == 0) {
        Serial.print(envlope_one);
      } else if (analogPin == 1) {
        Serial.print(envlope_two);
      } else if (analogPin == 2) {
        Serial.print(envlope_three);
      }
  
  
  if (analogPin < 2) {
    Serial.print(",");
    }
  else {
    Serial.println();
    }   
   }
  }

  Serial.flush();

}

// =======================================

void IMU_ONE()
{
  myICM.getAGMT();         // The values are only updated when you call 'getAGMT'
  printRawAGMT( myICM.agmt );     // Uncomment this to see the raw values, taken directly from the agmt structure
  //printScaledAGMT(&myICM); // This function takes into account the scale settings from when the measurement was made to calculate the values with units
  //Serial.print("IMUONE");
}


void printRawAGMT(ICM_20948_AGMT_t agmt)
{
  Serial.print(agmt.acc.axes.x);
  Serial.print(",");
  Serial.print(agmt.acc.axes.y);
  Serial.print(",");
  Serial.print(agmt.acc.axes.z);
  Serial.print(",");
  Serial.print(agmt.gyr.axes.x);
  Serial.print(",");
  Serial.print(agmt.gyr.axes.y);
  Serial.print(",");
  
  Serial.print(agmt.gyr.axes.z);
  Serial.print(",");
}

void printScaledAGMT(ICM_20948_I2C *sensor)
{
  Serial.print(sensor->gyrX());
  Serial.print(",");
  Serial.print(sensor->gyrY());
  Serial.print(",");
  Serial.print(sensor->gyrZ());
  Serial.print(",");
  //Serial.print(",");
  //Serial.println();
}

void IMU_TWO()
{
  icm_20948_DMP_data_t data;
  myICM.readDMPdataFromFIFO(&data);

  if ((myICM.status == ICM_20948_Stat_Ok) || (myICM.status == ICM_20948_Stat_FIFOMoreDataAvail)) // Was valid data available?
  {

    if ((data.header & DMP_header_bitmap_Quat6) > 0) // We have asked for GRV data so we should receive Quat6
    {
      double q1 = ((double)data.Quat6.Data.Q1) / 1073741824.0; // Convert to double. Divide by 2^30
      double q2 = ((double)data.Quat6.Data.Q2) / 1073741824.0; // Convert to double. Divide by 2^30
      double q3 = ((double)data.Quat6.Data.Q3) / 1073741824.0; // Convert to double. Divide by 2^30

      double q0 = sqrt(1.0 - ((q1 * q1) + (q2 * q2) + (q3 * q3)));

      double q2sqr = q2 * q2;

      // roll (x-axis rotation)
      double t0 = +2.0 * (q0 * q1 + q2 * q3);
      double t1 = +1.0 - 2.0 * (q1 * q1 + q2sqr);
      double roll = atan2(t0, t1) * 180.0 / PI;

      // pitch (y-axis rotation)
      double t2 = +2.0 * (q0 * q2 - q3 * q1);
      t2 = t2 > 1.0 ? 1.0 : t2;
      t2 = t2 < -1.0 ? -1.0 : t2;
      double pitch = asin(t2) * 180.0 / PI;

      // yaw (z-axis rotation)
      double t3 = +2.0 * (q0 * q3 + q1 * q2);
      double t4 = +1.0 - 2.0 * (q2sqr + q3 * q3);
      double yaw = atan2(t3, t4) * 180.0 / PI;

#ifndef QUAT_ANIMATION
      //Serial.print(F("Roll:"));
      //Serial.print("IMU Rotary Position");
      //Serial.print(",");
      Serial.print(roll, 1);
      Serial.print(",");
      //Serial.print(F(" Pitch:"));
      Serial.print(pitch, 1);
      Serial.print(",");
      //Serial.print(F(" Yaw:"));
      Serial.print(yaw, 1);
      Serial.print(",");
      //Serial.print(",");
      //Serial.println();
#else
      // Output the Quaternion data in the format expected by ZaneL's Node.js Quaternion animation tool
      Serial.print(F("{\"quat_w\":"));
      Serial.print(q0, 3);
      Serial.print(F(", \"quat_x\":"));
      Serial.print(q1, 3);
      Serial.print(F(", \"quat_y\":"));
      Serial.print(q2, 3);
      Serial.print(F(", \"quat_z\":"));
      Serial.print(q3, 3);
      Serial.println(F("}"));
#endif
    }
  }
  
}


