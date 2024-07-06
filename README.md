# AirSim DQN Training

## Directions
Collect the dinstance sensors installed on the vehicle, input them into the DQN neural network, and learn to operate the drone to complete the visit mission.

## Requirement
### VR environment
* Unreal Engine
* AirSim Plugin (or Colosseum)
### python environment
* python 3.9
```
pip install numpy==1.21.6
pip install opencv-python
pip install msgpack-rpc-python
pip install airsim
pip install matplotlib==3.4.3
pip install gymnasium
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install keyboard
```


## Installation
```cmd
git clone https://github.com/Kura0913/Airsim-DQN.git
```

file tree

```
Airsim-DQN
│  README.md
│  requirements.txt
│  run.py
│  settings.json
│  ShortestPath.py
│  
├─DQN
│      DQNAgent.py
│      DQNNet.py
│      Env.py
│      ReplayBuffer.py
│      
├─execute
│  └─runs
└─Tools
        AirsimTools.py
        DQNTools.py
```
## Usage
### settings.json
The contents of settings.json are as follows:
The vehicle name is drone_1, the category is VehicleType, and 11 distance sensors are installed on the vehicle.

You can adjust the name of the vehicle according to your preference.

You can adjust the location or name of the distance sensor installed according to your own needs, but please note that if you adjust the name or number of distance sensors, you must modify the run.py code.

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "LogMessagesVisible": false,
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 1920,
        "Height": 1080,
        "FOV_Degrees": 90,
        "AutoExposureSpeed": 100,
        "AutoExposureBias": 0,
        "AutoExposureMaxBrightness": 0.64,
        "AutoExposureMinBrightness": 0.03,
        "MotionBlurAmount": 0,
        "TargetGamma": 1.0,
        "ProjectionMode": "",
        "OrthoWidth": 5.12
      },
      {
        "ImageType": 1,
        "Width": 1920,
        "Height": 1080
      },
      {
        "ImageType": 3,
        "Width": 1920,
        "Height": 1080
      },
      {
        "ImageType": 5,
        "Width": 1920,
        "Height": 1080
      }
    ]
  },
  "Vehicles": {
    "drone_1": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "AllowAPIAlways": true,
      "EnableTrace": false,
      "ClockType": "ScalableClock",
      "Cameras": {
        "front_camera": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 960,
              "Height": 540,
              "FOV_Degrees": 90
            }],
            "X": 0.0,
            "Y": 0.0,
            "Z": -0.5,
            "Pitch": 0.0,
            "Roll": 0.0,
            "Yaw": 0.0
        }
      },
      "Sensors":{
        "front": {
        "SensorType": 5,
        "Enabled" : true,
        "X": 0.5, "Y": 0, "Z": 0,
        "Yaw": 0, "Pitch": 0, "Roll": 0,
        "MaxDistance":20,
        "MinDistance": 0
        },
        "rfront": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0.7, "Y": 0.7, "Z": 0,
          "Yaw": 45, "Pitch": 0, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "lfront": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0.7, "Y": -0.7, "Z": 0,
          "Yaw": -45, "Pitch": 0, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "left": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0, "Y": -0.5, "Z": 0,
          "Yaw": -90, "Pitch": 0, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "right": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0, "Y": 0.5, "Z": 0,
          "Yaw": 90, "Pitch": 0, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "top": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0, "Y": 0, "Z": 0,
          "Yaw": 0, "Pitch": 90, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "bottom": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0, "Y": 0, "Z": 0,
          "Yaw": 0, "Pitch": -90, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "rfbottom": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0.7, "Y": 0.7, "Z": 0,
          "Yaw": 0, "Pitch": -90, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "lfbottom": {
          "SensorType": 5,
          "Enabled" : true,
          "X": 0.7, "Y": -0.7, "Z": 0,
          "Yaw": 0, "Pitch": -90, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "rbbottom": {
          "SensorType": 5,
          "Enabled" : true,
          "X": -0.7, "Y": 0.7, "Z": 0,
          "Yaw": 0, "Pitch": -90, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        },
        "lbbottom": {
          "SensorType": 5,
          "Enabled" : true,
          "X": -0.7, "Y": -0.7, "Z": 0,
          "Yaw": 0, "Pitch": -90, "Roll": 0,
          "MaxDistance":20,
          "MinDistance": 0
        }
      }
    }
  },
  "PawnPaths": {
    "BareboneCar": {"PawnBP": "Class'/AirSim/VehicleAdv/Vehicle/VehicleAdvPawn.VehicleAdvPawn_C'"},
    "DefaultCar": {"PawnBP": "Class'/AirSim/VehicleAdv/SUV/SuvCarPawn.SuvCarPawn_C'"},
    "DefaultQuadrotor": {"PawnBP": "Class'/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn_C'"},
    "DefaultComputerVision": {"PawnBP": "Class'/AirSim/Blueprints/BP_ComputerVisionPawn.BP_ComputerVisionPawn_C'"}

  }
}

```

### train
If there is no object corresponding to the object name in the VR environment, training cannot be started.

After the training is completed, a folder named today's date will be generated in 'Airsim-DQN\execute\runs\', which contains the weight of the training.

If you want to terminate the training, press the **p** key in cmd. After completing the current eposide, the training will be terminated and the weight will be stored.

```cmd
python run.py
```

| **parameters**  | **initial** | **directions**                                                                  |
|:---------------:|:-----------:|:--------------------------------------------------------------------------------|
| --episodes      | 5           | training cycle                                                                  |
| --batch_size    | 64          | number of training samples                                                      |
| --gamma         | 0.99        | weight of future reward                                                         |
| --epsilon       | 1.00        | random action rate                                                              |
| --epsilon_min   | 0.2         | epsilon's minimum                                                               |
| --decay         | 0.999       | epsilon's decay rate                                                            |
| --object        | BP_Grid     | eearch object name                                                              |
| --device        | cpu         | cuda or cpu                                                                     |
| --weight        |             | the default value is empty. You can enter the weight path to continue training  |
| --infinite_loop | False       | choose whether to enable infinite training mode                                 |

### test

```cmd
python test.py
```

| **parameters** | **initial** | **directions**                                                                 |
|:--------------:|:-----------:|:-------------------------------------------------------------------------------|
| --episodes     | 5           | testing times                                                                  |
| --object       | BP_Grid     | search object name                                                             |
| --weight       |             | the default value is empty. You need select a weight for testing, otherwise the test cannot be performed.              |