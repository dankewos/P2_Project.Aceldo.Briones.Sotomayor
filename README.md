# P2_Project - F1TENTH simulator. Reactive Controllers: Follow the Gap and Wall Following

## General Description

This project implements two **reactive controllers** for an F1TENTH autonomous vehicle in a simulated environment with ROS 2. The approaches used are the **Follow the Gap (FTG)** and **Wall Following (WF)** control algorithms. FTG is an algorithm that analyzes LiDAR data to identify the safest space to move forward and adjust the turning angle and speed in real time. On the other hand, WF is an algorithm that seeks to keep the vehicle at a certain fixed distance from a wall. In addition, a **lap counting** and **timing** system is integrated, which automatically detects when the vehicle completes a lap and records its duration.

---

## Instructions to run the project

### 1. F1Tenth simulator

Before starting it's necessary that you install the simulator. You can find a tutorial on this repository: https://github.com/widegonz/F1Tenth-Repository

### 2. Clone repository

Now that you already have the dependencies, and all, you can clone this actual repository in /home.

```bash
git clone https://github.com/dankewos/P2_Project.Aceldo.Briones.Sotomayor.git
```

### 2. Access to the repository location

```bash
cd F1Tenth-Repository
```
### 3. Compile the workspace

```bash
colcon build
source install/setup.bash
```

### 4. Run the simulator and the controller
 First, we launch the simulator. You must see the enviroment in Rviz.
 
 ```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

 Then, there's going to be four different controllers that you can run depending on the algorithm. 
 
```bash
# This one is a simple FTG algorithm, used to explore the map without obstacles
ros2 run controllers proyectoFTG

# This is a WF algorithm, used in the map without obstacles as well
ros2 run controllers proyectoWF

# Now this is a FTG algorithm that uses a PID controller to improve the performance of the algorithm
ros2 run controllers obstaculosFTG

# THis one is a WF algorithm that also uses a PID controller, a little bit more strict than the regular algorithm
ros2 run controllers obstaculosWF 
```

### 5. Change the map (Optional)
The map that shows up by default is going to be the Spielberg obstacle map, so if you want to see the Spielberg map you must do some changes in the file 'sim.yaml'

```bash
cd F1Tenth-Repository/src/controllers/f1tenth_gym_ros/config
nano sim.yaml
#On line 45 change this
map_path: '/home/maria/F1Tenth-Repository/src/f1tenth_gym_ros/maps/Spielberg_obs_map'
#For this
map_path: '/home/maria/F1Tenth-Repository/src/f1tenth_gym_ros/maps/Spielberg_map'
```
---
## Results

You will see the car moving around the selected map

---


## 🎥 Youtube Playlist with all the examples

[![YouTube Playlist](https://img.shields.io/badge/YouTube-Playlist-red?logo=youtube)](https://www.youtube.com/playlist?list=PLidPDTG67PkgnYUemQukS1OT_5GfH3HCp)

Authosr: Aceldo Grazia, Briones Bryan, Sotomayor Marcelo

Course: Mobile and Serial Robots

College: Escuela Superior Politecnica del Litoral
