#參考自 https://blog.csdn.net/apple_51522252/article/details/125010239
import sys
import traci
import time
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
class Sim(object):
    def __init__(self, sumo_config, GUI=False):
        self.sumo_config = sumo_config 
        self.n = 10
        self.launch_env_flag = False
        self.GUI = GUI
        self.n_observation = 17
        self.n_action = 4
        self.state_lane = ["E0_0","E0_1","E0_2","E1_0","E1_1","E1_2","E1_3","E1_4","E2_0","E2_1","E2_2","-E3_0","-E3_1","-E3_2","-E3_3","-E3_4"]
        #self.state_lane = ["E0_0","E0_1","E1_0","E1_1","E2_0","E2_1","-E3_0","-E3_1"]
        self.durations = [20,25,30,45]
        self.entry_exit_times = {}
        self.waiting_times = {}
        self.traveltimes=[]
    def launchEnv(self):
        
        if self.GUI:
            sumo_gui = 'sumo-gui'
        else:
            sumo_gui = 'sumo'
        #traci.start([
        #    sumo_gui,
        #    "-c", self.sumo_config,
        #    "--no-warnings",
        #    "--seed", "2"])
        traci.start([
            sumo_gui,
            "-c", self.sumo_config,
            "--no-warnings",
            "--seed","2"])
        self.launch_env_flag = True
    def close(self):
        
        traci.close()
        self.launch_env_flag = False
        sys.stdout.flush()
    def reset(self):
        mean_flow = [2500,2978,1964,2078]
        std_deviation = 100
        portion = [[0.035,0.9,0.065],[0.048,0.855,0.097],[0.07,0.88,0.04],[0.12,0.83,0.05]]
        entry_edges = ["E2", "E0", "E1", "-E3"]
        end_points  = [['E3','-E0','-E1'],['-E1','-E2','E3'],['-E2','E3','-E0'],['-E0','-E1','-E2']]
        # 生成路徑文件的根元素
        root = ET.Element("routes")
        num=10
        # 生成不同入口道的車輛流量
        vtype_element = ET.SubElement(root, "vType", id="Car", vClass="passenger", color="1,1,1", maxSpeed="31.06")
        for i in range(len(entry_edges)):
            random_flow = int(np.random.normal(0.1*mean_flow[i], std_deviation))
            random_flow = max(random_flow,100)
            # 使用隨機數生成車流量
            
            
            # 創建 flow 元素
            for j in range(0,3):
                flow_id = f"F{num}"
                flow_element = ET.SubElement(root, "flow",attrib={"id": flow_id, "type": "Car", "begin": "0", "end": "3600", "vehsPerHour": str(portion[i][j]*random_flow), "color": "1,1,1", "from": entry_edges[i], "to": end_points[i][j]})
                #time.sleep(0.01)
                num = num + 1
        # 保存生成的路徑文件
        xml_str = ET.tostring(root, encoding="utf-8").decode()
        xml_formatted = minidom.parseString(xml_str).toprettyxml(indent="    ")

        # 保存生成的路徑文件
        with open("one_intersection_left/one_intersection_random.rou.xml", "w", encoding="utf-8") as f:
            f.write(xml_formatted)
    def step(self,action = 0):
        step=0
        state = []
        ql = 0
        duration =  self.durations[action]
        traci.trafficlight.setPhaseDuration('J1',duration)
        while step<duration + 4:
            traci.simulationStep() 
            step=step+1
            arrived_vehicles = traci.simulation.getArrivedIDList()
            departed_vehicles = traci.simulation.getDepartedIDList()
            #print(traci.simulation.getCurrentTime())
            # Record entry time for newly arrived vehicles
            for vehicle_id in departed_vehicles:
                self.entry_exit_times[vehicle_id] = traci.simulation.getCurrentTime()
            # Record exit time for departed vehicles
            for vehicle_id in arrived_vehicles:
                if vehicle_id in self.entry_exit_times:
                    entry_time = self.entry_exit_times[vehicle_id]
                    exit_time = traci.simulation.getCurrentTime()
                    #print(f"Vehicle {vehicle_id}: Entered at {entry_time} ms, Exited at {exit_time} ms,travel time{exit_time-entry_time} ms")
                    self.traveltimes.append(exit_time-entry_time)
                     
            # Get the list of waiting vehicles at each simulation step
            all_vehicles = traci.vehicle.getIDList()
            # Update waiting times for each waiting vehicle
            for vehicle_id in all_vehicles:
                if vehicle_id not in self.waiting_times:
                    self.waiting_times[vehicle_id] = 0

                # Get the current waiting time for the vehicle
                current_waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                self.waiting_times[vehicle_id] += current_waiting_time
        for i in self.state_lane:   
                state.append(traci.lane.getLastStepVehicleNumber(i))
        for i in self.state_lane:
            ql = ql + traci.lane.getLastStepHaltingNumber(i)
        state.append(int(traci.trafficlight.getPhase("J1")))
        reward = -ql
        terminated = not(traci.simulation.getMinExpectedNumber() > 0)
        return state,reward,terminated
    def runSim(self):
        state = []
        self.launchEnv()  
        self.step()  
        for i in self.state_lane:   
                state.append(traci.lane.getLastStepVehicleNumber(i))
        #self.close()  
        state.append(int(traci.trafficlight.getPhase("J1")))
        return state
         

    
    
