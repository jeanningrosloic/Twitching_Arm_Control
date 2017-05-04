# Twitching_Arm_Control
Twitching neural control of arm movements

TwitchingEnv.py contains the class TwitchingEnv : Muscuskeletal environment (opensim library) Motor command called 'actuators' can be set to the model ('activate_muscle()'), then simulate the muscuskeletal behavior thanks to the 'step()' function, finally sensory informations called sensors are available.

Network.py : The neural network that takes sensors as inputs and return actuators as outputs.

ArmNetwork_v....ipynb : The latest version usually simulates the more recent results

MSA_process.py : Work in progress. Merge neural model and muscuskeletal environment and simulates short single muscle twitchs. The goal is to compare correlation matrix of the muscuskeletal sensors/actuators and the correlation matrix of the neural network.
