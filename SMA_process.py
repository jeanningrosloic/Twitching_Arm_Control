class SMA_process():
    ''' SMA_process simulates the short single muscle twitchs that enables a first learning of correlations between sensors and actuators activity'''
    
    def __init__(self, muscuskeletal_environment, network_model):
        
        self.env = muscuskeletal_environment
        self.net = network_model
        self.time_step = 10. # [msec]
        
        self.current_time = 0. # [msec]
        #self.twitch_duration = 10. # [msec]
        
        self.N_muscles = 6
        self.input_sequence = np.zeros((3500,self.N_muscles))
        for n in range(self.N_muscles):
            self.input_sequence[500 + i*500, i] = 0.99
            
        
        
    def step(self):
        
        for inputs in self.input_sequence:
            env.step([0]) # To do : modify step
            sensors = env.sensorsValues
            actuators = env.actuatorsValues
            action = net.step(inputs)
    
            
        
        