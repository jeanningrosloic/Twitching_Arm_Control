class TwitchingEnv(ArmEnv):
    def __init__(self, visualize = False, musclesName = '', iterDuration = 4, twitchDuration = 0.05, stepsize = 0.01):
        
        print("Parameters copy...")
        self.model_path = './arm2dof6musc.osim'
        folder = ''
        self.musclesName = musclesName
        self.iterNumber = 0
        
        self.iterDuration = iterDuration # How many iterations 

        self.twitchDuration = twitchDuration
        #self.twitchStartPercentage = 0 # not used
        
        self.n_iter_waiting = 5

        self.twitchingEpisodeEnded = True
        
        print("Files loading...")
        self.sensorsFile = open('sensors_arm.csv', 'w')
        self.actuatorsFile = open('actuators_arm.csv', 'w')

        self.musclesLength = [ 0 for i in range(len(musclesName))]
        self.musclesLengthConverged = [ 0 for i in range(len(musclesName))]

        print("Headers wirting...")

        self.writeHeaders([m + '_len' for m in self.musclesName] + [m + '_dlen' for m in self.musclesName],self.sensorsFile)
        self.writeHeaders([m + '_a' for m in self.musclesName],self.actuatorsFile)
        
        print("GaitEnv __init__ calling...")
        super(TwitchingEnv, self).__init__(visualize = visualize)
        
        
        
    def get_observation(self):
        super(TwitchingEnv, self).get_observation()

        #m1_length = self.osim_model.model.getActuators().get(1).getStateVariableValues(self.osim_model.state).get(1);
        #m1_name = self.osim_model.model.getActuators().get(1).getStateVariableNames().get(1)
        
    def get_values(self):
        # Get input / outputs from open sim
        musclesLength_new = [self.osim_model.model.getActuators().get(m).getStateVariableValues(self.osim_model.state).get(1)
                             for m in self.musclesName]
        musclesdLength = [ (x-y)/self.stepsize 
                          for x,y in zip( musclesLength_new, self.musclesLength) ]
        
        #values of the actuators
        actuatorsValues = [self.osim_model.model.getActuators().get(m).getStateVariableValues(self.osim_model.state).get(0) 
                           for m in self.musclesName]

        #musclesLength: length, musclesdLength: speed (d for differentiation ?)
        sensorsValues = musclesLength_new + musclesdLength 
            
        self.musclesLength = musclesLength_new
        
        return actuatorsValues, sensorsValues
        

    def writeHeaders(self,variable_names,file):
        file.write("{}\n".format(",".join(variable_names)))
    def writeContent(self,variables,file):
        file.write("{}\n".format(",".join([str(x) for x in variables])))


    def activate_muscles(self, action, twitch, muscleNum): # the argument action is not used
        muscleSet = self.osim_model.model.getMuscles()
        for j in range(self.noutput):
            muscle = muscleSet.get(j)
            if j == muscleNum and twitch == True:
                muscle.setActivation(self.osim_model.state, 0.99) # action ?
            
            #muscle = muscleSet.get(j)
            #muscle.setActivation(self.osim_model.state, action)
    
    def compute_reward(self):       
        #reward = np.norm(musclesLength - targetsLength)
        #reward = reward - self.meanReward
        #self.meanReward = self.meanReward*(N-1)/N
        
        return 0
    
    def _step(self, action):
        
        self.last_action = action

        totalTime = self.istep*self.stepsize # where are initialized istep and stepsize ? But it works
        iter_time = math.fmod(totalTime,self.iterDuration) # what's the current iteration within an episode
        
        iter_episode = int(totalTime/self.iterDuration) # what's the current episode
        
        self.muscleNumber = iter_episode - self.n_iter_waiting # which muscle has been twitched
        
        
        #no muscle is activated
        if self.muscleNumber < 0:
            self.activate_muscles(action,False,0)
        else:
            if iter_time == 0:
                print("Beginning of twitching for muscle number " + str(int(self.muscleNumber)))
            if iter_time == self.twitchDuration:
                print("End of stimulation " + str(self.muscleNumber))
            if(iter_time <= self.twitchDuration):
                self.activate_muscles(action,True,self.muscleNumber)
            else:
                self.activate_muscles(action,False,self.muscleNumber) 
                # actually, if twitch=False, the fct activate muscle does nothing
                
        # get sensorsValues and actuatorsValues, also update self.musclesLength
        self.actuatorsValues, self.sensorsValues = self.get_values()
        
        # Logging
        self.writeContent(self.sensorsValues,self.sensorsFile)
        self.writeContent(self.actuatorsValues,self.actuatorsFile)

        # Integrate one step
        self.osim_model.manager.setInitialTime(self.stepsize * self.istep)
        self.osim_model.manager.setFinalTime(self.stepsize * (self.istep + 1))

        try:
            self.osim_model.manager.integrate(self.osim_model.state)
        except Exception:
            print ("Exception raised")
            return self.get_observation(), -500, True, {}

        self.istep = self.istep + 1
        #TT = self.osim_model.model.getActuators().get(1)

        res = [ self.get_observation(), self.compute_reward(), self.is_done(), {} ]
        return res