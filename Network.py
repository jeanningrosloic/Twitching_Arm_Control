import nest
import numpy as np
import matplotlib.pyplot as plt


class Network():
    
    def __init__(self, time_step, neuron_model="iaf_psc_exp"): #iaf_psc_alpha
        '''
        initialization of the neural network
        Create a set of 6 motor neurons and one of 12 sensory neurons based on the neuron model passed in argument
        Each neuron is connected to a voltmeter and a spike detector to possibly get informations
        A all-to-all connexion is made from sensory neurons to motor neurons through a synapse model described by synapse_dict             passed in argument
        '''
        
        ###__ Nodes Creation __###
        
        self.N_in = 12 # number of sensory neurons
        self.N_out = 6 # number of motor neurons
        # Discret current generators : represents sensory inputs    
        self.generators = nest.Create('dc_generator', self.N_in, {"amplitude": 0.}) # amplitude : [pA]
        self.max_current = 1000. # generator amplitude maximum value
        
        ## Neurons parameters can be set in the neuron dictionary
        # self.neuron_dict = {"I_e": 0.0, "tau_m": 20.0}
        
        ## Creation of motor and sensory neurons
        self.sensory_neurons = nest.Create(neuron_model, self.N_in)#, self.neuron_dict)
        self.motor_neurons = nest.Create(neuron_model, self.N_out)#, self.neuron_dict)
        
        ## Creation of excitatory and inhibitory neurons
        self.N_exc = 1000
        self.N_inh = 200
        self.excitatory_neurons = nest.Create(neuron_model, self.N_exc)
        self.inhibitory_neurons = nest.Create(neuron_model, self.N_inh)
        
        
        ###__ Nodes Connections __###
        #static_syn_dict = {'model': 'static_synapse'}
        #stdp_syn_dict = {'model': 'stdp_synapse'}
        #all_conn_dict = {"rule":"all_to_all"}
        #bernouilli_conn_dict = {'rule':'pairwise_bernoulli', 'p':0.2, 'autapses':True, 'multapses':False}
        
        # Connections from sensory to excitatory neurons
        for neuron in self.sensory_neurons:
            i = neuron-self.sensory_neurons[0]
            nest.Connect((neuron,), self.excitatory_neurons[20*i:20*(i+1)],
                         {"rule":"all_to_all"}, {'model': 'static_synapse'})
        
        # Connections from excitatory to motor neurons
        nest.Connect(self.excitatory_neurons[20*12:], self.motor_neurons,
                     {"rule":"all_to_all"}, {'model':'static_synapse'})
                     
        # Connections from excitatory to excitatory neurons
        nest.Connect(self.excitatory_neurons, self.excitatory_neurons,
                    {'rule':'pairwise_bernoulli', 'p':0.5, 'autapses':True, 'multapses':False},
                     {'model': 'stdp_synapse'})
        
        # Connections from inhibitory to inhibitory neurons
        nest.Connect(self.inhibitory_neurons, self.inhibitory_neurons,
                    {'rule':'pairwise_bernoulli', 'p':0.5, 'autapses':True, 'multapses':False},
                     {'model': 'static_synapse'})
        
        # Connections from excitatory to inhibitory neurons
        nest.Connect(self.excitatory_neurons, self.inhibitory_neurons,
                    {'rule':'pairwise_bernoulli', 'p':0.5, 'autapses':True, 'multapses':False},
                     {'model': 'static_synapse'})
        
        # Connections from inhibitory to excitatory neurons
        nest.Connect(self.inhibitory_neurons, self.excitatory_neurons,
                    {'rule':'pairwise_bernoulli', 'p':0.5, 'autapses':True, 'multapses':False} , 
                     {'model': 'static_synapse'})
                     
        
        # Get connections of interest
        self.sensory_connections = nest.GetConnections(self.sensory_neurons, self.excitatory_neurons, 'static_synapse')
        self.excitatory_connections = nest.GetConnections(self.excitatory_neurons, self.excitatory_neurons, 'stdp_synapse')
        self.motor_connections = nest.GetConnections(self.excitatory_neurons, self.motor_neurons, 'static_synapse')
         
        # Connection from generators to sensory neurons
        nest.Connect(self.generators,self.sensory_neurons,'one_to_one')

        # Multimeters and Spike detectors
        self.motor_multimeter = self.connect_multimeter(self.motor_neurons)
        self.sensory_multimeter = self.connect_multimeter(self.sensory_neurons)
        self.excitatory_multimeter = self.connect_multimeter(self.excitatory_neurons[0:240])
        
        self.motor_spike_detector = self.connect_spike_detector(self.motor_neurons)
        self.sensory_spike_detector = self.connect_spike_detector(self.sensory_neurons)
        self.excitatory_spike_detector = self.connect_spike_detector(self.excitatory_neurons[0:240])
        
        
        ##__ Correlation matrix __##
        self.correlations = nest.Create('correlomatrix_detector')
        nest.SetStatus(self.correlations, {'N_channels': 2, 'tau_max': 2.5})
        nest.Connect(self.sensory_neurons, self.correlations, syn_spec={"receptor_type": 0})
        nest.Connect(self.motor_neurons, self.correlations, syn_spec={"receptor_type": 1})
        
           
        ##__ Filtering parameters __###
        
        #self.input_min = [ 0.10588593,  0.06498628,  0.06065437,  0.0563928 ,  0.05639861,
        #                   0.03930887, -1.15125806, -0.32063974, -0.32063974, -1.32466746,
        #                   -1.38976679, -0.70290913]
        
        #self.input_max = np.array([ 0.19836181,  0.11560998,  0.1139666 ,  0.16298497,  0.19006007,
        #                            0.0937287 ,  1.30701804,  1.51319622,  1.50905595,  2.10230032,
        #                            1.74378887,  1.54169264])
        self.input_min = np.zeros((12,))
        self.input_max = np.ones((12,))
        self.output_max = 1.
        
        ###__ Time __###
        self.time_step = time_step
        self.current_time = 0.
    
    def inputs_filter(self, inputs):
        ''' inputs are the set of muscle length and dlength/dt values got from the muscuskeletal environment.
        First, inputs are projected onto the [0:1] space
        Second, normed inputs are set as a generator value (inputs multiplied by amplitude factor 'max_current')
        '''
        i=0
        for inp in inputs:
            if np.max(inp) > self.input_max[i]:
                self.input_max[i] = np.max(inp)
            if np.min(inp) < self.input_min[i]:
                self.input_min[i] = np.min(inp)
            i+=1
        
        self.normed_inputs = (inputs - self.input_min)/(self.input_max-self.input_min)

        if not len(inputs) == self.N_in:
            print('inputs and sensory neurons should have equal sizes')
        else:
            dicts = []
            for i in range(self.N_in):
                dicts.append({"amplitude": self.normed_inputs[i].astype(float)*self.max_current})
        
        nest.SetStatus(self.generators, dicts)
        
    def outputs_filter(self):
        
        # Compute spike frequency
        freq = np.zeros((self.N_out,))
        for sender in self.motor_spike_senders:
            freq[sender-self.motor_neurons[0]-1] += 1    
        freq = freq/self.time_step
        
        # Norm outputs in the space [0,1] considering that frequencies is positive
        # Note that the maximum frequency (i.e. =1 if normed) is common to the whole set of outputs
        if max(freq) > self.output_max:
            self.output_max = max(freq)
  
        self.normed_outputs = freq/self.output_max
        
        return self.normed_outputs

        
    def connect_multimeter(self, neurons):
        ''' Connect a voltmeter to the neurons (arg)'''
        
        multimeter_dict = {"withtime":True, "record_from":["V_m"]}
        multimeter = nest.Create("multimeter", len(neurons), multimeter_dict)
        nest.Connect(multimeter, neurons)
        return multimeter
    
    def connect_spike_detector(self, neurons):
        '''Connect a spike detector from the neurons (arg)'''
        
        sd_dict = {"withtime":True}
        spike_detector = nest.Create("spike_detector", len(neurons), sd_dict)
        nest.Connect(neurons, spike_detector)
        return spike_detector
        
    def get_membrane_potential(self, multimeter):
        '''Get the multimeter informations after a simulation'''
        
        infos = nest.GetStatus(multimeter)[0]
        membrane_potential = infos["events"]["V_m"]
        time = infos["events"]["times"]
        return membrane_potential, time
    
    def get_spike_times(self, spike_detector):
        '''Get the spike time events after a simulation'''
        
        infos = nest.GetStatus(spike_detector,keys="events")[0]
        sender = infos["senders"]
        times = infos["times"]
        return sender, times
        
    
    def compute_correlation_matrix(self):
              
        stat = nest.GetStatus(self.correlations)[0]
        C = stat["count_covariance"]
        tau_max = stat["tau_max"]
        h = stat["delta_tau"]
        print(C)
        
        m = np.zeros(2, dtype=float)
        for i in range(2):
            m[i] = C[i][i][int(tau_max / h)] * (h / self.time_step)
        print('mean activities =', m)
 
    
    def step(self, inputs):
        ''' Main function. Simulate a step of time_step [ms]'''
        
        
        #nest.SetStatus(self.generators, {"start": self.current_time, "stop": self.current_time + self.time_step})
        
        #Input filtering
        self.inputs_filter(inputs)
        
        # Run the network
        nest.Simulate(self.time_step)
        self.current_time += self.time_step
        
        # Get network infos
        self.sensory_spike_senders, self.sensory_spike_times = self.get_spike_times(self.sensory_spike_detector)
        self.motor_spike_senders, self.motor_spike_times = self.get_spike_times(self.motor_spike_detector)
        self.excitatory_spike_senders, self.excitatory_spike_times = self.get_spike_times(self.excitatory_spike_detector)
        self.sensory_potential, self.sensory_time = self.get_membrane_potential(self.sensory_multimeter)
        self.motor_potential, self.motor_time = self.get_membrane_potential(self.motor_multimeter)
        self.excitatory_potential, self.excitatory_time = self.get_membrane_potential(self.excitatory_multimeter)
        
        # Calculate outputs (based on frequencies of motor neurons)
        outputs = self.outputs_filter()
        
        # Update the synapse weights
        #self.weigths = 
        #C = self.compute_correlation_matrix()
        
        
        #self.figures()
        
        return outputs
        
    def figures(self):
        
        # Sike events
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(3,1,1)
        ax2 = fig1.add_subplot(3,1,2)
        ax3 = fig1.add_subplot(3,1,3)
        
        ax1.scatter(self.sensory_spike_times, self.sensory_spike_senders)
        ax2.scatter(self.motor_spike_times, self.motor_spike_senders)
        ax3.scatter(self.excitatory_spike_times, self.excitatory_spike_senders)
        
        ax1.set_title('Spike events')
        ax2.set_xlabel('Time [s]')
        ax1.set_ylabel('Sensory N indices')
        ax2.set_ylabel('Motor N indices')
        ax3.set_ylabel('Excitatory N indices')
        
        # Membrane potential
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(3,1,1)
        ax2 = fig2.add_subplot(3,1,2)
        ax3 = fig2.add_subplot(3,1,3)
        
        ax1.plot(self.sensory_time, self.sensory_potential)
        ax2.plot(self.motor_time, self.motor_potential)
        ax3.plot(self.excitatory_time, self.excitatory_potential)
        
        ax1.set_title('Membrane potential')
        ax2.set_xlabel('Time [s]')
        ax1.set_ylabel('Voltage [mV]')
        ax2.set_ylabel('Voltage [mV]')
        ax3.set_ylabel('Voltage [mV]')
        
        plt.show()



